import os
import itertools
import copy

import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from sys import exit

class Order():

    def __init__(self, asset, level_entry_signal, level_entry, time_entry, position, quantity):
        self.asset = asset
        self.level_entry_signal = level_entry_signal
        self.level_entry = level_entry
        self.level_exit_signal = level_entry_signal
        self.level_exit = level_entry
        self.pnl = 0
        self.time_entry = time_entry
        self.time_exit = time_entry
        self.position = position
        self.quantity = quantity
        self.transaction_costs = self.compute_tc()

    def compute_pnl(self):
        self.pnl = ((self.level_exit - self.level_entry) * self.quantity) - self.compute_bf()
        return self.pnl

    def compute_tc(self):
        amount = self.level_entry * self.quantity
        tc = min(max(1, self.quantity * self.asset.transaction_costs) / amount, 0.01) * amount
        return tc
    
    def compute_bf(self):
        if self.position == -1 :
            date_format = '%Y-%m-%d %H:%M:%S'
            days = datetime.strptime(self.time_exit, date_format) - datetime.strptime(self.time_entry, date_format)
            amount = self.level_entry * self.quantity
            bc =  self.asset.borrowing_fees * (days.total_seconds()/(360 * 86400)) * amount
            return bc
        else:
            return 0


class Asset(object):

    def __init__(self, name, prices):
        self.name = name
        self.prices = prices
        self.order = None
        self.position = 0
        self.returns_mean = 0
        self.returns_std = 0
        self.transaction_costs = 0.005
        self.borrowing_fees = 0
        
        if "date" in prices.columns: self.prices.set_index('date', inplace=True)

        self.ibapi_asset = None

    def compute_next_open(self, index):
        loc = self.prices.index.get_loc(index)
        next_open = self.prices.iloc[min(loc + 1, len(self.prices.index)-1)].open
        return next_open
    
    def compute_slippage(self, intensity=0):

        log_returns = np.log(self.prices.close).diff()
        slippage = np.exp(log_returns.std()) -1 

        return slippage * intensity
    
    def long_order(self, level_entry_signal, time_entry, quantity, intensity=0, simulated=False):

        level_entry = level_entry_signal * (1 + self.compute_slippage(intensity))
        order = Order(self, level_entry_signal, level_entry, time_entry, 1, quantity)

        if not simulated:
            self.position = 1
            self.order = order

        return order
    
    def short_order(self, level_entry_signal, time_entry, quantity, intensity=0, simulated=False):

        level_entry = level_entry_signal * (1 - self.compute_slippage(intensity))
        order = Order(self, level_entry_signal, level_entry, time_entry, -1, quantity)

        if not simulated:
            self.position = -1
            self.order = order

        return order
    
    def close_position(self, level_exit_signal, time_exit, intensity=0, simulated=False):

        if self.position == 1:
            level_exit = level_exit_signal * (1 - self.compute_slippage(intensity))
        elif self.position == -1:
            level_exit = level_exit_signal * (1 + self.compute_slippage(intensity))
        self.order.level_exit_signal = level_exit_signal
        self.order.level_exit = level_exit
        self.order.time_exit = time_exit

        if not simulated:
            self.position = 0

        return self.order

class Basket():
    def __init__(self, assets):
        self.name = '-'.join([asset.name for asset in assets])
        self.assets = {asset.name : asset for asset in assets}
        self.prices = self.consolidate()
        self.quantities = 0
        self.position = 0
        self.spread_trade_recap = pd.Series()

        self.spread = None
        self.zscore = None
        self.johansen = None
        self.ev = None
        self.half_life = None
        self.drift_mean = None
        self.drift_vol = None

        self.prices.dropna(inplace=True)

    def consolidate(self):
        """
        @param path:
        @return:
        """

        df = pd.DataFrame()
        assets = self.assets.copy()

        for name, asset in assets.items():

            prices = asset.prices.rename({'close': name}, axis=1)
            if df.equals(pd.DataFrame()):
                df = prices[name].to_frame()
            else:
                df = df.merge(prices[name], left_index=True, right_index=True, how='outer')

        return df
    
    def compute_barsize(self):
        self.prices.index = pd.to_datetime(self.prices.index)
        time_diffs = self.prices.index.to_series().diff().dt.total_seconds()
        barsize = time_diffs.dropna().min()
        return barsize

    def compute_spread(self):
        self.spread = pd.Series(np.dot(np.log(self.prices.values.astype(float)), self.ev.values), self.prices.index)
        return self.spread
    
    def compute_zscore(self, period):
        self.zscore = (self.spread - self.spread.rolling(period).mean())/self.spread.rolling(period).std()
        return self.zscore

    def compute_spread_drift(self):
        drift = (self.drift_mean + abs(self.zscore.iloc[-1]) * self.drift_vol)
        return np.exp(drift) - 1

    def compute_drift(self, window=None):
        if window is None:
            window = abs(self.half_life)
        self.drift_mean = self.spread.rolling(window).mean().std()
        self.drift_vol = self.spread.rolling(window).std().std()

        return self.drift_mean, self.drift_vol    

    def johansen_test(self):
        self.johansen = coint_johansen(np.log(self.prices), det_order=0, k_ar_diff=1)
        ev = self.johansen.evec[:, 0]
        self.ev = pd.Series(ev/ev[0], list(self.assets.keys()))
        return self.ev
    
    def compute_half_life(self):
        lag = np.roll(self.spread, 1)
        lag[0] = 0
        ret = self.spread - lag
        ret.iloc[0] = 0

        lag2 = sm.add_constant(lag)
        model = sm.OLS(ret, lag2)
        res = model.fit()

        phi = 1 + res.params.iloc[1] 

        if 0 < abs(phi) < 1:
            half_life = int(round(-np.log(2) / np.log(abs(phi)), 0))
        else:
            half_life = False  # Not mean-reverting

        self.half_life = half_life
        return self.half_life

    def compute_profit_function(self, start=0.0, end=2, step=0.1):
        thresholds = np.arange(start, end, step) 
        zscores = self.zscore.dropna()
        frequencies = np.array([np.sum(zscores > i) / len(zscores.index) for i in thresholds])
        profits = frequencies * thresholds

        return pd.Series(profits, thresholds)
    
    def compute_optimal_zscore(self):
        profits = self.compute_profit_function()
        return profits.idxmax()

    def compute_quantities(self, total_amount):
        prices = [asset.prices.iloc[-1].close for asset in self.assets.values()]
        quantities = np.trunc(np.dot(total_amount / np.dot(np.abs(self.ev.values), prices), self.ev.values))
        self.quantities = pd.Series(quantities, list(self.assets.keys()))
        return self.quantities
    
    def initialise(self):
        if self.prices.empty:
            return False
        else:
            self.johansen_test()
            self.compute_spread()
            self.compute_half_life()
            self.compute_drift()
            if not self.half_life:
                return self
            else:        
                self.compute_zscore(self.half_life)

                if self.zscore.dropna().empty:
                    return False
                else: 
                    self.compute_optimal_zscore()

        return self
    
    def long_order(self, level_entries, time_entry, intensity=0, simulated=False):

        orders = []
        for asset in self.assets.values():
            quantity = self.quantities[asset.name]
            level_entry = level_entries[asset.name]
            if quantity > 0: 
                order = asset.long_order(level_entry, time_entry, quantity, intensity, simulated)
            elif quantity < 0:
                order = asset.short_order(level_entry, time_entry, quantity, intensity, simulated)
            orders.append(order)

        if not simulated:
            self.position = 1

        return orders
    
    def short_order(self, level_entries, time_entry, intensity=0, simulated=False):

        orders = []
        for asset in self.assets.values():
            quantity = -self.quantities[asset.name]
            level_entry = level_entries[asset.name]

            if quantity > 0: 
                order = asset.long_order(level_entry, time_entry, quantity, intensity, simulated)
            elif quantity < 0:
                order = asset.short_order(level_entry, time_entry, quantity, intensity, simulated)
            else:
                continue
            orders.append(order)
            
        if not simulated:
            self.position = -1

        return orders
    
    def close_positions(self, level_exits, time_exit, intensity=0, simulated=False):
        
        orders=[]
        for asset in self.assets.values():
            order = asset.close_position(level_exits[asset.name], time_exit, intensity, simulated)
            orders.append(order)

        if not simulated:    
            self.position = 0
        
        return orders

    def compute_pnl(self, level_exits, time_exit, intensity=0):

        orders = self.close_positions(level_exits, time_exit, intensity, simulated=True)
        pnl = np.sum(order.compute_pnl() for order in orders)

        return pnl
    
    def compute_spread_trade_recap(self, tradeID, orders, order_type):

        if order_type == 'entry':
            self.spread_trade_recap['tradeID'] = tradeID
            self.spread_trade_recap['position'] = self.position
            self.spread_trade_recap['amount'] = np.sum(abs(order.level_entry_signal * order.quantity) for order in orders)
            self.spread_trade_recap['quantities'] = ';'.join((str(order.quantity) for order in orders))

            self.spread_trade_recap['level_entry_signal'] = np.sum(np.log(order.level_entry_signal) * self.ev[order.asset.name] for order in orders)
            self.spread_trade_recap['level_entry'] = np.sum(np.log(order.level_entry) * self.ev[order.asset.name] for order in orders)
            self.spread_trade_recap['time_entry'] = [order.time_entry for order in orders][0]
            self.spread_trade_recap['transaction_costs'] = np.sum(order.transaction_costs for order in orders)
            
        elif order_type == 'exit':
            self.spread_trade_recap['level_exit_signal'] = np.sum(np.log(order.level_exit_signal) * self.ev[order.asset.name] for order in orders)
            self.spread_trade_recap['level_exit'] = np.sum(np.log(order.level_exit) * self.ev[order.asset.name] for order in orders)
            self.spread_trade_recap['time_exit'] = [order.time_exit for order in orders][0]
            self.spread_trade_recap['pnl'] = np.sum(order.compute_pnl() for order in orders)
            self.spread_trade_recap['transaction_costs'] += np.sum(order.transaction_costs for order in orders)

        return self.spread_trade_recap

class BasketSelection(object):

    def __init__(self, assets=None, path=None):

        if assets is None: 
            self.assets = {}
        else:
            self.assets = assets

        if path is not None:
            self.read_folder(path)

        self.prices = self.consolidate()

        self.baskets = None
        self.statistics = None

    def read_folder(self, path):

        folders = os.listdir(path)
        for file in tqdm(folders):
            if file != '.DS_Store':
                df = pd.read_csv(path + '/' + file)
                stock = file.replace('.csv', '')
                self.assets[stock] = Asset(stock, df)

        return self.assets
    
    def filter(self, proportion, inplace=True):
        """
        @param proportion:
        @return:
        """

        missing = self.prices.isnull().sum() / len(self.prices)
        mask = missing.gt(1 - proportion/100)
        self.prices = self.prices.loc[:, ~mask]
        if inplace:
            for k,v in mask.to_dict().items():
                if v: del self.assets[k]
        return self.prices

    def create_baskets(self, number_assets):
            """
            @return:
            """
            basket_generator = tqdm(itertools.combinations(list(self.assets.keys()), number_assets))
            self.baskets = {'-'.join(i): Basket([self.assets[_] for _ in i]) for i in basket_generator}

            return self.baskets
    
    def consolidate(self):
        """
        @param path:
        @return:
        """

        df = pd.DataFrame()
        assets = self.assets.copy()

        for name, asset in assets.items():

            prices = asset.prices.rename({'close': name}, axis=1)
            if df.equals(pd.DataFrame()):
                df = prices[name].to_frame()
            else:
                df = df.merge(prices[name], left_index=True, right_index=True, how='outer')

        return df
    
    def compute_pca(self, components=99, inplace=True):
        
        log_prices = np.log(self.prices.dropna().astype(float))
        log_returns = np.diff(log_prices,axis=0)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(log_returns)

        pca = PCA(n_components=components/100, svd_solver='full')
        pca.fit_transform(scaled_data)

        n_pcs= pca.components_.shape[0]
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        most_important_names = [self.prices.columns[most_important[i]] for i in range(n_pcs)]
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs) if most_important_names[i] not in most_important_names[:i]}

        assets = list(asset for asset in self.assets.keys())
        if inplace:
            for name in assets:
                if name not in most_important_names:
                    del self.assets[name]
                    self.prices.drop(name, axis=1,inplace=True)
        
        return pca, dic

    def compute_statistics(self):
            """
            @return:
            """

            stats = {}
            to_delete = []
            gen = (item for item in self.baskets.items())
            for name, basket in tqdm(gen):
            
                if not basket.initialise():
                    to_delete.append(name)
                else: 
                    stats[name] = {
                        'Half Life' : basket.half_life,
                        'Trace Statistic': basket.johansen.lr1[0],
                        'Max Eigen Statistic' : basket.johansen.lr2[0],
                        'Eigenvalue' : basket.johansen.eig[0],
                        'Eigenvector' : basket.johansen.evec[:, 0]
                        }
                    
            self.stats = pd.DataFrame(stats).T
            for name in to_delete:
                del self.baskets[name]

            return self.stats.sort_values('Half Life')
    

class BasketTrading():
    def __init__(self, basket, params):
        """
        @param data:
        """

        basket.barsize = basket.compute_barsize()
        self.basket = basket
        self.params = params
        self.basket_train, self.basket_test = self.split_train_test(self.params.get('proportion', 0.5))

        self.basket_bt = copy.deepcopy(self.basket_train)
        self.basket_bt.prices.drop(self.basket_bt.prices.index, inplace=True)
        [asset.prices.drop(asset.prices.index, inplace=True) for asset in self.basket_bt.assets.values()]

        self.params['threshold_exit'] = - params['threshold_entry'] * self.params['exit']
        self.tradeID = 1
        self.vl = self.params.get('amount',1)

        self.signals = self.basket_test.prices.copy()
        self.history = pd.DataFrame()

    def split_train_test(self, proportion=0.5):
        
        assets_train=[]
        assets_test=[]

        for asset in self.basket.assets.values():
            assets_train.append(Asset(asset.name, 
                                      asset.prices.drop(asset.prices.iloc[:int(np.ceil(len(asset.prices.index)*proportion))].index)))
            assets_test.append(Asset(asset.name, 
                                     asset.prices.drop(asset.prices.iloc[int(np.floor(len(asset.prices.index)*proportion)):].index)))
        basket_train = Basket(assets_train).initialise()
        basket_test = Basket(assets_test)

        return basket_train, basket_test

    def add_bar(self, bar):
        for asset in self.basket_bt.assets.values():
            asset.prices.loc[bar.Index] = bar.loc[asset.name]

            if len(asset.prices.index) > self.params['window']:
                asset.prices.drop(asset.prices.index[0], inplace=True)
            
        if len(self.basket_bt.prices.index) > 0:
            self.basket_bt.prices = self.basket_bt.prices.reset_index()

        self.basket_bt.prices = pd.concat([self.basket_bt.prices, bar.to_frame().T], ignore_index=True)
        self.basket_bt.prices.set_index('Index', inplace=True)

        if len(self.basket_bt.prices.index) > self.params['window']:
                self.basket_bt.prices.drop(self.basket_bt.prices.index[0], inplace=True)        

    def compute_indicators(self, bar):

        if not self.params.get('constant', False):
            self.basket_bt.johansen_test

        self.basket_bt.compute_spread()
        self.basket_bt.compute_zscore(self.params['window'])

        self.signals.loc[bar.Index, 'Ev'] = ';'.join(str(v) for v in self.basket_bt.ev.values)
        self.signals.loc[bar.Index, 'Spread'] = self.basket_bt.spread.iloc[-1]
        self.signals.loc[bar.Index, 'Zscore'] = self.basket_bt.zscore.iloc[-1]


    def compute_costs(self, orders):

        tc = np.sum(order.transaction_costs for order in orders)
        #bf = np.sum((self.basket_bt.half_life * self.basket_bt.barsize) * (order.asset.borrowing_fees/(360 * 86400)) * order.quantity * order.level_entry for order in orders)
        slippage = np.sum((order.level_entry - order.level_entry_signal) * order.quantity for order in orders)
        drift = self.basket_bt.compute_spread_drift() * self.vl

        costs = 2 * (tc + slippage) + drift

        return costs

    def estimated_pnl(self, bar):
        
        zscore = self.basket_bt.zscore.iloc[-1] 
        if zscore < 0:
            orders = self.basket_bt.long_order(bar, bar.Index, intensity=self.params.get('intensity', 0),simulated=True)
        elif zscore > 0:
            orders = self.basket_bt.short_order(bar, bar.Index, intensity=self.params.get('intensity', 0), simulated=True)

        gain = (np.exp(abs(self.basket_bt.spread.std() * (zscore - np.sign(zscore) * self.params['threshold_exit']))) - 1) * self.vl
        costs = self.compute_costs(orders)

        self.signals.loc[bar.Index, 'Costs'] = costs
        self.signals.loc[bar.Index, 'Gain'] = gain
        self.signals.loc[bar.Index, 'Estimated Pnl'] = gain - costs
        self.signals.loc[bar.Index, 'Estimated Pnl Ratio'] = gain / costs

        return gain/costs
    
    def compute_pnl(self, bar):

        old_pnl = np.sum(asset.order.compute_pnl() for asset in self.basket_bt.assets.values() if asset.order is not None)
        pnl = self.basket_bt.compute_pnl(bar, bar.Index, intensity=self.params.get('intensity', 0))
        self.signals.loc[bar.Index, 'Pnl Trade'] = pnl

        self.vl += pnl - old_pnl
        self.signals.loc[bar.Index, 'Liquid Value'] = self.vl

        return pnl
    
    def long_order(self, bar):
        level_entries = pd.Series((asset.compute_next_open(bar.Index) for asset in self.basket_test.assets.values()), bar.axes[0][1:])
        orders = self.basket_bt.long_order(level_entries, bar.Index, intensity=self.params.get('slippage', 0))

        for order in orders:
            self.vl -= order.transaction_costs
        
        spread_trade_recap = self.basket_bt.compute_spread_trade_recap(self.tradeID, orders, 'entry')

        self.signals.loc[bar.Index, 'TradeID'] = spread_trade_recap.tradeID
        self.signals.loc[bar.Index, 'Position'] = spread_trade_recap.position
        self.signals.loc[bar.Index, 'Amount'] = spread_trade_recap.amount
        self.signals.loc[bar.Index, 'Quantities'] = spread_trade_recap.quantities

        self.signals.loc[bar.Index, 'Level Entry Signal'] = spread_trade_recap.level_entry_signal
        self.signals.loc[bar.Index, 'Level Entry'] = spread_trade_recap.level_entry
        self.signals.loc[bar.Index, 'Transaction Costs'] = spread_trade_recap.transaction_costs
        self.signals.loc[bar.Index, 'Liquid Value'] = self.vl

        return

    def short_order(self, bar):
        level_entries = pd.Series((asset.compute_next_open(bar.Index) for asset in self.basket_test.assets.values()), bar.axes[0][1:])
        orders = self.basket_bt.short_order(level_entries, bar.Index, intensity=self.params.get('slippage', 0))

        for order in orders:
            self.vl -= order.transaction_costs
        
        spread_trade_recap = self.basket_bt.compute_spread_trade_recap(self.tradeID, orders, 'entry')
        
        self.signals.loc[bar.Index, 'TradeID'] = spread_trade_recap.tradeID
        self.signals.loc[bar.Index, 'Position'] = spread_trade_recap.position
        self.signals.loc[bar.Index, 'Amount'] = spread_trade_recap.amount
        self.signals.loc[bar.Index, 'Quantities'] = spread_trade_recap.quantities

        self.signals.loc[bar.Index, 'Level Entry Signal'] = spread_trade_recap.level_entry_signal
        self.signals.loc[bar.Index, 'Level Entry'] = spread_trade_recap.level_entry
        self.signals.loc[bar.Index, 'Transaction Costs'] = spread_trade_recap.transaction_costs
        self.signals.loc[bar.Index, 'Liquid Value'] = self.vl

        return

    def close_positions(self, bar):
        unadjusted_pnl = np.sum(asset.order.compute_pnl() for asset in self.basket_bt.assets.values())
        level_exits = pd.Series((asset.compute_next_open(bar.Index) for asset in self.basket_test.assets.values()), bar.axes[0][1:])
        orders = self.basket_bt.close_positions(level_exits, bar.Index, intensity=self.params.get('slippage', 0))
        transaction_costs = np.sum(order.transaction_costs for order in orders)
        
        spread_trade_recap = self.basket_bt.compute_spread_trade_recap(self.tradeID, orders, 'exit')
        self.vl += (- transaction_costs - unadjusted_pnl + spread_trade_recap.pnl)

        self.signals.loc[bar.Index, 'Level Exit Signal'] = spread_trade_recap.level_exit_signal
        self.signals.loc[bar.Index, 'Level Exit'] = spread_trade_recap.level_exit
        self.signals.loc[bar.Index, 'Pnl Trade'] = spread_trade_recap.pnl
        self.signals.loc[bar.Index, 'Transaction Costs'] = transaction_costs
        self.signals.loc[bar.Index, 'Liquid Value'] = self.vl

        self.history = pd.concat([self.history, spread_trade_recap.to_frame().T.copy()], ignore_index=True)
        self.tradeID += 1

        return

    def run(self):
    
        itertuples = tqdm(self.signals.itertuples())

        for i in itertuples:
            bar = pd.Series(i, list(i._fields))

            self.add_bar(bar)
            self.compute_indicators(bar)

            if self.basket_bt.position == 1 or self.basket_bt.position == -1:
                self.compute_pnl(bar)
                if self.vl <= 0 :
                    print('Loss of all capital, end of Backtest')
                    exit(1)
                if self.basket_bt.position == 1:
                    if (self.basket_bt.zscore.iloc[-1] > - self.params['threshold_exit']):
                        self.close_positions(bar)
                elif self.basket_bt.position == -1:
                    if self.basket_bt.zscore.iloc[-1] < self.params['threshold_exit']:
                        self.close_positions(bar)

            if self.basket_bt.position == 0:
                self.basket_bt.compute_quantities(self.vl)
                if ((self.basket_bt.zscore.iloc[-1] > self.params['threshold_entry'])
                        and (self.estimated_pnl(bar) >= self.params.get('security', 0))) :
                    self.short_order(bar)

                elif ((self.basket_bt.zscore.iloc[-1] < -self.params['threshold_entry'])
                      and (self.estimated_pnl(bar) >= self.params.get('security', 0))):
                    self.long_order(bar)
        return 

class Report(object):
    """

    """

    def __init__(self, backtest):
        self.backtest = backtest
        self.report = None

    def compute_report(self):
        """
        @param signals:
        @return:
        """

        report = {}
        date_format = '%Y-%m-%d %H:%M:%S'

        report['Equity Initial [$]'] = self.backtest.params.get('amount', 1)
        report['Equity Final [$]'] = self.backtest.vl
        report['Return [%]'] = (report['Equity Final [$]']/self.backtest.params.get('amount', 1) - 1) * 100
        
        signal =  self.backtest.signals['Liquid Value'].ffill()
        rets = pd.DataFrame(signal).pct_change()

        report['Volatility'] = np.nanstd(rets) * np.sqrt((self.backtest.basket.barsize/86400)*252) * 100
        report['Sharpe'] = (report['Return [%]'] / 100) / report['Volatility']
        report['Positions'] = self.backtest.tradeID
        report['Skew'] = self.backtest.history.pnl.skew()
        report['Kurtosis'] = self.backtest.history.pnl.kurtosis()
        report['Total Profit'] = self.backtest.history.pnl[self.backtest.history.pnl > 0].sum()
        report['Total Loss'] = self.backtest.history.pnl[self.backtest.history.pnl < 0].sum()
        report['Profit Factor'] = report['Total Profit'] / report['Total Loss']
        report['Average Pnl'] = self.backtest.history.pnl.mean()
        report['Average Profit'] = self.backtest.history.pnl[self.backtest.history.pnl > 0].mean()
        report['Average Loss'] = self.backtest.history.pnl[self.backtest.history.pnl < 0].mean()
        report['Average Time Position'] = (pd.to_datetime(self.backtest.history.time_exit,format=date_format)-pd.to_datetime(self.backtest.history.time_entry, format=date_format)).mean()
        report['Average Slippage'] = (self.backtest.history.level_entry-self.backtest.history.level_entry_signal+self.backtest.history.level_exit-self.backtest.history.level_exit_signal).mean()

        roll_max = signal.cummax()
        drawdown = signal/roll_max - 1.0

        report['Max Drawdown'] = drawdown.min()

        self.report = pd.Series(report)

        return self.report
    
    def plot_report(self):

        self.compute_report()
        print(self.report)

        signals = self.backtest.signals
        history = self.backtest.history

        long_positions = history[history.position > 0]
        short_positions = history[history.position < 0]

        short_position_color = '#FFA500'  # A blend of yellow and orange

        # Plotting
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle("Backtest Analysis Dashboard", fontsize=16, y=0.95)

        # Subplots
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
        ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
        ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1)

        # Top Left: Spread with markers for positions and shading
        ax1.plot(signals.index, signals.Spread, label="Spread", color='skyblue')

        # Add vertical lines and shading for positions
        for i in range(len(history)):
            
            ax1.axvline(x=history.iloc[i]['time_entry'], color='blue' if history.iloc[i]['position'] > 0 else short_position_color, linestyle='--', alpha=0.7)
            ax1.axvline(x=history.iloc[i]['time_exit'], color='blue' if history.iloc[i]['position'] > 0 else short_position_color, linestyle='--', alpha=0.7)
                
                # Shade the area between opening and closing
            ax1.axvspan(history.iloc[i]['time_entry'], history.iloc[i]['time_exit'], color='blue' if history.iloc[i]['position'] > 0 else short_position_color, alpha=0.2)

        # Add position markers
        ax1.scatter(long_positions.time_entry, long_positions.level_entry, color='blue', marker='^', label='Long Open')
        ax1.scatter(short_positions.time_entry, short_positions.level_entry, color=short_position_color, marker='^', label='Short Open')
        ax1.scatter(history.time_exit, history.level_exit, color='purple', marker='^', label='Close')
        ax1.set_title("Spread with Positions and Shading")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Spread")
        ax1.legend()

        # Top Right: Liquid Value with Maximum Drawdown
        ax2.plot(signals.index, signals['Liquid Value'], label="Liquid Value", color='blue', linewidth=2)
        ax2.fill_between(signals.index, signals['Liquid Value'], signals['Liquid Value'].cummax(), color='red', alpha=0.3, label="Drawdown")
        ax2.set_title("Liquid Value with Drawdown")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Value")
        ax2.legend()

        # Mid Left: PnL Trade over time without vertical lines or shading
        ax3.fill_between(signals.index, signals['Pnl Trade'].fillna(0), where=(signals['Pnl Trade'] > 0), interpolate=True, color='green', alpha=0.5)
        ax3.fill_between(signals.index, signals['Pnl Trade'].fillna(0), where=(signals['Pnl Trade'] <= 0), interpolate=True, color='red', alpha=0.5)
        ax3.set_title("PnL Trade Over Time")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("PnL")

        # Mid Right: Histogram of Trade PnL
        ax4.hist(history.pnl, bins=20, color='skyblue', edgecolor='black')
        ax4.set_title("Histogram of Trade PnL")
        ax4.set_xlabel("PnL")
        ax4.set_ylabel("Frequency")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()