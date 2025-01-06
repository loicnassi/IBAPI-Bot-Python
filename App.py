"""
Comment on purpose of this file

"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper, Decimal, floatMaxString, decimalMaxString
from ibapi.contract import Contract, ContractDetails
from ibapi.order import Order

from datetime import datetime

import pandas as pd

import time
import threading
import random


class IBapi(EWrapper, EClient):
    """

    """

    def __init__(self, HOST, PORT, CLIENTID, VERBOSE=False):
        EClient.__init__(self, self)

        self.host = HOST
        self.port = PORT
        self.clientId = CLIENTID
        self.verbose = VERBOSE

        self.connection = False
        self.nextorderId = None

        self.req = {}
        self.error_received = {}

    def threadStart(self):
        """
        """
        self.connect(self.host, self.port, self.clientId)

        thread = threading.Thread(target=self.run, daemon=True, name='main')
        thread.start()

        while not self.connection: time.sleep(0.01)

    def threadEnd(self):
        """
        """
        self.disconnect()
        self.connection = False

    @iswrapper
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """ """
        print("Error: ", reqId, " ", errorCode, " ", errorString, ' ', advancedOrderRejectJson)

        if reqId in self.error_received:
            error_func = self.error_received[reqId]
            error_func(errorCode, reqId)

            del self.error_received[reqId]

        self.connection = True


    @iswrapper
    def nextValidId(self, orderId):
        self.nextorderId = orderId
        print('The next valid ID is : ', self.nextorderId)

    @iswrapper
    def contractDetails(self, reqId: int, contractDetails: ContractDetails):

        func = self.req[reqId]
        func(contractDetails)


    @iswrapper
    def contractDetailsEnd(self, reqId: int):
        self.req[reqId].retrieval = True

    @iswrapper
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        summary = pd.Series([account, tag, value, currency], index=['Account', 'Tag', 'Value', 'Currency'])

        func = self.req[reqId]
        func(summary)

    @iswrapper
    def accountSummaryEnd(self, reqId: int):
        self.req[reqId].retrieval = True
        self.cancelAccountSummary(reqId)

    @iswrapper
    def position(self, account: str, contract, position: Decimal, avgCost: float):
        position = pd.Series([account, contract.symbol, contract.secType, contract.currency,
                              decimalMaxString(position), floatMaxString(avgCost)],
                             index=['Account', 'Symbol', 'SecType', 'Currency', 'Position', 'Avg Cost'])
        func = self.req['positions']
        func(position)

    @iswrapper
    def positionEnd(self):
        self.req['positions'].retrieval = True

    @iswrapper
    def realtimeBar(self, reqId, date: int, open_: float, high: float, low: float, close: float,
                    volume: Decimal, wap: Decimal, count: int):

        date = datetime.fromtimestamp(int(date)).strftime('%Y-%m-%d %H:%M:%S')
        bar = pd.Series([date, open_, high, low, close], index=['date', 'open', 'high', 'low', 'close'])
        bar.date = pd.to_datetime(bar.date)

        """Have to set a condition to control difference between bar time and current time """
        func = self.req[reqId]
        func(bar)

    @iswrapper
    def historicalTicks(self, reqId: int, ticks, done: bool):
        func = self.req[reqId]
        func(ticks)
        self.req[reqId].retrieval = True

    @iswrapper
    def historicalTicksLast(self, reqId: int, ticks, done: bool):
        func = self.req[reqId]
        func(ticks)
        self.req[reqId].retrieval = True

    @iswrapper
    def historicalTicksBidAsk(self, reqId: int, ticks, done: bool):
        func = self.req[reqId]
        func(ticks)
        self.req[reqId].retrieval = True

    @iswrapper
    def historicalData(self, reqId, bar):
        """
        """
        if len(bar.date) == 8:
            date =  datetime.strptime(bar.date, "%Y%m%d")
        else:
            date = datetime.fromtimestamp(int(bar.date)).strftime('%Y-%m-%d %H:%M:%S')
        quote = pd.Series([date, bar.open, bar.high, bar.low, bar.close, int(bar.volume)],
                          index=['date', 'open', 'high', 'low', 'close', 'volume'])

        func = self.req[reqId]
        func(quote)

    @iswrapper
    def historicalDataEnd(self, reqId, start, end):
        """
        Define what happens when retrieval ends
        """
        self.req[reqId].retrieval = True

    @iswrapper
    def openOrder(self, orderId, contract, order, orderState):
        order = pd.Series([orderId, contract.symbol, contract.secType, contract.exchange,
                           order.action, order.orderType, order.totalQuantity, orderState.status],
                          index=['OrderId', 'Symbol', 'SecType', 'Exchange', 'Action', 'Type', 'Quantity', 'Status'])

        try:
            func = self.req['orders']
            func(order)
        except KeyError:
            print(f'New Order – orderId : {orderId} | Symbol : {order.Symbol} | Action : {order.Action} | '
                  f'Quantity : {order.Quantity} | Status : {order.Status}')

    @iswrapper
    def openOrderEnd(self):
        self.req['orders'].retrieval = True

    @iswrapper
    def execDetails(self, reqId, contract, execution):
        print('Order Executed: ', reqId, contract.symbol, contract.secType, contract.currency,
              execution.orderId, execution.shares)

    def newReq(self, func, reqId=None):
        """

        """
        if reqId is None: reqId = random.randint(0, 99999)

        func.__setattr__('retrieval', False)
        self.req[reqId] = func

        return reqId

    def reqError(self, func, reqId=None):
        """

        """
        if reqId is None: reqId = random.randint(0, 99999)
        self.error_received[reqId] = func

        return reqId

    def getAccountSummary(self, groupName='All', tags='$LEDGER'):
        """

        @param groupName:
        @param tags:
        @return:
        """
        df = pd.DataFrame(columns=['Account', 'Tag', 'Value', 'Currency'])

        def parseAccount(summary):
            nonlocal df
            df = df.append(summary, ignore_index=True)

        reqId = self.newReq(parseAccount)
        self.reqAccountSummary(reqId, groupName, tags)

        while not self.req[reqId].retrieval: time.sleep(0.01)
        del self.req[reqId]

        df.set_index('Tag', inplace=True)
        return df

    def getPositions(self):
        """

        @return:
        """
        df = pd.DataFrame(columns=['Account', 'Symbol', 'SecType', 'Currency', 'Position', 'Avg Cost'])

        def parsePositions(position):
            nonlocal df
            df = df.append(position, ignore_index=True)

        reqId = self.newReq(parsePositions, 'positions')
        self.reqPositions()

        while not self.req[reqId].retrieval: time.sleep(0.01)

        df.set_index('Symbol', inplace=True)
        return df

    def getOpenOrders(self):

        openOrders = pd.DataFrame(columns=['OrderId', 'Symbol', 'SecType', 'Exchange', 'Action',
                                           'Type', 'Quantity', 'Status'])

        def parseOpenOrders(order):
            nonlocal openOrders
            openOrders = openOrders.append(order, ignore_index=True)

        reqId = self.newReq(parseOpenOrders, 'orders')
        self.reqOpenOrders()

        while not self.req[reqId].retrieval: time.sleep(0.01)

        openOrders.set_index('Symbol', inplace=True)
        return openOrders

    @staticmethod
    def BracketOrder(parentOrderId: int, action: str, quantity: Decimal,
                     limitPrice, takeProfitLimitPrice,
                     stopLossPrice):

        bracketOrder = []
        # This will be our main or "parent" order
        parent = Order()
        if not limitPrice:
            parent.orderType = "MKT"
        else:
            parent.orderType = "LMT"
            parent.lmtPrice = limitPrice

        parent.orderId = parentOrderId
        parent.action = action
        parent.totalQuantity = quantity
        # The parent and children orders will need this attribute set to False to prevent accidental executions.
        # The LAST CHILD will have it set to True,
        parent.transmit = False
        bracketOrder.append(parent)

        if takeProfitLimitPrice:
            takeProfit = Order()
            takeProfit.orderId = parent.orderId + 1
            takeProfit.action = "SELL" if action == "BUY" else "BUY"
            takeProfit.orderType = "LMT"
            takeProfit.totalQuantity = quantity
            takeProfit.lmtPrice = takeProfitLimitPrice
            takeProfit.parentId = parentOrderId
            takeProfit.transmit = False

            bracketOrder.append(takeProfit)

        stopLoss = Order()
        stopLoss.orderId = parent.orderId + 2
        stopLoss.action = "SELL" if action == "BUY" else "BUY"
        stopLoss.orderType = "STP"
        # Stop trigger price
        stopLoss.auxPrice = stopLossPrice
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parentOrderId
        # In this case, the low side order will be the last child being sent. Therefore, it needs to set this attribute
        # to True to activate all its predecessors
        stopLoss.transmit = True
        bracketOrder.append(stopLoss)

        return bracketOrder


class FinAssets(object):
    """

    """

    def __init__(self, APP: IBapi, CONTRACT: Contract, TRADING: Contract = False, SHOW="MIDPOINT"):

        self.app = APP
        self.contract = CONTRACT

        self.trading = TRADING if TRADING else self.contract
        self.show = SHOW
        # self.minTick = self.getMinTick()

    def getMinTick(self):
        minTick = 0

        def parseMinTick(contract):
            nonlocal minTick
            minTick = contract.minTick

        reqId = self.app.newReq(parseMinTick)
        self.app.reqContractDetails(reqId, self.contract)

        while not self.app.req[reqId].retrieval: time.sleep(0.01)
        del self.app.req[reqId]

        rnd = 0
        while minTick < 1:
            rnd += 1
            minTick *= 10

        return rnd

    def getHistData(self, endDateTime='', durationStr='2000 S', barSizeSetting='5 secs',
                    useRTH=1, formatDate=2, keepUpToDate=False, chartOptions=None):

        df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

        def errorFunction(errorID, reqId):
            if errorID == 162:
                self.app.req[reqId].retrieval = True
                print('No available data')
            elif errorID == 200:
                self.app.req[reqId].retrieval = True
                print('No available contract')

        def parseHistData(bar):
            """

            @param bar:
            @return:
            """
            nonlocal df
            df = pd.concat([df, bar.to_frame().T], ignore_index=True)

        reqId = self.app.newReq(parseHistData)
        self.app.reqError(errorFunction, reqId)
        self.app.reqHistoricalData(reqId, self.contract, endDateTime, durationStr, barSizeSetting,
                                   self.show, useRTH, formatDate, keepUpToDate, chartOptions)

        while not self.app.req[reqId].retrieval: time.sleep(0.01)
        del self.app.req[reqId]

        df.set_index('date', inplace=True)
        return df


    def getHisTicks(self, startDateTime='', endDateTime='', numberOfTicks=1,
                    useRTH=1, ignoreSize=True, miscOptions=None):

        df = pd.DataFrame(columns=['Time', 'Price', 'Size'])

        def parseHistTicks(ticks):
            """

            @param ticks:
            @return:
            """

            nonlocal df
            for tick in ticks:
                df = df.append(pd.Series(data={'Time': tick.time,
                                               'Price': tick.price,
                                               'Size': tick.size}), ignore_index=True)

        reqId = self.app.newReq(parseHistTicks)
        self.app.reqHistoricalTicks(reqId, self.contract, startDateTime, endDateTime, numberOfTicks, self.show,
                                    useRTH, ignoreSize, miscOptions)

        while not self.app.req[reqId].retrieval: time.sleep(0.01)
        del self.app.req[reqId]

        return df

    def getOpenOrderId(self):
        """

        @return:
        """
        try:
            df = self.app.getOpenOrders()
            orders = df.loc[self.trading.symbol]
            orders = orders.OrderId
            orders = orders.values if type(orders) is not int else [orders]
        except KeyError:
            orders = []

        return orders

    def getPosition(self):
        """

        @return:
        """
        try:
            df = self.app.getPositions()
            position = df.loc[self.trading.symbol]
            position = int(position.Position)

        except (KeyError, ValueError, TypeError):
            position = 0
            print('No current position on this contract')
        return position

    def getRealTime(self, barSize=5, useRTH=False, realTimeBarOptions=None, bot=None):
        """

        @param bot:
        @param barSize:
        @param useRTH:
        @param realTimeBarOptions:
        @return:
        """

        def parseRealTime(bar):
            nonlocal bot
            nonlocal barSize

            if bar.date >= bot.dates[-1]:

                msg = f'Real-time bar – Date : {bar.date} | Symbol : {self.contract.symbol}, | Open : {bar.open} | '
                msg += f'High : {bar.high} | Low : {bar.low} | Close : {bar.close}'
                print(msg)

                if bot is not None:
                    args = [bar]
                    thread = threading.Thread(target=bot, args=args, name='Bot')
                    thread.start()
            else:
                pass

        reqId = self.app.newReq(parseRealTime)
        self.app.reqRealTimeBars(reqId, self.contract, barSize, self.show, useRTH, realTimeBarOptions)

    def getLastTick(self):

        tick = []

        def parseLastTick(ticks):
            nonlocal tick
            tick.append(ticks[-1])

        reqId = self.app.newReq(parseLastTick)

        t = time.strftime('%Y%m%d-%H:%M:%S', time.gmtime())
        self.app.reqHistoricalTicks(reqId, self.contract, '', t, 1, self.show, 1, True, [])

        while not self.app.req[reqId].retrieval: time.sleep(0.01)
        del self.app.req[reqId]

        tick = str(tick[0]).split(', ')
        tick = [_ for _ in tick if 'Price: ' in _][0]
        tick = float(tick.split(': ')[1])

        return tick

    def placeSimpleOrder(self, action, orderType, quantity):

        if quantity == 0:
            print('No current position on this contract')
        else:
            order = Order()
            order.action = action
            order.orderType = orderType
            order.totalQuantity = quantity
            order.orderId = self.app.nextorderId

            self.app.placeOrder(order.orderId, self.trading, order)
            self.app.reqIds(-1)

    def placeBracketOrder(self, action, quantity, limitPrice, takeProfit, stopLoss):
        price = self.getLastTick() if not limitPrice else limitPrice
        sens = 1 if action == 'BUY' else -1

        takeProfitPrice = round(price * (1 + sens * takeProfit), self.minTick) if takeProfit else False
        stopLossPrice = round(price * (1 - sens * stopLoss), self.minTick)

        bracketorder = self.app.BracketOrder(self.app.nextorderId, action, quantity,
                                             limitPrice, takeProfitPrice, stopLossPrice)

        for order in bracketorder:
            self.app.placeOrder(order.orderId, self.trading, order)
        self.app.reqIds(-1)

    def closePosition(self):

        position = self.getPosition()

        if position < 0:
            action = 'BUY'
            position *= -1
        else:
            action = 'SELL'

        self.placeSimpleOrder(action, 'MKT', position)
        self.cancelPendingOrders()

    def cancelPendingOrders(self):
        pendingOrders = self.getOpenOrderId()
        if len(pendingOrders) == 0:
            print('No pending orders on this contract')
        else:
            [self.app.cancelOrder(_, '') for _ in pendingOrders]


    def getDetails(self):

        contractDetails= None

        def errorFunction(errorID, reqId):
            if errorID == 200:
                self.app.req[reqId].retrieval = True
                print('No available contract')
        def parseDetails(details):
            nonlocal contractDetails
            contractDetails = details

        reqId = self.app.newReq(parseDetails)
        self.app.reqError(errorFunction, reqId)
        self.app.reqContractDetails(reqId, self.contract)

        while not self.app.req[reqId].retrieval: time.sleep(0.01)
        del self.app.req[reqId]

        return contractDetails

