import time
from datetime import datetime

import pandas as pd

import App as a
from ibapi.contract import Contract

import logging

# Enable detailed logging
#logging.basicConfig(level=logging.DEBUG)

file = pd.ExcelFile("/Users/LN/Desktop/IBAPI/datasets/Ibapi/Underlyings.xlsx")
index = file.parse('ETF1', index_col=0)
index=index.loc[(index.CURRENCY == 'USD') & (index.Liquid == 1)]
#index =index.drop(index.loc[:'TECS'].index)
print(f"Retrieving data of {len(index.index)} assets")
app = a.IBapi("127.0.0.1", 4002, 1)
app.threadStart()

for i in index.itertuples():

    try:
        start = time.time()
        print("Starting Time : ", datetime.fromtimestamp(start), "| Symbole : ", i.Index)

        contract = Contract()
        contract.symbol = i.IBKR
        contract.secType = "STK"
        contract.exchange = "ARCA"
        contract.currency = i.CURRENCY

        asset = a.FinAssets(app, contract)
        details = asset.getDetails()
        asset.contract.symbol = details.contract.symbol

        df = asset.getHistData(durationStr='6 M', barSizeSetting='5 mins', useRTH=0)
        print(df.shape)
        df.to_csv(f"/Users/LN/Desktop/IBAPI/datasets/Ibapi/etf/5mins/liquid/{asset.contract.symbol.lower()}.csv")

        end = time.time() - start

        print(f'End in : {end // 60} Min and {round(end % 60,2)} Sec')

    except AttributeError:
        continue

app.threadEnd()


# os.system('Say "This is the end of the process"')
print("ok")

