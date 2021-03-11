#%%
import requests
from pathlib import Path
# from utils import data_download
import json
import pandas as pd
#%%
# Download cik, ticker, company_name data
# data_download(url='https://www.sec.gov/files/company_tickers.json', filename='ticker.json')
# Download Privacy Rights Clearinghouse (PRC) breach data
# data_download(url='https://privacyrights.org/sites/default/files/2020-01/PRC%20Data%20Breach%20Chronology%20-%201.13.20.csv', filename='prc.csv')
# %%

# Pre-process ticker.json into csv file
with open(Path("..") / 'data' / 'ticker.json') as f:
    df = pd.read_json(f, orient='index')

df.head()
df.to_csv(Path("..") / 'data' / 'ticker.csv')


# %%
