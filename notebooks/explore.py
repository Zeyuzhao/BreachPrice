#%%
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from scipy import stats
# Visualization
import matplotlib.pyplot as plt
import datetime
import os

# os.system("pip install wrds") # TODO: Probably put this in utils.py
import wrds
# os.system("pip install pandas-datareader")
import pandas_datareader.data as web

# os.system("pip install seaborn")
import seaborn as sns
pd.set_option('display.max_columns', None)
#%%

import math
def sic_to_industry(sic_code):
    if math.isnan(sic_code):
        return "N/A"
    x = int(sic_code/100)
    if x <= 9:
        return "Agriculture, Forestry, Fishing"
    elif x <= 14:
        return "Mining"
    elif x <= 17:
        return "Construction"
    elif x <= 39:
        return "Manufacturing"
    elif x <= 49:
        return "Transportation & Public Utilities"
    elif x <= 51:
        return "Wholesale Trade"
    elif x <= 59:
        return "Retail Trade"
    elif x <= 67:
        return "Finance, Insurance, Real Estate"
    elif x <= 89:
        return "Services"
    elif x <= 99:
        return "Public Administration"
    return "N/A"


# %%
stock_price_aa_records = pd.read_csv("../data/stock_price_aa_records_new.csv")

#%%

stock_price_aa_records["SIC Code"] = stock_price_aa_records["SIC Code"].apply(sic_to_industry)
stock_price_aa_records.tail()
# %%

# Cleaning and Dummy encoding
stock_price_aa_records['Type of Info'] = stock_price_aa_records['Type of Info'].str.replace(" ", "")
stock_price_aa_records['Attack'] = stock_price_aa_records['Attack'].str.replace("; ", "|")
stock_price_aa_records = pd.concat([stock_price_aa_records.drop(
    'Type of Info', 1), stock_price_aa_records['Type of Info'].str.get_dummies(sep="|").add_suffix(" (Type of Info)")], 1)
stock_price_aa_records = pd.concat([stock_price_aa_records.drop(
    'Attack', 1), stock_price_aa_records['Attack'].str.get_dummies(sep="|").add_suffix(" (Attack)")], 1)
stock_price_aa_records = pd.concat([stock_price_aa_records.drop(
    'SIC Code', 1), stock_price_aa_records['SIC Code'].str.get_dummies(sep="|").add_suffix(" (Industry)")], 1)
stock_price_aa_records = pd.concat([stock_price_aa_records.drop(
    'Region', 1), stock_price_aa_records['Region'].str.get_dummies(sep="|").add_suffix(" (Region)")], 1)

# %%
stock_price_aa_records.head()
# %%
stock_price_aa_records.to_csv("../data/stock_indicators.csv")
#%%






lst = []
months_after = 12  # Toggle this value
col = []
for i in range(0, months_after + 1):
    col.append("Stock Price (%s months DoD)" % i)
col


# %%
stock_prices = pd.DataFrame()
n = 1 
for n, x in enumerate(col[1:]):
    stock_prices[col[n]] = stock_price_aa_records.apply(lambda row: (row[x] - row[col[0]])/row[col[0]], axis = 1)

boxplot = sns.boxplot(x="variable", y = "value", data=pd.melt(stock_prices).dropna())
boxplot.set(xlabel="Months after Disclosure", ylabel='Percent Stock Price Change') # Where x month is percent change from start of breach
boxplot.set_title("Percent Change of Actual Stock Price (Box Plot)")
plt.show()

ax = sns.violinplot(x='variable', y='value', data=pd.melt(stock_prices).dropna())

plt.xlabel('Months after Disclosure')
plt.ylabel('Percent Stock Price Change')
ax.set_title("Percent Change of Actual Stock Price (Violin Plot)")
ax.plot()
