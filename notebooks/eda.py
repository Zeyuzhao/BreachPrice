#%%
# Standard libraries
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

# %%
db = wrds.Connection()

# %%
# PRC Dataset
PRC_df = pd.read_csv("../data/prc.csv")

# %%
PRC_df.head()
# %%
PRC_df.drop(PRC_df.columns[[13, 14, 15]], axis=1, inplace=True)
#%%
PRC_df["Total Records"].value_counts()
#%%
PRC_df['Total Records'].fillna(0, inplace=True)
#%%


def records_to_int(record_str):
    return int(str(record_str).replace(",", ""))

PRC_df['Total Records'] = PRC_df['Total Records'].apply(records_to_int)
# %%
# About 50-50 zero/non-zero records (2187 : 2863)
PRC_df_nonzero_records = PRC_df[PRC_df['Total Records'] != 0]
PRC_df_nonzero_records['Total Records'].value_counts()

# %%
# Audit Analytics Dataset
xls = pd.ExcelFile('../data/audit_analytics.xlsx')
aa_records_df = pd.read_excel(xls, 'PublicCyber')
aa_ddos_df = pd.read_excel(xls, 'DDoS')

#%%
aa_records_df.head()
# %%
# Drop rows without tickers
aa_records_df.dropna(subset=['Ticker']).reset_index(drop=True)
aa_records_df["Ticker"].value_counts() # Multiple companies get muliple breaches
# %%
table_columns = ['Company name', 'Ticker', 'Date of Breach', 'Date Became Aware of Breach', 'Date of Disclosure',
                 'Number of Records', 'Type of Info', 'Information', 'Attack', 'Region', 'SIC Code']
aa_records_df = aa_records_df[aa_records_df.columns.intersection(
    table_columns)]
aa_records_df.head()

# %%
today = datetime.datetime.today().date()


def nearest(items, pivot):
    """
    Gets closest day in a set (used to obtain stock price y months after disclosure)
    """
    return min(items, key=lambda x: abs((x - pivot).days))


def stock_after_disclosure(row, num_months):
    """
    Returns an array containing the monthly stock price of a firm after date of disclosure (0 - num_months months after breach).
    If firm exists in YahooFinance database, but no stock price available for a month (either b/c that date has yet to occur or b/c simply N/A),
    returns np.nan.
    If firm does not exist in YahooFinance database, return array of np.nan's.
    
    Parameters: 
    row : Dataframe row
        Input dataframe's row (used along with df.apply)
    num_months : int
        Month limit
    """
    start = pd.to_datetime(row['Date of Disclosure'])
    end = start + pd.DateOffset(months=num_months)
    # Don't know if i should include this, check stock day before breach to control for large stock dip when breach is disclosed
    start -= datetime.timedelta(days=1)
    #print(row['Ticker'])
    try:
        # Yahoo Ticker subclass notation
        ticker = str(row['Ticker']).replace(".", "-")
        df = web.DataReader(ticker, 'yahoo', start, end)
        #display(df)
        lst = []
        for month in range(0, num_months + 1):
            date = nearest(df.index, (start + pd.DateOffset(months=month)))
            if today <= date.date():
                for x in range(month, num_months + 1):
                    lst.append(np.nan)
                break
            lst.append(df.loc[date]["Close"])
        return lst
    except Exception as e:
        print("Error at %s" % row['Ticker'])
        print(repr(e))
        return [np.nan] * (num_months + 1)

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
lst = []
months_after = 12  # Toggle this value
col = []
for i in range(0, months_after + 1):
    col.append("Stock Price (%s months DoD)" % i)
#%%
print(col)
print(lst)
# %%
# Create array of arrays that contains stock prices after date of disclosure for each breach
# RUN ONCE ONLY
for index, row in tqdm(aa_records_df.iterrows(), total=len(aa_records_df)):
    #     print("%s: %s" %(index, row['Ticker']))
    x = stock_after_disclosure(row, months_after)
    lst.append(x)

# %%
stock_prices = pd.DataFrame(lst, columns=col)
stock_price_aa_records = pd.concat(
    [aa_records_df, stock_prices], axis=1, join='inner')

#%%
stock_prices = pd.DataFrame(lst, columns=col)
stock_prices.tail()
#%%

stock_price_aa_records.tail()
# %%


def analyst_stock_price(row):
    """
    Returns the median and mean of analyst stock price forecasts for a firm, where the forecasts are within a month after the beach. 
    These forecasts predict the stock price 12 months into the future.
    
    Parameters
        row - Dataframe row
        Input dataframe's row (used along with df.apply)
    Returns
        List of length 2. [median, mean]
    """
    date = pd.to_datetime(row['Date of Disclosure'])

    # Yearly estimates seem to be the norm
    sql_query = """
    SELECT VALUE as stock_price
    FROM ibes.ptgdet
    WHERE OFTIC ='{}' AND CAST(HORIZON as int) = 12 AND ANNDATS BETWEEN '{}' and '{}'
    """.format(row['Ticker'], date, date + pd.DateOffset(months=1))

    df = db.raw_sql(sql_query)

    if len(df.index) == 0:
        return [np.nan, np.nan, 0]
    return [df['stock_price'].median(), df['stock_price'].mean(), len(df)]


# %%
# Create array of arrays that contains stock prices after date of disclosure for each breach
analyst_predictions = []
for index, row in tqdm(stock_price_aa_records.iterrows(), total=len(stock_price_aa_records)):
    analyst_predictions.append(analyst_stock_price(row))

# Merge stock price after breach with original dataframe
median_mean_count_df = pd.DataFrame(analyst_predictions, columns= ['median stock forecast', 'mean stock forecast', 'count stock forecast'])

# %%
stock_price_aa_records = pd.concat([stock_price_aa_records, median_mean_count_df], axis=1, join='inner')
# %%
stock_price_aa_records["count stock forecast"].value_counts()
# A good chunck of companies don't have stock estimates
# %%
stock_price_aa_records.to_csv(
    "../data/stock_price_aa_records_new.csv",  index=False)

#%%
stock_price_aa_records.info()

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

# %%
