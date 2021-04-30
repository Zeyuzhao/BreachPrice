#%%
# Standard libraries
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
import yfinance as yf

# os.system("pip install seaborn")
import seaborn as sns
pd.set_option('display.max_columns', None)

#%%
stock_price_aa_records = pd.read_csv("../data/stock_price_aa_records_new.csv")

# %%
stock_price_aa_records.head()