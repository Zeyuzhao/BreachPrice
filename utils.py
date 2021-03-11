#%%
import difflib
import json
import requests
from pathlib import Path

# with open(Path('data') / 'ticker.json', "r") as f:
#     ticker_db = json.load(f)

def data_download(url: str, filename: str):
    """Downloads file from url and saves it to data/{filename}"""
    r = requests.get(url)
    with open(Path("..") / 'data' / filename, 'wb') as f:
        f.write(r.content)

def get_ticker(company_name: str) -> str:
    pass

#%%
ticker_db
# %%
