from datasets import load_dataset
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict 
import json
from requests import exceptions

en = load_dataset("allenai/c4", "en", streaming=True) # You can choose the subset of the data as well (e.g. blocklist)

counts_dict = defaultdict(lambda: {'t':0,'c':0}) 

words = re.compile(r'\w+')
def word_tokenize(text):
    """An extremely simple but fast tokenizer to help us rank webpage contributions."""
    return len(words.findall(text)) # 15.3 ms on M2

# This is the naive unthreaded approach to looping through the streamed data. It has no restart mechanism aside from rerunning the instance (perhaps with an updated dataset.skip(i)). 
# Fortunately, the dataset is small enough that we can just rerun the whole thing if we need to, only takes ~24hr
for i, row in enumerate(tqdm(en['train'], total=393391519)):
    try:
        tokenscount = word_tokenize(row['text'])
        netloc = urlparse(row['url']).netloc
        counts_dict[netloc]['t'] += tokenscount
        counts_dict[netloc]['c'] += 1
    except exceptions.RequestException as e:
        print(f"You can dataset.skip({i}) to restart this; we have no graceful handling.")
        raise exceptions.RequestException(e)

    if i % 1000000 == 0:
        df = pd.DataFrame.from_dict(counts_dict, orient='index')
        df.to_csv('c4_stream_blocklist.csv')
        print('Saved up to index:', i)