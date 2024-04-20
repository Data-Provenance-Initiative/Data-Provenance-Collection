
"""For this we're doing to first download the full dataset, and then process it locally as streaming such a large dataset often ends in pain."""
from datasets import load_dataset
from collections import defaultdict 
from requests import exceptions
from urllib.parse import urlparse
from tqdm import tqdm
import pandas as pd
import time
import os
    
# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("tiiuae/falcon-refinedweb", split='train') # You might want to choose where you download this. 

words = re.compile(r'\w+')
def word_tokenize(text):
    """An extremely simple but fast tokenizer to help us rank webpage contributions."""
    return len(words.findall(text)) # 15.3 ms on M2

counts_dict_en = defaultdict(lambda: {'t':0,'c':0}) 
for i, row in enumerate(tqdm(dataset_skipped, total=len(dataset_skipped), smoothing=0)):
    try:
        tokenscount = word_tokenize(row['content'])
        netloc = urlparse(row['url']).netloc
        counts_dict_en[netloc]['t'] += tokenscount
        counts_dict_en[netloc]['c'] += 1
        if i % 1000000 == 0:
            df = pd.DataFrame.from_dict(counts_dict_en, orient='index')
            df.to_csv(f'refinedweb_download.csv')
            print('Saved up to index:', i)
    except Exception as e:
        # This might drop an occasional URL (but less that 0.001% of the data based on experiments)
        print(e)
        time.sleep(60)
        continue


# Restart logic. Sometimes your server will crash for nonsense reasons, it happens. To restart, load in the last save and skip to the last index.
refined_download = pd.read_csv('refinedweb_download.csv')
counts_dict_en_old = refined_download.set_index('Unnamed: 0').to_dict(orient='index')

# We're gonna make sure we convert this dict to a default dict to work with above. 
counts_dict_en_old = counts_dict_en
counts_dict_en = defaultdict(lambda: {'t':0,'c':0}, counts_dict_en_old)

dataset_skipped = dataset.select(range(675000001, len(dataset)))
# dataset_skipped = dataset[675000001:] # Don't do this or it will try to load the whole dataset into memory

# You can now rerun the above code (usually in an interactive session in tmux, but whatever works.)

