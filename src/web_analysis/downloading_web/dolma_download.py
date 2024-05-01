
# Download instructions
"""
Follow the download documentation online or use the following script.

```
DATA_DIR="dolma_data_1_7"
PARALLEL_DOWNLOADS="30"
DOLMA_VERSION="v1_7"

git clone https://huggingface.co/datasets/allenai/dolma
mkdir -p "${DATA_DIR}"

cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"
```"""

# Processing code
import pandas as pd 
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from fasttokens import word_tokenize # Same tokenizer as other scripts
from urllib.parse import urlparse

def default_count():
    return {'t': 0, 'c': 0}

def process_file(file):
    local_counts = defaultdict(default_count)
    try:
        reader = pd.read_json(file, lines=True, chunksize=5000)
        for df in reader:
            for metadata, tokenscount, source in zip(df['metadata'], df['text'].apply(word_tokenize), df['source']):
                if 'source_domain' in metadata:
                    netloc = metadata['source_domain']
                elif 'subreddit' in metadata:
                    netloc = 'reddit.com'
                elif 'url' in metadata:
                    netloc = urlparse(metadata['url']).netloc
                else:
                    netloc = source
                local_counts[netloc]['t'] += tokenscount
                local_counts[netloc]['c'] += 1
            del df
        return local_counts
    except Exception as e:
        print("Error in process", e)
        return {'error': file, 'e':e}

files = glob.glob('dolma_data/*.json.gz')
counts_dict_en = defaultdict(default_count)

with ProcessPoolExecutor(max_workers=30) as executor:
    with tqdm(total=len(files), desc="Processing Files", smoothing=0) as progress:
        futures = {executor.submit(process_file, file): file for file in files}
        for future in as_completed(futures):
            local_counts = future.result()
            if 'error' in local_counts:
                print("Error returned for", local_counts['error'], local_counts['e'])
                continue
            else:
                for netloc in local_counts:
                    counts_dict_en[netloc]['t'] += local_counts[netloc]['t']
                    counts_dict_en[netloc]['c'] += local_counts[netloc]['c']
            progress.update(1)

count_df = pd.DataFrame.from_dict(counts_dict_en, orient='index')
count_df.to_csv(f'dolma_v1_6_unfiltered_multithread.csv')

count_df['t'].sum()
count_df.sort_values('t', ascending=False, inplace=True)
count_df.head(20)
