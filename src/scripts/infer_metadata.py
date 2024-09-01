"""
# usage (within repo root):
python scripts/infer_metadata.py --collections ExpertQA --github_token YOURTOKEN
# Github token info https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28
# You can also omit the github token but it will not fill github fields.
# requirements:
pip install datasets ratelimit humanfriendly jsonlines funcy semanticscholar tenacity -q
# reproduce:
https://colab.research.google.com/drive/1btjynODfCIbuq0c1Wx4FveiM4lWTkWZ_?usp=sharing
"""

import os
import sys
import argparse
from ratelimit import limits
import sys
sys.path.append("src/")
from humanfriendly import parse_size
from helpers import io
from collection_mapper import COLLECTION_FN_MAPPER
from tqdm.auto import tqdm
import copy
import json
import funcy as fc
import functools
from tqdm.auto import tqdm
from collections import defaultdict
import datasets
import yaml
import re
import numpy as np
from bs4 import BeautifulSoup
import requests
import concurrent.futures
import time
import datetime
import functools
from semanticscholar import SemanticScholar
from tenacity import Retrying, RetryError, stop_after_attempt

class edict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __hash__(self):
        return hash(str(sorted(self.items())))
    def __reduce__(self):
        # Return a tuple of class_name_to_call, optional_parameters_to_pass
        return (self.__class__, (copy.deepcopy(dict(self)),))

    def __setstate__(self, state):
        self.update(state)

def parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--collections', nargs='*', default=[], help='List of items for collection')
    parser.add_argument('--sch_api_key', type=str, required=False, help='API key for the scheduler')
    parser.add_argument('--github_token', type=str, required=False, help='Token for GitHub access')
    parser.add_argument('--f', type=str, required=False)
    
    args = parser.parse_args()
    return edict(vars(args))

args = parse_args()

flatten = lambda x: list(fc.flatten(x))
cache = functools.cache


github_license_map = {
    'BSD 3-Clause "New" or "Revised" License': 'BSD 3-Clause License',
    'Creative Commons Attribution 4.0 International': 'CC BY 4.0',
    'Creative Commons Attribution Share Alike 4.0 International': 'CC BY-SA 4.0',
    'Creative Commons Zero v1.0 Universal': 'CC0 1.0',
    'GNU General Public License v3.0': 'GNU Affero General Public License v3.0',
    'Other':''
}

sch = SemanticScholar(api_key=args.sch_api_key,timeout=45) # 100 request per second
date = f' ({datetime.datetime.now().strftime("%B %Y")})'
year = date.split(' ')[-1]

@cache
def get_HF_Popularity(x):
    url = x['Hugging Face URL']
    if not ('hf' in str(url) or 'huggingface' in str(url)):
        return None
    html=requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    if not soup.find('dl'):
        print(f"url error:[{url}]")
        return {}
    downloads = int(soup.find('dl').find('dd').text.replace(',', ''))
    likes = parse_size(soup.find('button', {'title': 'See users who liked this repository'}).text)
    return {f'HF Downloads{date}':downloads, f'HF Likes{date}': likes}

@cache
def get_HF_date(x): 
    url = x['Hugging Face URL']
    if not ('hf' in str(url) or 'huggingface' in str(url)):
        return None
    try:
        html_content = requests.get(f'{url}/commits/main/.gitattributes').text
    except Exception as e:
        print(url, e)
        return {"HF Date":None}
    soup = BeautifulSoup(html_content, 'html.parser')
    commit_dates = soup.find_all('time')
    sorted_dates = sorted([commit_date['datetime'] for commit_date in commit_dates])
    oldest_commit_date = sorted_dates[0][:10] if sorted_dates else None
    return {"HF Date":oldest_commit_date}

def cosmetic(x):
    if not x:
        return x
    return x.replace('_',' ').title().replace('Url','URL').replace('Introduced ','')

@cache
def get_PWC(x):
    url = x['Papers with Code URL']
    if not (type(url)==str and "paperswithcode" in url):
        return None
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    y={}
    for k in ('license_name', 'license_url', 'introduced_date'):
        element = soup.find('input', {'id': f'id_{k}'})
        if element is not None:
            value = element.get('value')
            y[k] = value
    
    y= {f'PwC {cosmetic(k)}': ([v] if 'license' in k else v) for k,v in y.items()}
    y['PwC Description'] = soup.find('meta', {'name': 'description'}).get('content')
    return y

@cache
def get_S2(x):
    id = x['Semantic Scholar Corpus ID']
    default = {'S2 Citation Count': None, 'S2 Publication Date': None}
    #if not id or np.isnan(id):
    #    return default
    try:
        id=int(id)
    except:
        print("no sch id", id,x)
    try:
        for attempt in Retrying(stop=stop_after_attempt(3)):
            with attempt:
                x=sch.get_paper(f'CorpusId:{int(id)}')
    except RetryError:
        print("sch fail", id, x['Unique Dataset Identifier'])
        return default
    
    return {f'S2 Citation Count{date}': x.citationCount, 'S2 Date': x['publicationDate']}

@cache
def get_HF_config(x):
    hf_url = x['Hugging Face URL']
    if "/datasets/" not in str(hf_url):
        return None
    dataset_name=hf_url.split('/datasets/')[-1].rstrip('/')
    try:
        configs = datasets.get_dataset_config_names(dataset_name)
    except (ImportError,FileNotFoundError) as e:
        return print(dataset_name, e)
    
    for a in ['main','all','raw','full','default']+[x['Dataset Name'],x['Dataset Name'].split('_')[-1]]:
        if a in configs:
            configs=[x for x in configs if x.lower()==a.lower()]
            break
    return {'HF Dataset':dataset_name,'HF Config': configs[0]}
  
def get_HF_license(x):
    y=dict()
    if 'license' in dir(x.config_info):
        y['HF Config License']=[x.config_info.license]
    if x['yaml'].get('license',None):
        y['HF Yaml License']=flatten([x['yaml']['license']])
    return y

@cache
def get_HF_yaml(x):
    dataset_name=x['HF Dataset']
    try:
        x=requests.get(f'https://huggingface.co/datasets/{dataset_name}/raw/main/README.md').text.split('---')[1]
    except Exception as e:
        return dict(yaml=dict())
    try:
        d=yaml.safe_load(x)
    except (ScannerError,ParserError):
        d=defaultdict(list)
    if type(d)!=dict:
        d=defaultdict(list)
    return dict(yaml=d)

@cache
def get_HF_config_info(x):
    try:
        return dict(config_info=datasets.get_dataset_config_info(x['HF Dataset'], x['HF Config']).__dict__)
    except:
        return dict(config_info=defaultdict(None))

@cache
def get_github_license_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.findAll('a', attrs={'href': re.compile("license", re.IGNORECASE)})
    if not links or '#' in links[0]['href']:
        return None
    url= 'https://raw.githubusercontent.com'+links[0]['href'].replace('/blob/','/')
    try:
        return "\n".join(requests.get(url).text.split('\n')[:2]).strip()
    except:
        "connexion error"
        
@cache
def get_github_date(url):
    time.sleep(2)
    response = requests.get(f"https://api.github.com/repos/{repo}",
    headers={'User-Agent': 'dpi-academic', 'headers': args.github_token})
    date=response.json().get('created_at','')[:10]
    if not date:
        print(response.text,repo)
    return date


@cache
def get_GH(x):
    url = x['GitHub URL']
    if "https://github.com" not in str(url):
        return {"GitHub License":"", "GitHub Date":""}
    repo = url.split("github.com")[1].split("/tree/")[0].lstrip('/').rstrip('/')
    return {"GitHub License":get_github_license_url(url),'GitHub Date':get_github_date(url)}


@cache
@limits(calls=1000, period=3600)
def get_GH(x):
    url = x['GitHub URL']
    if "https://github.com" not in str(url):
        return {"GitHub License":"", "GitHub Date":""}
    time.sleep(3.6)
    repo = url.split("github.com")[1].split("/tree/")[0].lstrip('/').rstrip('/')
    response = requests.get(f"https://api.github.com/repos/{repo}",
    headers={'User-Agent': 'dpi-academic', 'authorization': f"token {args.github_token}"})
    date=response.json().get('created_at','')[:10]
    license=response.json().get('license',None)
    license = license.get('name','') if license else ''
    topics=response.json().get('topics',[''])
    stars=response.json().get('stargazers_count',0)
    if not date:
        print(response.text,repo)
    return {"GitHub License":github_license_map.get(license,license),
            'GitHub Date':date,
            f'GitHub Stars{date}':stars,
            'GitHub Topics':topics}
    

collection_names = args.collections or list(COLLECTION_FN_MAPPER.keys())


INFERRED_METADATA_KEYS = [
 'HF Dataset', 'HF Config', 'HF Config License', 'HF Yaml License',
 'HF Date',
 f'HF Downloads{date}',
 f'HF Likes{date}',
 'PwC License Name',
 'PwC License URL',
 'PwC Date',
 'PwC Description',
 f'S2 Citation Count{date}',
 'S2 Date',
 'GitHub License',
 'Github Date', f'GitHub Stars{date}', 'GitHub Topics',
] 

def add_inferred_metadata(dataset):
    extended_dataset = copy.copy(dataset)

    functions = [
        get_HF_config,
        get_HF_config_info,
        get_HF_yaml,
        get_HF_Popularity,
        get_HF_license,
        get_HF_date,
        get_PWC,
        get_S2,
        get_GH
    ]

    for getter in functions:
        metadata_subset = getter(edict(extended_dataset))
        metadata_subset = metadata_subset or dict() 
        extended_dataset={**extended_dataset, **metadata_subset}
    metadata = {k:v for k,v in sorted(extended_dataset.items()) if k in INFERRED_METADATA_KEYS}
    for k in INFERRED_METADATA_KEYS:
        metadata[k] = metadata.get(k,'')
    metadata={k: v if v is not None else '' for k, v in metadata.items()}
    metadata={k: v if v is not [None] else [''] for k, v in metadata.items()}    
    metadata = json.loads(json.dumps(metadata, indent=4, sort_keys=True, default=str))

    if 'Inferred Metadata' not in dataset:
        dataset['Inferred Metadata']=dict()
        
    for k in metadata:
        dataset['Inferred Metadata'][k] = metadata[k]

    for k in copy.deepcopy(dataset['Inferred Metadata']):
        if year in k and date not in k:
            del dataset['Inferred Metadata'][k]
    return dataset


collection_summaries = {}
for cname in collection_names:
    cname=cname.replace('Masking','masking')
    collection_summaries[cname] = io.read_json(f"data_summaries/{cname}.json")


fixes={
  "Dataset Name": {
    "BSC-TeMU": "PlanTL-GOB-ES/SQAC",
    "BSC-TeMU/tecla": "projecte-aina/tecla",
    "BSC-TeMU/viquiquad": "projecte-aina/viquiquad",
    'bsc/ancora-ca-ner': 'projecte-aina/ancora-ca-ner'},
  "Papers with Code URL": {
    "https://paperswithcode.com/dataset/tatoeba-translation-challenge": "https://paperswithcode.com/paper/the-tatoeba-translation-challenge-realistic-1"
  }
}
def fix(x):
    x["Dataset Name"]=x['Dataset Name'].rstrip('/')
    d=x['Dataset URL']
    if d:
        x['Dataset URL']=d.split("/viewer/?")[0]
    x['Hugging Face URL']=x['Hugging Face URL'].replace('viewer/?dataset=','')

    for key, sub in fixes.items():
        for k,v in sub.items():
            x[key]=x[key].replace(k,v)
    return x
    
for cname, cname_infos in tqdm(collection_summaries.items()):
    for dataset, dataset_infos in tqdm(cname_infos.items()):
        time.sleep(1.0)
        try:
            cname_infos[dataset]=add_inferred_metadata(fix(dataset_infos))
        except Exception as e:
            di=dataset_infos['Unique Dataset Identifier']
            print("Error:",di,e)
    io.write_json(cname_infos, f"data_summaries/{cname}.json")
