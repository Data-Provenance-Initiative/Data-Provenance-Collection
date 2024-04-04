import os
import sys
import gzip
import shlex
import subprocess
import yaml
import json
import jsonlines
import constants
from ast import literal_eval
import pandas as pd
import typing
from collections import defaultdict
from semanticscholar import SemanticScholar


#############################################################################
############### Local File IO
#############################################################################

def listdir_nohidden(path: str) -> typing.List[str]:
    """Returns all non-hidden files within a directory."""
    assert os.path.exists(path) and os.path.isdir(path)
    return [os.path.join(path, f) for f in os.listdir(path) if not f.startswith(".")]

def read_txt(path: str) ->  typing.List[typing.Any]:
    with open(path, "r", encoding="utf8") as f:
        return [l.strip() for l in f.readlines()]

def write_txt(path: str, data: str):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf8") as outf:
        outf.write(data)

def write_json(data, outpath, compress: bool=False):
    dirname = os.path.dirname(outpath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if compress:
        with gzip.open(outpath, 'wt', encoding='UTF-8') as zipfile:
            zipfile.write(json.dumps(data, ensure_ascii=False, indent=4))
    else:
        with open(outpath, 'w', encoding='utf-8') as outf:
            json.dump(data, outf, ensure_ascii=False, indent=4)


def read_json(inpath: str, verbose=False):
    if verbose:
        print(f"Reading {inpath}...")
    if inpath[-2:] in ["gz", "gzip"]:
        with gzip.open(inpath, 'rb') as fp:
            return json.load(fp)

    with open(inpath, 'rt', encoding='UTF-8') as inf:
        return json.load(inf)

def write_jsonl(
    data: typing.Union[pd.DataFrame, typing.List[typing.Dict]], 
    outpath: str, 
    compress: bool=False, 
    dumps=None,
):
    dirname = os.path.dirname(outpath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if isinstance(data, list):
        if compress:
            with gzip.open(outpath, 'wb') as fp:
                json_writer = jsonlines.Writer(fp)#, dumps=dumps)
                json_writer.write_all(data)
        else:
            with open(outpath, "wb") as fp:
                json_writer = jsonlines.Writer(fp) #, dumps=dumps)
                json_writer.write_all(data)
    else: # Must be dataframe:
        data.to_json(outpath, orient="records", lines=True, compression="gzip" if compress else "infer")

def read_jsonl(inpath: str) -> typing.List[typing.Dict]:
    if inpath[-2:] in ["gz", "gzip"]:
        with gzip.open(inpath, 'rb') as fp:
            j_reader = jsonlines.Reader(fp)
            return [l for l in j_reader]
    else:
        with open(inpath, "rb") as fp:
            j_reader = jsonlines.Reader(fp)
            return [l for l in j_reader]

def read_yaml(inpath: str):
    with open(inpath, 'r') as inf:
        return yaml.safe_load(inf)


#############################################################################
############### Data Provenance I/O Helpers
#############################################################################


def read_data_summary_json(summary_dir: str):
    collection_summaries = []
    for collection_fp in listdir_nohidden(summary_dir):
        if "_template.json" in collection_fp or "_template_spec.yaml" in collection_fp:
            continue
        collection_summaries.extend(list(read_json(collection_fp).values()))
    return collection_summaries
    # return pd.DataFrame(collection_summaries).fillna("")

def read_all_constants():
    license_classes = read_json(constants.LICENSE_CONSTANTS_FP)
    custom_license_classes = read_json(constants.CUSTOM_LICENSE_CONSTANTS_FP)
    language_groups = read_json(constants.LANGUAGE_CONSTANTS_FP)
    task_groups = read_json(constants.TASK_CONSTANTS_FP)
    model_groups = read_json(constants.MODEL_CONSTANTS_FP)
    creator_groups = read_json(constants.CREATOR_CONSTANTS_FP)
    all_formats = read_json(constants.FORMATS_CONSTANTS_FP)
    topic_groups = read_json(constants.TOPIC_CONSTANTS_FP)
    domain_groups = read_json(constants.DOMAINS_CONSTANTS_FP)

    return {
        "LICENSE_CLASSES": license_classes,
        "CUSTOM_LICENSE_CLASSES": custom_license_classes,
        "LANGUAGE_GROUPS": language_groups,
        "TASK_GROUPS": task_groups,
        "MODEL_GROUPS": model_groups,
        "CREATOR_GROUPS": creator_groups,
        "FORMATS": all_formats,
        "TOPICS": topic_groups,
        "DOMAIN_GROUPS": domain_groups,
    }


def get_bibtex_from_paper(paper: str):
    sch = SemanticScholar(timeout=50)
    result = sch.get_paper(paper)
    bibtex = result.citationStyles.get("bibtex", None)
    return bibtex


def write_bib(bibtex_entries: str, append: bool = True, save_dir: str = None):
    """
    Writes BibTeX entries to a .bib file. By default, appends entries to the existing file.

    :param bibtex_entries: The BibTeX entries to write.
    :param append: If True, append entries to the file. If False, overwrite the file.
    :param save_dir: The directory to save the .bib file.
    """
    mode = 'a' if append else 'w'
    dirname = os.path.dirname(save_dir)

    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(save_dir, mode, encoding="utf8") as bib_file:
        bib_file.write(bibtex_entries + '\n\n')
