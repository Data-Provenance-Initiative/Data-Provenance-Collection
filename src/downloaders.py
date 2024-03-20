# import os
import re
import json
# import random
import zipfile
import multiprocessing
import itertools as it

from io import BytesIO
from functools import partial
# from collections import Counter

import chardet
import requests
import pandas as pd

from datasets import load_dataset, Dataset  # ,list_datasets

# `HfFileSystem` requires the latest version of `huggingface_hub`
from huggingface_hub import HfFileSystem, hf_hub_url, hf_hub_download  # , login

from helpers import io


# ##########################################################################
# ############## Download Utils
# ##########################################################################


def filter_dataset_on_task_name(ex, task_key, accepted_filter_ids):
    """ Filters a dataset based on the task name.

    Args:
        ex (dict): A single example from the dataset.
        task_key (str): The key in the example dict that contains the task name.
        acceptable_tasks (list): A list of acceptable task names.

    Returns:
        bool: True if the example's task name is in the acceptable tasks list, False otherwise.
    """
    return ex[task_key] in accepted_filter_ids


def pool_filter(candidates, task_key, accepted_filter_ids):
    """ Filters a list of candidates using multiprocessing.

    Args:
        candidates (list): A list of candidates to filter.
        task_key (str): The key in the example dict that contains the task name.
        acceptable_tasks (list): A list of acceptable task names.

    Returns:
        list: A list of candidates that passed the filter.
    """
    with multiprocessing.Pool() as pool:
        return [
            c for c, keep in zip(candidates, pool.map(
                partial(
                    filter_dataset_on_task_name,
                    task_key=task_key,
                    accepted_filter_ids=accepted_filter_ids
                ), candidates)) if keep
        ]


def annotate_source(dset, source):
    updated_dset = []
    for row in dset:
        row["_source"] = source
        updated_dset.append(row)
    return updated_dset


def direct_data_request(url):
    response = requests.get(url)
    # The response content is in bytes, we need to convert it to string
    content = response.content.decode('utf-8')
    # Now we can parse the JSON content into a Python list/dictionary
    return json.loads(content)


def huggingface_download(
        data_address, name=None, data_dir=None, data_files=None, split=None
):
    """Download a dataset from the Hugging Face Hub.

    It supports various options for specifying the dataset to download,
    such as providing a name, a data directory, data files, or a split.

    Args:
        data_address (str): The address or identifier of the dataset
        name (str, optional): Name of the dataset to download. Defaults to None.
        data_dir (str, optional): Path to the directory containing the dataset files. Defaults to None.
        data_files (str or list, optional): Path(s) to specific dataset files. Defaults to None.
        split (str, optional): Name of the split to take (usually "train"). Defaults to None.

    Returns:
        list or Dataset: The downloaded dataset as a list of items,
            or Hugging Face Dataset object (if failed converted to list).
    """
    assert not (data_dir and data_files)

    num_proc = max(multiprocessing.cpu_count() // 2, 1)
    if data_files:
        dset = load_dataset(data_address, data_files=data_files, num_proc=num_proc)
    elif data_dir:
        dset = load_dataset(data_address, data_dir=data_dir, num_proc=num_proc)
    elif name:
        dset = load_dataset(data_address, name)
    else:
        dset = load_dataset(data_address, num_proc=num_proc)

    if split:
        dset = dset[split]

    try:
        dset = dset.to_list()
    except ValueError:
        print("Trouble converting Hugging Face dataset to list...")
        pass
    return dset


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    return result['encoding']


def convert_to_utf8(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    utf8_content = content.encode('utf-8')
    return json.loads(utf8_content.decode('utf-8'))


def process_zipped_file(zip_file):
    dset = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        for json_file in z.namelist():
            if json_file.endswith(".json"):
                data = json.load(z.open(json_file))
                dset.append(data)
    return dset


# ##########################################################################
# ############## Collection Downloader Functions
# ##########################################################################


def download_flan_collection_sni(accepted_filter_ids):
    dset = huggingface_download("DataProvenanceInitiative/niv2_submix_original", split="train")
    return pool_filter(dset, "task_name", accepted_filter_ids)


def download_flan_collection_cot(accepted_filter_ids):
    dset = huggingface_download("DataProvenanceInitiative/cot_submix_original", split="train")
    return pool_filter(dset, "task_name", accepted_filter_ids)


def download_flan_collection_dialog(accepted_filter_ids):
    dset = huggingface_download("DataProvenanceInitiative/dialog_submix_original", split="train")
    return pool_filter(dset, "task_name", accepted_filter_ids)


def download_flan_collection_flan2021(accepted_filter_ids):
    dset = huggingface_download("DataProvenanceInitiative/flan2021_submix_original", split="train")
    return pool_filter(dset, "task_name", accepted_filter_ids)


def download_flan_collection_p3(accepted_filter_ids):
    dset = huggingface_download("DataProvenanceInitiative/t0_submix_original", split="train")
    return pool_filter(dset, "task_name", accepted_filter_ids)


def download_xp3x(accepted_filter_ids, sample_threshold=100):
    # The more accepted_filter_ids the longer it will take. So if it's too many switch to the sample (likely e.g. when people just choose everything).
    # Meanwhile if people just choose a few ids, maybe just one language or so, then use the big one.
    if len(accepted_filter_ids) > sample_threshold:
        print(f"xP3x: Detected {len(accepted_filter_ids)} filter IDs. Defaulting to xP3x-sample to reduce download size. Increase sample_threshold to download full dataset.")
        return download_xp3x_sample(accepted_filter_ids)

    fs = HfFileSystem()
    fps = [
        (task, fs.resolve_path(fp)) for task in accepted_filter_ids
        for fp in fs.glob(f"datasets/Muennighoff/xP3x/data/{task.split('/')[0]}/*{task.split('/')[1].replace('-', '_')}*")
    ]

    data_files = [
        hf_hub_url(
            resolved_path.repo_id,
            resolved_path.path_in_repo,
            repo_type=resolved_path.repo_type)
        for (task, resolved_path) in fps
    ]

    dset = []
    if data_files:
        dset = huggingface_download("json", data_files=data_files, split="train")
        dset = pd.DataFrame(dset).to_dict('records')
    return dset


def download_xp3x_sample(accepted_filter_ids):
    dset = []
    tasks = set([task.split('/')[1].lower().replace("-", "_") for task in accepted_filter_ids])
    langs = list(set([task.split('/')[0] for task in accepted_filter_ids]))

    for t in tasks:
        raw_dset = huggingface_download('Muennighoff/xP3x-sample', t, split='train')
        raw_dset = pool_filter(raw_dset, "language", langs)
        dset.extend(raw_dset)

    return dset


def download_commitpackft(accepted_filter_ids):
    dset = []
    for lang in accepted_filter_ids:
        raw_dset = huggingface_download('bigcode/commitpackft', lang, split='train')
        dset.extend(raw_dset)
    return dset


def download_dolly_15k(accepted_filter_ids):
    dset = huggingface_download("databricks/databricks-dolly-15k", split="train")
    return pool_filter(dset, "category", accepted_filter_ids)

def download_thai_gen_ai_dolly(accepted_filter_ids):
    dset = huggingface_download("Thaweewat/databricks-dolly-15k-th", split="train")
    return pool_filter(dset, "category", accepted_filter_ids)

def download_laion_oig(accepted_filter_ids):
    dsets = []
    for dset_name in accepted_filter_ids:
        dset = huggingface_download("laion/oig", data_files=f"{dset_name}.jsonl", split="train")
        # annotate each example with source
        dset = annotate_source(dset, dset_name)
        dsets.extend(dset)
    return dsets

def download_capybara(accepted_filter_ids):
    dset = huggingface_download('LDJnr/Capybara', split='train')
    return pool_filter(dset, "source", accepted_filter_ids)

def download_self_instruct(accepted_filter_ids):
    return huggingface_download('yizhongw/self_instruct', split='train')


def download_everything_lm(accepted_filter_ids):
    return huggingface_download('totally-not-an-llm/EverythingLM-data-V3', split='train')


def download_anthropic_hh_rlhf(accepted_filter_ids):
    return huggingface_download('anthropic/hh-rlhf', split='train')

def download_thai_gen_ai_alpaca(accepted_filter_ids):
    return huggingface_download('Thaweewat/alpaca-cleaned-52k-th', split='train')

def download_stanford_human_preferences(accepted_filter_ids):
    dset = huggingface_download('stanfordnlp/SHP', split='train')
    return pool_filter(dset, "domain", accepted_filter_ids)


def download_open_assistant(accepted_filter_ids):
    dset = huggingface_download("OpenAssistant/oasst1", split='train')
    return pool_filter(dset, "lang", accepted_filter_ids)


def download_open_assistant_octopack(accepted_filter_ids):
    return huggingface_download("bigcode/oasst-octopack", split='train')


def download_longform(accepted_filter_ids):
    # Intentionally omitting BigBench.
    dset = huggingface_download("akoksal/LongForm", split='train')
    return pool_filter(dset, "source", accepted_filter_ids)


def download_gpteacher(accepted_filter_ids):
    dset = []
    if "gpteacher_instruct" in accepted_filter_ids:
        instruct_dset = huggingface_download("teknium/GPTeacher-General-Instruct", split='train')
        dset += annotate_source(instruct_dset, "gpteacher_instruct")
    if "gpteacher_codegen" in accepted_filter_ids:
        codegen_dset = direct_data_request("https://raw.githubusercontent.com/teknium1/GPTeacher/main/Codegen/codegen-instruct.json")
        dset += annotate_source(codegen_dset, "gpteacher_codegen")
    if "gpteacher_toolformer" in accepted_filter_ids:
        toolformer_dset = direct_data_request("https://raw.githubusercontent.com/teknium1/GPTeacher/main/Toolformer/toolformer-dedupe-only-dataset.json")
        dset += annotate_source(toolformer_dset, "gpteacher_toolformer")
    if "gpteacher_roleplay" in accepted_filter_ids:
        roleplay_dset = direct_data_request("https://raw.githubusercontent.com/teknium1/GPTeacher/main/Roleplay/roleplay-simple-deduped-roleplay-instruct.json")
        dset += annotate_source(roleplay_dset, "gpteacher_roleplay")
    return dset


def download_baize_data(accepted_filter_ids):
    dset = []
    if "alpaca_chat_data" in accepted_filter_ids:
        alpaca_chat_dataset = direct_data_request("https://raw.githubusercontent.com/project-baize/baize-chatbot/main/data/alpaca_chat_data.json")
        dset += annotate_source(alpaca_chat_dataset, "alpaca_chat_data")

    if "medical_chat_data" in accepted_filter_ids:
        medical_chat_dataset = direct_data_request("https://raw.githubusercontent.com/project-baize/baize-chatbot/main/data/medical_chat_data.json")
        dset += annotate_source(medical_chat_dataset, "medical_chat_data")

    if "quora_chat_data" in accepted_filter_ids:
        quora_chat_dataset = direct_data_request("https://raw.githubusercontent.com/project-baize/baize-chatbot/main/data/quora_chat_data.json")
        dset += annotate_source(quora_chat_dataset, "quora_chat_data")

    if "stackoverflow_chat_data" in accepted_filter_ids:
        stackoverflow_chat_dataset = direct_data_request("https://raw.githubusercontent.com/project-baize/baize-chatbot/main/data/stackoverflow_chat_data.json")
        dset += annotate_source(stackoverflow_chat_dataset, "stackoverflow_chat_data")
    return dset


def download_openai_summarization(accepted_filter_ids):
    # we don't need any filtering because comparisons>train contains rows only from Reddit TL;DR dataset.
    return huggingface_download("openai/summarize_from_feedback", name="comparisons", split="train")


def download_openai_webgpt(accepted_filter_ids):
    dset = huggingface_download("openai/webgpt_comparisons", split="train")
    # we copy the dataset key from `question::dataset` as a top-level key in each instance
    for ex in dset:
        ex["dataset"] = ex["question"]["dataset"]
    return pool_filter(dset, "dataset", accepted_filter_ids)


def download_alpaca(accepted_filter_ids):
    return huggingface_download('tatsu-lab/alpaca', split='train')

def download_metamathqa(accepted_filter_ids):
    dset = huggingface_download('meta-math/MetaMathQA', split='train')
    return pool_filter(dset, "type", accepted_filter_ids)

def download_pure_dove(accepted_filter_ids):
    return huggingface_download('LDJnr/Pure-Dove', split='train')

def download_evol_instruct(accepted_filter_ids):
    return huggingface_download('WizardLM/evol_instruct_70k', split='train')

def download_llama2_med_tuned_instructions(accepted_filter_ids):
    return huggingface_download('nlpie/Llama2-MedTuned-Instructions', split='train')

def download_sharegpt_vicuna(accepted_filter_ids):
    sharegpt_dir = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    sv_dset_p1 = hf_hub_download(
        repo_id=sharegpt_dir,
        filename="sg_90k_part1_html_cleaned.json",
        subfolder="HTML_cleaned_raw_dataset",
        repo_type="dataset"
    )
    sv_dset_p2 = hf_hub_download(
        repo_id=sharegpt_dir,
        filename="sg_90k_part1_html_cleaned.json",
        subfolder="HTML_cleaned_raw_dataset",
        repo_type="dataset"
    )
    return pd.concat(
        [pd.read_json(sv_dset_p1), pd.read_json(sv_dset_p2)]
    ).to_dict('records')


def download_code_alpaca(accepted_filter_ids):
    return huggingface_download("sahil2801/CodeAlpaca-20k", split="train")


def download_hc3_en(accepted_filter_ids):
    dset_fpath = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename="all.jsonl",
        repo_type="dataset"
    )
    dset = pd.read_json(dset_fpath, lines=True).to_dict('records')
    return pool_filter(dset, "source", accepted_filter_ids)


def download_hc3_zh(accepted_filter_ids):
    dset_fpath = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3-Chinese",
        filename="all.jsonl",
        repo_type="dataset"
    )
    dset = pd.read_json(dset_fpath, lines=True).to_dict('records')
    return pool_filter(dset, "source", accepted_filter_ids)


def download_camel_science(accepted_filter_ids):
    dset = []
    if "physics" in accepted_filter_ids:
        physics_zip = hf_hub_download(
            repo_id="camel-ai/physics",
            filename="physics.zip",
            repo_type="dataset"
        )
        physics_dset = process_zipped_file(physics_zip)
        dset += annotate_source(physics_dset, "physics")
    if "chemistry" in accepted_filter_ids:
        chemistry_zip = hf_hub_download(
            repo_id="camel-ai/chemistry",
            filename="chemistry.zip",
            repo_type="dataset"
        )
        chemistry_dset = process_zipped_file(chemistry_zip)
        dset += annotate_source(chemistry_dset, "chemistry")
    if "biology" in accepted_filter_ids:
        biology_zip = hf_hub_download(
            repo_id="camel-ai/biology",
            filename="biology.zip",
            repo_type="dataset"
        )
        biology_dset = process_zipped_file(biology_zip)
        dset += annotate_source(biology_dset, "biology")
    return dset


def download_cot_collection(accepted_filter_ids):
    dset = []
    for lang in accepted_filter_ids:
        if lang == "en":
            raw_dset = huggingface_download('kaist-ai/CoT-Collection', split='train')
            dset.extend(annotate_source(raw_dset, "en"))
        else:
            fpath = hf_hub_download(
                repo_id='kaist-ai/CoT-Collection_multilingual',
                filename=f"CoT_collection_{lang}.json",
                subfolder="data",
                repo_type="dataset",
            )
            raw_dset = [row for _, row in io.read_json(fpath).items()]
            raw_dset = annotate_source(raw_dset, lang)
            dset.extend(raw_dset)
    return dset


def download_gpt4all(accepted_filter_ids):
    dset = huggingface_download('nomic-ai/gpt4all-j-prompt-generations', split='train')
    return pool_filter(dset, "source", accepted_filter_ids)


def download_evol_instruct_v2(accepted_filter_ids):
    return huggingface_download('WizardLM/WizardLM_evol_instruct_V2_196k', split='train')


def download_gpt4_alpaca(accepted_filter_ids):
    return huggingface_download('teknium/GPT4-LLM-Cleaned', split='train')


def download_tasksource_instruct(accepted_filter_ids):
    dset = huggingface_download('tasksource/tasksource-instruct-v0', split='train')
    return pool_filter(dset, "task", accepted_filter_ids)


def download_tasksource_symbol_tuning(accepted_filter_ids):
    dset = huggingface_download('tasksource/icl-symbol-tuning-instruct', split='train')
    return pool_filter(dset, "task", accepted_filter_ids)


def download_stack_exchange_instruction(accepted_filter_ids):
    return huggingface_download('ArmelR/stack-exchange-instruction', split='train')


def download_unnatural_instructions(accepted_filter_ids):
    return huggingface_download('mrm8488/unnatural-instructions', name='core', split='train')


def download_starcoder_self_instruct(accepted_filter_ids):
    return huggingface_download('codeparrot/self-instruct-starcoder', split='curated')

def download_thai_gen_ai_gpteacher(accepted_filter_ids):
    return huggingface_download('Thaweewat/gpteacher-20k-th', split='train')

def download_tiny_stories(accepted_filter_ids):
    return huggingface_download('roneneldan/TinyStoriesInstruct', split='train')


def download_joke_explanation(accepted_filter_ids):
    return huggingface_download('theblackcat102/joke_explaination', split='train')


def download_book_summaries(accepted_filter_ids):
    dset = huggingface_download('emozilla/booksum-summary-analysis_gptneox-8192', split='train')
    return pool_filter(dset, "type", accepted_filter_ids)


def download_pii_masking_200k(accepted_filter_ids):
    return huggingface_download('ai4privacy/pii-masking-200k', split='train')


def download_ultrachat(accepted_filter_ids):
    return huggingface_download('stingning/ultrachat', split='train')

def download_wildchat(accepted_filter_ids):
    '''downloads in the wild chat dataset from hugging face'''
    dset = huggingface_download('allenai/WildChat', split='train')
    return pool_filter(dset, "model", accepted_filter_ids)


def download_airoboros(accepted_filter_ids):
    dset_fpath = hf_hub_download(
        repo_id="jondurbin/airoboros-gpt4-1.2",
        filename="as_conversations.json",
        repo_type="dataset"
    )
    dset = convert_to_utf8(dset_fpath)
    return dset


def download_lima(accepted_filter_ids):
    dset = huggingface_download('GAIR/lima', split='train')
    return pool_filter(dset, "source", accepted_filter_ids)


def download_open_orca(accepeted_filter_ids):
    dset = huggingface_download('Open-Orca/OpenOrca', split='train')
    dset = list(map(lambda x: {**x, 'source': x['id'].split('.')[0]}, dset))
    return pool_filter(dset, "source", accepeted_filter_ids)


def download_pmc_llama(accepted_filter_ids):
    dset = huggingface_download("axiong/pmc_llama_instructions", split="train")
    return pool_filter(dset, "source", accepted_filter_ids)
  
  
def download_medical_meadow(accepted_filter_ids):
    dset = []
    if "medical-meadow-med-flashcards" in accepted_filter_ids:
        med_flashcards = huggingface_download("medalpaca/medical_meadow_medical_flashcards", split='train')
        dset += annotate_source(med_flashcards, "medical-meadow-med-flashcards")
    if "medical-meadow-wikidoc-living-textbook" in accepted_filter_ids:
        wikidoc_living_textbook = huggingface_download("medalpaca/medical_meadow_wikidoc", split='train')
        dset += annotate_source(wikidoc_living_textbook, "medical-meadow-wikidoc-living-textbook")
    if "medical-meadow-wikidoc-patient-information" in accepted_filter_ids:
        wikidoc_patient_information = huggingface_download("medalpaca/medical_meadow_wikidoc_patient_information", split='train')
        dset += annotate_source(wikidoc_patient_information, "medical-meadow-wikidoc-patient-information")
    if "medical-meadow-cord19" in accepted_filter_ids:
        cord19 = huggingface_download("medalpaca/medical_meadow_cord19", split='train')
        dset += annotate_source(cord19, "medical-meadow-cord19")
    if "medical-meadow-health-advice" in accepted_filter_ids:
        health_advice = huggingface_download("medalpaca/medical_meadow_health_advice", split='train')
        dset += annotate_source(health_advice, "medical-meadow-health-advice")
    if "medical-meadow-pubmed-causal" in accepted_filter_ids:
        pubmed_causal = huggingface_download("medalpaca/medical_meadow_pubmed_causal", split='train')
        dset += annotate_source(pubmed_causal, "medical-meadow-pubmed-causal")
    if "medical-meadow-medqa" in accepted_filter_ids:
        medqa = huggingface_download("medalpaca/medical_meadow_medqa", split='train')
        dset += annotate_source(medqa, "medical-meadow-medqa")
    if "medical-meadow-mediqa" in accepted_filter_ids:
        mediqa = huggingface_download("medalpaca/medical_meadow_mediqa", split='train')
        dset += annotate_source(mediqa, "medical-meadow-mediqa")
    return dset

  
def download_medinstruct(accepted_filter_ids):
    return direct_data_request("https://raw.githubusercontent.com/XZhang97666/AlpaCare/master/data/MedInstruct-52k.json")


def download_mathinstruct(accepted_filter_ids):
    mathinstruct = load_dataset('TIGER-Lab/MathInstruct', split='train')
    dset = []
    if 'cot_MATH_train' in accepted_filter_ids:
        cot_MATH_train = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/MATH_train.json').to_list()
        dset += annotate_source(cot_MATH_train, 'cot_MATH_train')
    if 'cot_TheoremQA' in accepted_filter_ids:
        cot_TheoremQA = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/TheoremQA.json').to_list()
        dset += annotate_source(cot_TheoremQA, 'cot_TheoremQA')
    if 'cot_aqua_rat' in accepted_filter_ids:
        cot_aqua_rat = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/aqua_rat.json').to_list()
        dset += annotate_source(cot_aqua_rat, 'cot_aqua_rat')
    if 'cot_college_math' in accepted_filter_ids:
        cot_college_math = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/college_math.json').to_list()
        dset += annotate_source(cot_college_math, 'cot_college_math')
    if 'cot_gsm_rft' in accepted_filter_ids:
        cot_gsm_rft = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/gsm_rft.json').to_list()
        dset += annotate_source(cot_gsm_rft, 'cot_gsm_rft')
    if 'cot_gsm_train' in accepted_filter_ids:
        cot_gsm_train = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/gsm_train.json').to_list()
        dset += annotate_source(cot_gsm_train, 'cot_gsm_train')
    if 'cot_math50k_camel' in accepted_filter_ids:
        cot_math50k_camel = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/math50k_camel.json').to_list()
        dset += annotate_source(cot_math50k_camel, 'cot_math50k_camel')
    if 'cot_number_comparison' in accepted_filter_ids:
        cot_number_comparison = mathinstruct.filter(lambda row: row['source'] == 'data/CoT/number_comparison.json').to_list()
        dset += annotate_source(cot_number_comparison, 'cot_number_comparison')
    if 'pot_MATH_train' in accepted_filter_ids:
        pot_MATH_train = mathinstruct.filter(lambda row: row['source'] == 'data/PoT/MATH_train.json').to_list()
        dset += annotate_source(pot_MATH_train, 'pot_MATH_train')
    if 'pot_TheoremQA' in accepted_filter_ids:
        pot_TheoremQA = mathinstruct.filter(lambda row: row['source'] == 'data/PoT/TheoremQA.json').to_list()
        dset += annotate_source(pot_TheoremQA, 'pot_TheoremQA')
    if 'pot_aqua_rat_filtered' in accepted_filter_ids:
        pot_aqua_rat_filtered = mathinstruct.filter(lambda row: row['source'] == 'data/PoT/aqua_rat_filtered.json').to_list()
        dset += annotate_source(pot_aqua_rat_filtered, 'pot_aqua_rat_filtered')
    if 'pot_gsm_gpt4' in accepted_filter_ids:
        pot_gsm_gpt4 = mathinstruct.filter(lambda row: row['source'] == 'data/PoT/gsm_gpt4.json').to_list()
        dset += annotate_source(pot_gsm_gpt4, 'pot_gsm_gpt4')
    if 'pot_mathqa' in accepted_filter_ids:
        pot_mathqa = mathinstruct.filter(lambda row: row['source'] == 'data/PoT/mathqa.json').to_list()
        dset += annotate_source(pot_mathqa, 'pot_mathqa')
    if 'pot_numglue' in accepted_filter_ids:
        pot_numglue = mathinstruct.filter(lambda row: row['source'] == 'data/PoT/numglue.json').to_list()
        dset += annotate_source(pot_numglue, 'pot_numglue')
    return dset   

def split_by_user(pairs):
    '''
    Group (user, value) pairs into lists of pairs with the same user value
    '''

    groups = []
    current_group = []

    for kind, value in pairs:
        if kind == 'user':
            if current_group:
                groups.append(current_group)
            current_group = [(kind, value)]
        else:
            current_group.append((kind, value))

    if current_group:
        groups.append(current_group)

    return groups


def download_tool_llama(accepted_filter_ids):
    '''
    Download Tool-Llama data and parse into (context, instruction, response)
    triples, with system prompts put into the context slots. The return value
    is a huggingface dataset rather than a list of tuples or dicts.
    '''

    # The data is distributed as a Google Drive file in the Github readme,
    # rather than via e.g. Huggingface
    url = "https://docs.google.com/uc"
    params = {
        'export': 'download',
        'id': '1Vis-RxBstXLKC1W1agIQUJNuumPJrrw0',
        'confirm': 'yes'
    }

    response = requests.get(url, params=params, verify=False, stream=True)
    response.raise_for_status()

    # Docs describe a directory hierarchy in this zip file containing all the
    # tasks, and says their training splits are consolidated in this one file
    with zipfile.ZipFile(BytesIO(response.content), 'r') as z:
        with z.open("data/toolllama_G123_dfs_train.json", 'r') as f:
            data = json.load(f)

    # This is possibly multi-turn dialog, though often it's only one turn
    tmp = []
    for line in data:
        line = line['conversations']
        assert line[0]['from'] == 'system'
        assert line[1]['from'] == 'user'

        context = line[0]['value'].strip()

        # we have arbitrary {'from': user, 'value': text} stuff, sometimes with
        # the same user speaking twice in a row, sometimes multi-turn,
        # sometimes single-turn; we want to group it up to turns: lists s.t.
        # the first element is a user statement and all subsequent statements
        # are assistant responses
        groups = it.groupby(line[1:], key=lambda s: s['from'])
        groups = [
            [g[0], '\n'.join(s['value'] for s in g[1])]
            for g in groups
        ]
        groups = split_by_user(groups)

        for group in groups:
            assert group[0][0] == 'user'
            instruction = group[0][1].strip()

            response = '\n'.join(g[1] for g in group[1:]).strip()

            tmp += [{
                'context': context,
                'instruction': instruction,
                'response': response,
            }]

    ret = {}
    for key in tmp[0].keys():
        for ex in tmp:
            ret[key] = ret.get(key, []) + [ex[key]]

    return Dataset.from_dict(ret)


def download_gorilla(accepted_filter_ids):
    '''
    Download Gorilla data from Huggingface, handle string manipulation issues,
    return a Huggingface dataset.
    '''

    # Something went wrong with the formatting in these source files and
    # datasets.load_dataset dies complaining of some obscure error. On
    # inspection it turns out that the problem is inconsistent formatting of
    # the 'code' attribute; we can handle it fine for purposes of getting
    # (instruction, response) pairs with some regex and string manipulation.
    base_url = 'https://huggingface.co/datasets/gorilla-llm/APIBench/resolve/main/'
    files = ['torchhub_train.json', 'tensorflow_train.json', 'huggingface_train.json']

    tmp = []
    for file in files:
        url = base_url + file
        txt = requests.get(url).text

        for line in txt.splitlines():
            line = json.loads(line)
            code = line['code'].strip()
            response = line['api_call']

            if not code.startswith('###'):  # torchhub
                rx = re.compile(r"'Instruction': (.*), 'Output'.*")
                instruction = re.search(rx, code).group(1)
            else:  # tensorflow, huggingface
                instruction = code.splitlines()[0].replace('###Instruction: ', '')
            instruction = instruction.lstrip("'").rstrip("'").strip()

            # this passes but let's not have assertions in production
            # assert len(instruction) > 0

            tmp += [{
                'instruction': instruction,
                'response': response,
            }]

    ret = {}
    for key in tmp[0].keys():
        for ex in tmp:
            ret[key] = ret.get(key, []) + [ex[key]]

    return Dataset.from_dict(ret)

def download_chatdoctor(accepted_filter_ids):
    dset = []
    if "chatdoctor-healthcaremagic-100k" in accepted_filter_ids:
        healthcaremagic_dset = huggingface_download("lavita/ChatDoctor-HealthCareMagic-100k", split='train')
        dset += annotate_source(healthcaremagic_dset, "chatdoctor-healthcaremagic-100k")
    if "chatdoctor-icliniq-10k" in accepted_filter_ids:
        icliniq_dset = load_dataset("lavita/ChatDoctor-iCliniq", split='train')
        icliniq_dset = icliniq_dset.rename_column("answer_icliniq", "output")
        icliniq_dset = icliniq_dset.to_list()
        dset += annotate_source(icliniq_dset, "chatdoctor-icliniq-10k")
    if "chatdoctor-genmedgpt-5k" in accepted_filter_ids:
        genmedgpt_dset = huggingface_download("wangrongsheng/GenMedGPT-5k-en", split='train')
        dset += annotate_source(genmedgpt_dset, "chatdoctor-genmedgpt-5k")
    return dset

def download_agentinstruct(accepted_filter_ids):
    dset = []
    if 'alfworld' in accepted_filter_ids:
        dset.append(huggingface_download('THUDM/AgentInstruct', split='alfworld'))
    if 'db' in accepted_filter_ids:
        dset.append(huggingface_download('THUDM/AgentInstruct', split='db'))
    if 'os' in accepted_filter_ids:
        dset.append(huggingface_download('THUDM/AgentInstruct', split='os'))
    if 'kg' in accepted_filter_ids:
        dset.append(huggingface_download('THUDM/AgentInstruct', split='kg'))
    if 'webshop' in accepted_filter_ids:
        dset.append(huggingface_download('THUDM/AgentInstruct', split='webshop'))
    if 'mind2web' in accepted_filter_ids:
        dset.append(huggingface_download('THUDM/AgentInstruct', split='mind2web'))

    return dset


def download_open_platypus(accepted_filter_ids):
    dset = huggingface_download("garage-bAInd/Open-Platypus", split="train")
    return pool_filter(dset, "data_source", accepted_filter_ids)

  
def download_bactrianx(accepted_filter_ids):
    """Download Bactrian-X dataset from HuggingFace"""
    dsets = []
    for dset_name in accepted_filter_ids:
        dset = huggingface_download('MBZUAI/Bactrian-X', name=dset_name, split="train")
        # annotate each example with source
        dset = annotate_source(dset, dset_name)
        dsets.extend(dset)
    return dsets
