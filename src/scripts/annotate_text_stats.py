import argparse
import numpy as np
import os
import pandas as pd
import sys
import tqdm
from typing import Dict, List
sys.path.append("./")
sys.path.append("src/")

import constants
from helpers import io
from downloader import Downloader
from download_and_filter import get_collection_to_uid_and_filter_ids
from collection_mapper import COLLECTION_FN_MAPPER

DATA_DIR = "data_summaries/"


def compute_text_metrics(dset: list) -> Dict[str, int | float]:
    user_lens, assistant_lens = [], []
    for messages in dset:
        for message in messages:
            if message["from"] == "user":
                user_lens.append(len(message["text"]))
            else:
                assistant_lens.append(len(message["text"]))

    num_dialogs = len(dset)
    dialog_lens = [len(ms) for ms in dset]
    return {
        "Num Dialogs": len(dset),
        "Mean Inputs Length": round(np.mean(user_lens), 4),
        "Mean Targets Length": round(np.mean(assistant_lens), 4),
        "Max Inputs Length": max(user_lens),
        "Max Targets Length": max(assistant_lens),
        "Min Inputs Length": min(user_lens),
        "Min Targets Length": min(assistant_lens),
        "Min Dialog Turns": min(dialog_lens),
        "Max Dialog Turns": max(dialog_lens),
        "Mean Dialog Turns": round(sum(dialog_lens) / num_dialogs, 4),
    }


def annotate_text_statistics(collection_name: str) -> None:
    collection_summary_path = os.path.join(DATA_DIR, f"{collection_name}.json")
    collection_info = io.read_json(collection_summary_path)
    uid_to_filter_ids = {uid: dset_info['Dataset Filter IDs'] for uid, dset_info in collection_info.items()}

    # Load configurations and run the downloader/preparer
    all_collection_infos = {r["Unique Dataset Identifier"]: r for r in io.read_data_summary_json(DATA_DIR)}
    downloader_args = COLLECTION_FN_MAPPER[collection_name]
    data_summary_df = pd.DataFrame(list(all_collection_infos.values())).fillna("")
    uid_task_keys = get_collection_to_uid_and_filter_ids(data_summary_df)[collection_name]
    downloader_args.update({"uid_key_mapper": uid_task_keys})
    downloader = Downloader(
        name=collection_name,
        **downloader_args,
    )

    all_acceptable_filter_ids = [v for vs in uid_to_filter_ids.values() for v in vs]
    full_dset = downloader.download_and_prepare(all_acceptable_filter_ids, debug=True)

    # Update the text statistics
    subset_total_rows = 0
    for duid, dset_info in collection_info.items():
        dset = [row for row in full_dset if row[0]["parent"] == duid]
        assert len(dset) > 0
        subset_total_rows += len(dset)
        text_metrics = compute_text_metrics(dset)
        dset_info.update({"Text Metrics": text_metrics})
    assert subset_total_rows == len(full_dset)

    # Write the updated summaries file back to disk.
    io.write_json(collection_info, collection_summary_path)


def get_collections_missing_metrics() -> List[str]:
    """Gets all collection anmes that are missing text metrics."""
    all_collection_infos = io.read_data_summary_json(DATA_DIR)
    missing_metrics = set()
    for entry in all_collection_infos:
        if "Text Metrics" not in entry:
            missing_metrics.add(entry["Collection"])
    return list(missing_metrics)


def main(collection_names: List[str]) -> None:
    lower_collection_names = [k.lower() for k in COLLECTION_FN_MAPPER.keys()]
    if not collection_names:
        collection_names = get_collections_missing_metrics()
        print(f"No collection names provided, generating stats for {len(collection_names)} unfinished dataset summaries.")
    
    for cname in collection_names:
        print(f"\nGenerating text statistics for {cname}.\n")
        if cname.lower() not in lower_collection_names:
            print(f"{cname.lower()} not in list of collection names.")
            continue
        try:
            annotate_text_statistics(cname)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--collection_names",
        required=False,
        default=None,
        nargs="+",
        help="Name of the collections you'd like to annotate for text topics. "
        " If empty, we annotate all dataset's text stats that are empty."
    )
    args = parser.parse_args()
    main(args.collection_names)
