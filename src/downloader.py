import os
# import pandas as pd
# from functools import partial
from collections import defaultdict, Counter
# from datasets import load_dataset, list_datasets
from helpers import io
import random

import multiprocessing


class Downloader:
    """Downloads, filters, prepares, then saves the data for a Collection."""

    def __init__(
        self,
        name,
        download_function,
        prepare_function,
        uid_key_mapper,
        custom_prepare=False,
    ):
        """
        name: Name of the Collection.
        download_function: Data Downloader function from `src/downloaders.py`
        prepare_function: Data Preparer function from `src/preparers.py`
        uid_key_mapper: Mapping the dataset Unique Identifiers to the Dataset Filter IDs used
            to select data subsets from the downloaded collection.
        custom_prepare: Whether the `prepare_fn` takes in the dataset or a single row.
            If True the `prepare_function` takes in the whole dataset from `download_function`.
            If False the `prepare_function` takes in one row at a time from `download_function`, so we can run in parallel.
        """
        self.name = name
        self.download_fn = download_function
        self.prepare_fn = prepare_function
        self.custom_prepare = custom_prepare

        # Allows us to map from keys back to Dataset UID so we can
        # track which dataset they came from:
        self.keys_to_uid = {v: k for k, vs in uid_key_mapper.items() for v in vs}

    def download_and_prepare(
        self,
        accepted_filter_ids,
        limit=None,
        debug=False,
    ):
        dset = self.download_fn(accepted_filter_ids)
        if self.custom_prepare:
            # In some cases we need to preprocess the whole dataset together
            prepared_dset = self.prepare_fn(dset)
        elif debug:
            # Easier to debug when not in parallel.
            prepared_dset = [self.prepare_fn(ex) for ex in dset]
        else:
            # Run in parallel by default once working.
            prepared_dset = self._pool_process(self.prepare_fn, dset)

        # If specified, randomly sample 'limit' dialogs.
        if limit and limit < len(prepared_dset):
            prepared_dset = random.sample(prepared_dset, limit)

        # Map the "parent" field back to the UIDs of the originating dataset
        normalized_dset = []
        for row in prepared_dset:
            new_row = []
            for message in row:
                assert isinstance(message["parent"], int) or message["parent"] in self.keys_to_uid, \
                    f"The `parent` field of the first message in a dialog must be from one of the `Dataset Filter Ids`, specified in the json. It is currently {message['parent']} when the options are {self.keys_to_uid.keys()}"
                if message["parent"] in self.keys_to_uid:
                    message["parent"] = self.keys_to_uid[message["parent"]]

                new_row.append(message)
            normalized_dset.append(new_row)

        return normalized_dset

    def run_and_save(
        self,
        accepted_filter_ids,
        savedir,
        limit=None,
        reformat="messages",
        debug=False,
    ):
        """Runs the data pipeline for this collection:

        Downloads --> Prepares --> Reformats (optional) --> Samples (optional) --> Saves.

        accepated_filter_ids: A list of every `dataset_filter_ids` (from each dataset yaml files)
            in the collection, that passes the metadata filters. Used to extract the relevant subset
            of data from the downloaded dataset.
        savedir: What directory to save the collection in.
        limit: Samples `limit` random samples from this collection. Takes all data if `None`.
        reformat: What format for the output data. Options are [`messages`, `supervised`].
            Default (`messages`) reformat is described here: TODO.
        debug: Turns of data parallelism so errors are easier to debug.

        Saves the data as a gzipped jsonlines file according to `format` argument.
        """
        prepared_dset = self.download_and_prepare(accepted_filter_ids, limit=limit, debug=debug)

        num_messages = sum([len(dialog) for dialog in prepared_dset])
        print(f"{self.name} -- Downloaded {len(prepared_dset)} dialogs, totaling {num_messages} messages.")

        # If specified, reformat dataset for supervised learning, multi-turn dialogs, or reward modeling.
        if reformat == "supervised":
            prepared_dset = self._reformat_supervised(prepared_dset)
        elif reformat == "concat_dialog":
            prepared_dset = self._reformat_concat_dialog(prepared_dset)

        # save.
        savepath = os.path.join(savedir, f"{self.name}.jsonl.gz")
        io.write_jsonl(prepared_dset, savepath, compress=True)

    def _pool_process(self, func, exs):
        """Applies a function (func) in parallel to every item in a list (exs).
        We use this to apply the `prepare_fn` to every row (example/dialog) in a dataset.
        """
        with multiprocessing.Pool() as pool:
            return [proc_ex for proc_ex in pool.map(func, exs)]

    def _reformat_supervised(self, dialogs):
        # TODO: Parallelize
        reformatted = []
        for dialog in dialogs:
            pairs = self._reformat_supervised_dialog(dialog)
            reformatted.extend(pairs)
        return reformatted

    def _reformat_supervised_dialog(self, dialog):
        dset_name = dialog[0]["parent"]
        pairs = []

        # Create an adjacency list based on 'parent' key
        adjacency_list = defaultdict(list)
        for i, msg in enumerate(dialog):
            adjacency_list[msg['parent']].append(i)

        # Recursive function to do a DFS and create pairs
        def dfs(node_id, parent_msg):
            if node_id not in adjacency_list:
                return
            for child_id in adjacency_list[node_id]:
                child_msg = dialog[child_id]
                # If current node ('child') is an assistant's message and parent node is a user's message
                if child_msg['from'] == 'assistant' and parent_msg['from'] == 'user':
                    # If there is a score field, it must be >= 1 to use this entry.
                    if 'score' not in child_msg or child_msg['score'] >= 1:
                        pairs.append({
                            'inputs': parent_msg['text'], 'targets': child_msg['text'], "dataset": dset_name,
                        })
                # Call dfs for the child
                dfs(child_id, child_msg)

        # Start dfs for each root (the nodes whose parent is the name of the dataset)
        for root_id in adjacency_list[dset_name]:
            dfs(root_id, dialog[root_id])

        return pairs

    def _reformat_concat_dialog(self, dialogs):
        reformatted = []
        lens = []
        for dialog in dialogs:
            full_dialog = []
            parent = None
            for idx, message in enumerate(dialog):
                if parent is None:
                    full_dialog.append(message["text"])
                    parent = 0
                elif message["parent"] == parent:
                    full_dialog.append(message["text"])
                    parent = idx
            lens.append(len(full_dialog))
            reformatted.append({
                "inputs": "\n".join(full_dialog),
                "targets": "",
                "dataset": dialog[0]["parent"],
            })

        # print(Counter(lens))
        # print(reformatted[0])

        return reformatted
