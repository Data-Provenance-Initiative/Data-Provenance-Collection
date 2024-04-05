# import json
import os
# import urllib.parse
from collections import Counter
import argparse
import pandas as pd
from helpers import io
import constants
# from constants import read_all_constants
from collection_mapper import COLLECTION_FN_MAPPER
from downloader import Downloader
# from downloaders import pool_filter
from download_and_filter import get_collection_to_uid_and_filter_ids


ATTRIBUTE_MAPPER = {
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict
}


class ErrorHandler:
    def __init__(self, no_halt_on_error, collection):
        self.no_halt_on_error = no_halt_on_error
        self.collection = collection
        self.errors = []

    def handle(self, error_message):
        if self.no_halt_on_error:
            self.errors.append(error_message)
        else:
            raise AssertionError(error_message)

    def print_errors(self):
        if self.errors:
            for error in self.errors:
                print(f"Error: {error}")
        else:
            print(f"Passed all tests for {self.collection}!")


def test_collection_summary(
    collection_name,
    collection_summary,
    error_handler,
):
    """Tests the collection's data summary entries are valid."""
    CONSTANTS = io.read_all_constants()

    # All acceptable licenses
    all_licenses = set(list(CONSTANTS["LICENSE_CLASSES"].keys()) + ["Custom"])
    # All acceptable languages
    all_langs = set([v for vs in CONSTANTS["LANGUAGE_GROUPS"].values() for v in vs])
    # All acceptable task categories
    all_tasks = set([v for vs in CONSTANTS["TASK_GROUPS"].values() for v in vs])
    # All acceptable text generation models
    all_models = set([v for vs in CONSTANTS["MODEL_GROUPS"].values() for v in vs])
    # All acceptable creators groups
    all_creators = set([v for vs in CONSTANTS["CREATOR_GROUPS"].values() for v in vs])
    # All acceptable formats
    all_formats = CONSTANTS["FORMATS"]

    # The collection must have an abbreviation that starts each dataset's Unique
    # Dataset Identifier (UDI). E.g., for Flan Collection, we use "fc" and a dataset
    # would have identifier such as "fc-p3-adversarial_qa"
    collection_abbr = next(iter(collection_summary)).split("-")[0]

    for dset_uid, dset_info in collection_summary.items():
        # Check that each dataset UID starts with the abbreviation
        if not dset_uid.startswith(f"{collection_abbr}"):
            error_handler.handle(f"Dataset {dset_uid} does not start with the collection abbreviation.")

        if dset_info["Unique Dataset Identifier"] != dset_uid:
            error_handler.handle(f"The `Unique Dataset Identifier` attribute and key in json for {dset_uid} must match.")

        if dset_info["Collection"] != collection_name:
            error_handler.handle(f"For {dset_uid}, the file name should match the `Collection` field in the json.")

        # Check that each dataset has appropriate languages, licenses, and tasks
        languages = set(dset_info["Languages"])
        licenses = set([d["License"] for d in dset_info["Licenses"]])
        tasks = set(dset_info["Task Categories"])
        models = set(dset_info["Model Generated"])
        creators = set(dset_info["Creators"])
        formats = set(dset_info["Format"])

        if not languages.issubset(all_langs):
            error_handler.handle(f"For {dset_uid}, languages {languages - all_langs} are not in list of acceptable languages in {constants.LANGUAGE_CONSTANTS_FP}.")
        if not licenses.issubset(all_licenses):
            error_handler.handle(f"For {dset_uid}, licenses {licenses - all_licenses} are not in list of acceptable licenses in {constants.LICENSE_CONSTANTS_FP}.")
        if not tasks.issubset(all_tasks):
            error_handler.handle(f"For {dset_uid}, task categories {tasks - all_tasks} are not in list of acceptable tasks in {constants.TASK_CONSTANTS_FP}")
        if not models.issubset(all_models):
            error_handler.handle(f"For {dset_uid}, generated models {models - all_models} are not in list of acceptable tasks in {constants.MODEL_CONSTANTS_FP}")
        if not creators.issubset(all_creators):
            error_handler.handle(f"For {dset_uid}, creators {creators - all_creators} are not in list of acceptable tasks in {constants.CREATOR_CONSTANTS_FP}")
        if not formats.issubset(all_formats):
            error_handler.handle(f"For {dset_uid}, formats {formats - all_formats} are not in list of acceptable formats {all_formats}.")

        if dset_info['ArXiv URL']:
            if "arxiv.org/abs/" not in dset_info['ArXiv URL'] and "aclanthology.org" not in dset_info['ArXiv URL']:
                error_handler.handle(f"For {dset_uid} the `ArXiv URL` should be to the paper abstract page `arxiv.org/abs/`.")

        if dset_info['Papers with Code URL'] and "paperswithcode.com/dataset/" not in dset_info['Papers with Code URL']:
            error_handler.handle(f"For {dset_uid} the `Papers with Code URL` should be to a 'dataset' entry: `paperswithcode.com/dataset/`")


def test_json_key_order(template, test):
    template_keys = list(template.keys())
    test_keys = list(test.keys())

    # Iterate over template keys and check their order in the test dictionary
    test_index = 0
    for template_key in template_keys:
        if template_key in test_keys[test_index:]:
            test_index = test_keys.index(template_key, test_index) + 1
        else:
            continue  # Skip if the template_key is not found in test
    # Check for nested dictionaries
    for key in template_keys:
        if isinstance(template[key], dict):
            if key in test and isinstance(test[key], dict):
                if not test_json_key_order(template[key], test[key]):
                    return False
    return True


def test_json_correctness(
    all_collection_infos,
    collection_info,
    template_spec,
    error_handler,
):
    # Test each `Unique Dataset Identifier` is actually unique.
    all_uids = Counter(all_collection_infos.keys())
    for uid, count in all_uids.most_common():
        if count > 1:
            error_handler.handle(f"Unique Dataset Identifier {uid} appears {count} times. Should be unique.")

    # For each dataset
    template_keys = set(template_spec.keys())
    for dataset_uid, dataset_info in collection_info.items():

        # Test for missing attributes
        required_template_keys = {k for k in template_keys if len(template_spec[k]) == 3 and template_spec[k][1]}
        missing_keys = required_template_keys - set(dataset_info.keys())
        if missing_keys:
            error_handler.handle(f"{dataset_uid} is missing required attributes: {missing_keys}")

        # Test for attributes with the wrong type
        # template_types = {k: template_spec[k][0] for k in template_keys if len(template_spec[k]) == 3} # :TODO: currently unused needs to be fixed
        for k, spec in template_spec.items():
            if len(spec) == 3 and k in dataset_info:
                correct_typ = ATTRIBUTE_MAPPER[spec[0]]
                if correct_typ == int:
                    typ_pass = isinstance(dataset_info[k], int) or dataset_info[k].isnumeric() or dataset_info[k] == ""
                elif correct_typ == float:
                    typ_pass = isinstance(dataset_info[k], float) or dataset_info[k].replace(".", "").isnumeric() or dataset_info[k] == ""
                else:
                    typ_pass = isinstance(dataset_info[k], correct_typ)

                if not typ_pass:
                    typ_descrip = f"{correct_typ} (or empty string)" if correct_typ in [int, float] else correct_typ
                    error_handler.handle(f"{dataset_uid} has attribute with wrong type: {k} is {type(dataset_info[k])} but should be `{typ_descrip}`.")

        # Test for additional attributes
        template_keys = set(template_spec.keys())
        additional_keys = set(dataset_info.keys()) - template_keys
        if additional_keys:
            error_handler.handle(f"{dataset_uid} contains additional attributes: {additional_keys}")

        # Test for additional attributes inside dictionary nests
        nested_attribute_keys = {k: list(vs.keys()) for k, vs in template_spec.items() if isinstance(vs, dict)}
        for k, nested_keys in nested_attribute_keys.items():
            if k in dataset_info and isinstance(dataset_info[k], dict):
                additional_nested_keys = set(dataset_info[k].keys()) - set(template_spec[k].keys())
                if additional_nested_keys:
                    error_handler.handle(f"{dataset_uid} contains additional nested attributes: {additional_nested_keys}")

        # Test the correct order.
        if not test_json_key_order(template_spec, dataset_info):
            error_handler.handle(f"{dataset_uid} does not have attributes in the same order as the template spec.")

        # Test present attributes have values when required
        template_req_vals = {k: template_spec[k][2] for k in template_keys if len(template_spec[k]) == 3}
        for k, spec in template_req_vals.items():
            if template_req_vals[k] and k in dataset_info and not dataset_info[k]:
                error_handler.handle(f"{k} should be populated with a value if present in {dataset_uid}.")


def test_outputs(collection, collection_info, error_handler):
    for uid, dset_info in collection_info.items():
        dset = [row for row in collection if row[0]["parent"] == uid]

        # Check if the output has correct format
        if not all([key in ex for dialog in dset for ex in dialog for key in ['from', 'text', 'parent']]):
            error_handler.handle(f"{uid} output does not have the correct keys: {['from', 'text', 'parent']}")

        for dialog in dset:
            message_parents = []
            for midx, message in enumerate(dialog):
                if midx == 0:
                    if not isinstance(message['parent'], str):
                        error_handler.handle(f"{uid} first message's `parent` field should be of type str, indicating the Unique Dataset Identifier, not {type(message['parent'])}")
                        return
                    else:
                        if not isinstance(message['parent'], (str, int)):
                            error_handler.handle(f"{uid} message's `parent` field should be of type str or int, not {type(message['parent'])}")
                            return
                if isinstance(message['parent'], int) and (message['parent'] < 0 or message['parent'] > midx):
                    error_handler.handle(f"{uid} dialog chains, from the `parent` field are poorly formed.")
                    return
                if message['from'] not in ['user', 'assistant']:
                    error_handler.handle(f"{uid} message's `from` field is {message['from']} but should be 'user' or 'assistant'")
                    return
                if not isinstance(message['text'], str):
                    error_handler.handle(f"{uid} message's `text` field is {type(message['text'])} but should be a string")
                    return
                if "Response Ranking" in dset_info["Format"] and message["from"] == "assistant":
                    if "score" not in message:
                        error_handler.handle(f"{uid} message's from 'assistants' should have a 'score' field.")
                        return
                    if not isinstance(message.get("score"), (int, float)):
                        error_handler.handle(f"{uid} message's from 'assistants' should have a 'score' field, that is an int or float, not `{message['score']}`.")
                        return
                message_parents.append(message["parent"])
        if not any([parent == 0 for parent in message_parents]):
            error_handler.handle(f"{uid} there does not appear to be a response to the original input in this dialog.")
            return

    #  Check number of examples and text statistics
    # if not config["Text Metrics"]["Num Dialogs"] or len(output) != cconfig["Text Metrics"]["Num Dialogs"]:
    #     error_handler.handle("Error: Number of examples does not match the number specified in the json config")

    # TODO: check text statistics (text length metrics)

        if dset_info['Format'] == 'Multi-turn Dialog':
            if not any(len(dialog) > 2 for dialog in dset):
                error_handler.handle(f"For {uid} no dialog has more than two turns, but it's 'Format' is specified as `Multi-turn Dialog`")


def test_dataset_reference(collection, collection_info, error_handler):
    for uid, dset_info in collection_info.items():
        dset = [row for row in collection if row[0]["parent"] == uid]
        for dialog in dset:
            message = dialog[0]
            if message['parent'] != dset_info['Unique Dataset Identifier']:
                error_handler.handle(f"For {uid}, the `parent` field of the first message in every dialog must be the same as the `Unique Dataset Identifier` so we can identify their origin.")
                break


def test_filter_id(collection, uid_task_keys, error_handler):
    dset_slice_counts = dict(Counter([row[0]["parent"] for row in collection]))
    for uid, filter_ids in uid_task_keys.items():
        if uid not in dset_slice_counts:
            error_handler.handle(f"{uid} `Dataset Filter Ids` {filter_ids} do not correspond to any examples. They may be misspelt.")


def test_bibtex_semanticscholar(collection_info, error_handler):
    for uid, dset_info in collection_info.items():
        if dset_info["Semantic Scholar Corpus ID"] and dset_info["ArXiv URL"]:
            CorpusId = dset_info["Semantic Scholar Corpus ID"]
            result = io.get_bibtex_from_paper("CorpusId:{}".format(CorpusId))
            if not result:
                error_handler.handle(f"{uid} Semantic Scholar Corpus ID {CorpusId} is not valid.")


def test_downloader_and_preparer(
    data_summary,
    collection_info,
    error_handler
):
    collection_name = collection_info[list(collection_info.keys())[0]]["Collection"]
    uid_to_filter_ids = {uid: dset_info['Dataset Filter IDs'] for uid, dset_info in collection_info.items()}

    # Load configurations and run the downloader/preparer
    data_summary_df = pd.DataFrame(list(data_summary.values())).fillna("")
    uid_task_keys = get_collection_to_uid_and_filter_ids(data_summary_df)[collection_name]

    downloader_args = COLLECTION_FN_MAPPER[collection_name]
    downloader_args.update({"uid_key_mapper": uid_task_keys})
    downloader = Downloader(
        name=collection_name,
        **downloader_args
        )
    all_acceptable_filter_ids = [v for vs in uid_to_filter_ids.values() for v in vs]
    full_dset = downloader.download_and_prepare(all_acceptable_filter_ids, debug=True)

    # Test the datset output format and characteristics are correct
    test_outputs(full_dset, collection_info, error_handler)

    # Test each dataset's Filter Ids is doing something.
    test_filter_id(full_dset, uid_task_keys, error_handler)

    # Test the prepared examples map back to the UIDs
    test_dataset_reference(full_dset, collection_info, error_handler)


def run_tests(
    collection_name,
    error_handler
):
    # Check necessary files are available
    DATA_DIR = 'data_summaries'
    assert os.path.exists(DATA_DIR), f"Error: {DATA_DIR} does not exist"
    all_collection_infos = {r["Unique Dataset Identifier"]: r for r in io.read_data_summary_json(DATA_DIR)}
    template_spec_filepath = os.path.join(DATA_DIR, '_template_spec.yaml')
    assert os.path.exists(template_spec_filepath), f"Error: `{template_spec_filepath}` file is missing or corrupted"
    template_spec = io.read_yaml(template_spec_filepath)

    # Open data collection metadata file
    collection_filepath = os.path.join(DATA_DIR, f"{collection_name}.json")
    assert os.path.exists(collection_filepath), f"There is no collection file at {collection_filepath}"
    collection_info = io.read_json(collection_filepath)

    # Test basic properties of the json file.
    # print(f"Testing the json file has the correct entry types and order in {collection_name}")
    test_json_correctness(
        all_collection_infos,
        collection_info,
        template_spec,
        error_handler
    )

    # Test the collection json file entries are valid
    test_collection_summary(
        collection_name,
        collection_info,
        error_handler
    )

    # Test the bibtex and semantic scholar ids are valid
    test_bibtex_semanticscholar(
        collection_info,
        error_handler
    )

    # Test the downloader and preparer works properly
    test_downloader_and_preparer(
        all_collection_infos,
        collection_info,
        error_handler
    )

    if error_handler.no_halt_on_error:
        error_handler.print_errors()


if __name__ == "__main__":
    """
    Example run:

    python src/test_new_collection.py --collection "Alpaca"

    Add `--no_halt` if you'd like it to collect up and print all errors at the end.
    """
    parser = argparse.ArgumentParser(description='Test a new collection.')
    parser.add_argument(
        '--collection',
        type=str,
        required=False,
        default=None,
        choices=list(COLLECTION_FN_MAPPER.keys()) + [None],
        help='Name of the collection to test')
    parser.add_argument(
        '--no_halt',
        action='store_true',
        help='If true, will no longer assert/halt on the first error. It will collect all errors and print at the end.')

    args = parser.parse_args()
    collections = [args.collection]
    collections = COLLECTION_FN_MAPPER.keys() if args.collection is None else [args.collection]
    for collection in collections:
        error_handler = ErrorHandler(args.no_halt, collection)
        run_tests(collection, error_handler)
    if not args.no_halt:
        print("Passed all tests!")
