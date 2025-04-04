import os
import configargparse
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd

# from helpers import io
# from helpers import filters
# from helpers import constants
from helpers import io, filters, constants
from collection_mapper import COLLECTION_FN_MAPPER
import data_provenance_card as data_provenance_card
from downloader import Downloader
import data_bibtex as data_bibtex
from dotenv import load_dotenv
from huggingface_hub import HfApi, utils
from datasets.exceptions import DatasetNotFoundError

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

try:
    api = HfApi(token=hf_token)
    print("Hugging Face Authorization Successful!")
except Exception as e:
    print(e)
    print("HF Authorization Failed, some datasets might not download properly!!!")


def check_args(args):
    def validate_date_format(date_str):
        """Validate if date_str is in yyyy-mm-dd format and is a valid date."""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    # Validate the provided start and end times
    if args.start_time and not validate_date_format(args.start_time):
        raise ValueError(f"Invalid start-time format: {args.start_time}")

    if args.end_time and not validate_date_format(args.end_time):
        raise ValueError(f"Invalid end-time format: {args.end_time}")

    if args.start_time and args.end_time:
        start_dt = datetime.strptime(args.start_time, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_time, '%Y-%m-%d')
        if start_dt >= end_dt:
            raise ValueError("Start time should be before end time")


def get_collection_to_uid_and_filter_ids(data_summary):
    filter_df_info = data_summary[[
        "Collection", "Unique Dataset Identifier", "Dataset Filter IDs"]]
    def extract_uids_and_filter_ids(group):
        return [(row["Unique Dataset Identifier"], row["Dataset Filter IDs"]) for _, row in group.iterrows()]
    collection_to_keys = dict(filter_df_info.groupby("Collection").apply(extract_uids_and_filter_ids))
    return {k: dict(v) for k, v in collection_to_keys.items()}


def main(args):
    data_summary = io.read_data_summary_json("data_summaries/")

    data_summary = filters.map_license_criteria(
        data_summary, ALL_CONSTANTS)

    data_summary_df = pd.DataFrame(data_summary).fillna("")
    filtered_data_summary = filters.apply_filters(
        data_summary_df,
        ALL_CONSTANTS,
        args.collection,
        io.read_txt(args.dataset_names) if args.dataset_names else None,
        args.licenses,
        args.license_use,
        int(args.openai_license_override),
        args.license_attribution,
        args.license_sharealike,
        args.languages,
        args.tasks,
        args.domains,
        False if int(args.model_generated) else True,
        args.text_sources,
        args.start_time,
        args.end_time,
        args.license_sources,
        int(args.dpi_undefined_license_override)
    )
    n_collections = set(filtered_data_summary["Collection"])
    n_datasets = len(filtered_data_summary)
    print(f"{n_datasets} datasets from {len(n_collections)} collections after filtering.")

    # IGNORE:
    # cols = ['Unique Dataset Identifier', 'Collection', 'Dataset Name', 'Languages', 'Text Sources','Model Generated', 
    #         'Derived from Datasets', 'License Use (DataProvenance)', 'License Use (GitHub)', 'Licenses', 'GitHub License',
    #         'Dataset URL', 'GitHub URL', 'ArXiv URL']
    # filtered_data_summary[cols].to_csv("pile_v2.csv", index=False)

    # Function to find domains for given text sources
    # def find_domains(text_sources):
    #     # This will hold all the domains for the sources found
    #     domains_found = set()
    #     # Iterate over each source in the text sources list
    #     for source in text_sources:
    #         # Check each domain in DOMAINS
    #         for domain, sources in ALL_CONSTANTS["DOMAIN_GROUPS"].items():
    #             if source in sources:
    #                 domains_found.add(domain)
    #     return list(domains_found)

    # filtered_data_summary['Domains'] = filtered_data_summary['Text Sources'].apply(find_domains)

    # # Define a function to concatenate lists within each group
    # def concatenate_lists(series):
    #     concatenated_list = []
    #     for item in series:
    #         concatenated_list.extend(item)

    #     if len(concatenated_list) and isinstance(concatenated_list[0], dict):
    #         return list(set([x["License"] for x in concatenated_list]))
    #     else:
    #         return list(set(concatenated_list))

    # def aggregate_strings(series):
    #     return list(set(series))  # Simply convert the series of strings to a list

    # # Group by 'Collection' and concatenate lists
    # grouped = filtered_data_summary.groupby('Collection').agg({
    #     # 'Collection': 'first',  # Assuming you want the first occurrence
    #     # 'Dataset Name': 'first',
    #     'Languages': concatenate_lists,
    #     'Text Sources': concatenate_lists,
    #     'Domains': concatenate_lists,
    #     'Model Generated': concatenate_lists,
    #     'License Use (DataProvenance)': aggregate_strings,
    #     'License Use (DataProvenance IgnoreOpenAI)': aggregate_strings,
    #     'Licenses': concatenate_lists,
    # }).reset_index()

    # # Save to CSV
    # cols = ['Collection', 'Languages', 'Text Sources', 'Domains', 'Model Generated', 'License Use (DataProvenance)',
    #     'License Use (DataProvenance IgnoreOpenAI)', 'Licenses']
    # grouped[cols].to_csv("dpi_collections_grouped.csv", index=False)

    # assert 0 == 1

    data_provenance_card.generate_datacard(
        filtered_data_summary,
        args.licenses,
        args.languages,
        args.tasks,
        args.savedir,
    )
    data_bibtex.generate_bibtex(filtered_data_summary, save_to_file=True, output_dir=args.savedir)

    collection_to_keys = get_collection_to_uid_and_filter_ids(filtered_data_summary)
    for collection_key, uid_task_keys in collection_to_keys.items():
        if args.collection and collection_key != args.collection:
            continue
        flat_task_keys = [tk for tks in uid_task_keys.values() for tk in tks]
        print(f"\n=====================\n{collection_key}")
        print(f"Found {len(flat_task_keys)} tasks...")
        downloader_args = COLLECTION_FN_MAPPER[collection_key]
        # dataset unique identifier --> dataset_filter_ids
        downloader_args.update({"uid_key_mapper": uid_task_keys})
        collection_filter_maps = {}
        downloader = Downloader(name=collection_key, **downloader_args)
        try:
            downloader.run_and_save(
                flat_task_keys,
                limit=args.data_limit,
                reformat=args.output_format,
                savedir=args.savedir,
                debug=args.debug
            )
        except utils.GatedRepoError:
            print(f"You are not authorized to download Collection: {collection_key}. Please go to their HF Page and accept the dataset terms and conditions.")
        except DatasetNotFoundError as e:
            print(f"Dataset {collection_key} can not be found, Exception Message: {e}")


if __name__ == "__main__":
    """
    Example commands:

    python src/download_and_filter.py -c src/configs/default.yaml

    OR

    python src/download_and_filter.py --collection "<Name of collection in data_summary>"

    Process:
        1. Load summaries of all the datasets (a mapping from their ID to metadata)
        2. Filter the dataset summaries by their properties, according to your passed in filters
        3. Iteratively load each data collection remaining, normalize and save them.
    """
    ALL_CONSTANTS = io.read_all_constants("constants/")

    parser = configargparse.ArgumentParser(
        description='download_and_filter.py',
        default_config_files=["src/configs/default.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    parser.add("-c", "--config", required=False, is_config_file=True, help="Path to config file.")
    # Specify one collection to pull, rather than all.
    parser.add(
        "--collection", required=False,
        default=None, choices=list(COLLECTION_FN_MAPPER.keys()),
        help="Name of the collection you'd like to download and filter. Default is to loop through all collections.")
    # Dataset UIDs to limit to
    parser.add(
        "--dataset-names", required=False, default=None,
        help=f"A path to a text file with newline separated dataset UIDs to confine our datasets to.")
    # Specify license categories
    parser.add(
        "-l", "--licenses", required=False,
        nargs='*', default=[],
        # choices=['academic', 'non-commercial', 'commercial'],
        help=f"A list of licenses you would confine our datasets to.")
    parser.add(
        "-lu", "--license_use", required=False,
        default="academic-only",
        choices=['commercial', 'unspecified', 'non-commercial', 'academic-only'],
        help=f"Which category of permitted use to allow, based on dataset licenses.")
    parser.add(
        "-ls", "--license_sources", required=False,
        nargs='*',
        default=["DataProvenance"],
        choices=['DataProvenance', 'HuggingFace', 'GitHub'],
        help="Source from where the license information should be retrieved"
    )
    parser.add(
        "-la", "--license_attribution", required=False,
        default='1',
        choices=['0', '1'],
        help=f"Whether to use all licenses, including those that require attribution (1) or only the ones that don't require attribution (0).")
    parser.add(
        "-lsa", "--license_sharealike", required=False,
        default='1',
        choices=['0', '1'],
        help=f"Whether to use all licenses, including those that require share alike (1) or only the ones that don't require share alike (0).")
    # Override license for Model Generated datasets
    parser.add(
        "-ol", "--openai-license-override", required=False,
        default=0, choices=['0', '1'],
        help=f"Whether to include datasets that include generations from OpenAI models, overriding any license filter.")
    # Override DPI license if "unspecified" and a GitHub license is available
    parser.add(
        "-od", "--dpi-undefined-license-override", required=False,
        default=0, choices=['0', '1'],
        help="Whether to use GitHub license information if not available ('unspecified') for our Data Provenance source.")
    # Specify language categories
    parser.add(
        "-g", "--languages", required=False,
        nargs='*', default=[],
        choices=list(ALL_CONSTANTS["LANGUAGE_GROUPS"].keys()),
        help=f"A list of language categories we would confine our datasets to.")
    # Specify task categories
    parser.add(
        "-t", "--tasks", required=False,
        nargs='*', default=[],
        choices=list(ALL_CONSTANTS["TASK_GROUPS"].keys()),
        help=f"A list of tasks categories we would confine our datasets to.")
    # Specify source domains
    parser.add(
        "-dd", "--domains", required=False,
        nargs='*', default=[],
        choices=list(ALL_CONSTANTS["DOMAIN_GROUPS"].keys()),
        help=f"A list of source domains we would confine our datasets to.")
    # Whether to exclude any synthetic (model-generated) data
    parser.add(
        "-mg", "--model-generated", required=False,
        default=1, choices=['0', '1'],
        help="Whether to include/exclude synthetic (model-generated) data. '1' is include, '0' is explicitly exclude.")
    # Whether to only allow datasets from certain text sources (including no text source). null or txt file path.
    parser.add(
        "-tts", "--text-sources", required=False,
        default="",
        help="Points to a txt file path to an allow list of text sources (including no text source), or null.")
    # Start time boundary
    parser.add(
        "-ts", "--start-time", required=False,
        default=None, type=str,
        # choices=[str(x) for x in range(1900, 2023)],
        help=f"The start date formatted as `YYYY-MM-DD, inclusive.")
    # End time boundary
    parser.add(
        "-te", "--end-time", required=False,
        default=None, type=str,
        help=f"The end date formatted as `YYYY-MM-DD, exclusive. Must be after the start time.")
    # Specify data limit
    parser.add(
        "-dl", "--data-limit", required=False,
        default=0, type=int,
        help=f"How many rows to randomly sample from each collection.")
    # Specify Data output format type
    parser.add(
        "-of", "--output-format", required=False,
        default="messages", type=str,
        choices=["messages", "supervised", "concat_dialog"],
        help="The output format to save the data. By default it mimcs the format described in `preparers.py`. `supervised` means it saves as input-target pairs.")
    # Specify savedir
    parser.add(
        "-s", "--savedir", required=False,
        default="data", type=str,
        help=f"The directory to save your downloaded data to.")
    # Specify debug
    parser.add(
        "-d", "--debug", default=False, dest='debug', action='store_true',
        help=f"Debug mode does not run preparer in parallel. Defaults to False.")
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    if args.debug:
        print("Debug Mode activated.")
    print(parser.format_values())
    check_args(args)
    main(args)