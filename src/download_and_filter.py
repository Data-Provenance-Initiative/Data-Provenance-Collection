import os
import configargparse
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd

from helpers import io
from collection_mapper import COLLECTION_FN_MAPPER
import constants
import data_provenance_card as data_provenance_card
from downloader import Downloader
import data_bibtex as data_bibtex


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

def classify_license(license_name, license_url, all_constants):
    if license_name == "Custom":
        use_case, attribution, share_alike = all_constants["CUSTOM_LICENSE_CLASSES"].get(license_url, ("?", "?", "?"))
    else:
        use_case, attribution, share_alike = all_constants["LICENSE_CLASSES"][license_name]
    return {
        "use": use_case, 
        "attribution": int(attribution) if attribution.isnumeric() else 1, 
        "share_alike": int(share_alike) if share_alike.isnumeric() else 1,
    }

def resolve_multiple_licenses(license_criterias):
    if not license_criterias:
        # Return empty if no licenses from this aggregator
        return ["", "", ""]
    use_cases = [l["use"] for l in license_criterias]
    attributions = [l["attribution"] for l in license_criterias]
    share_alikes = [l["share_alike"] for l in license_criterias]

    if "?" in use_cases:
        resolved_use_case = "academic-only"
    elif "Acad" in use_cases:
        resolved_use_case = "academic-only"
    elif "NC" in use_cases:
        resolved_use_case = "non-commercial"
    elif "Unspecified" in use_cases:
        resolved_use_case = "unspecified"
    elif "All":
        resolved_use_case = "commercial"
    
    resolved_attribution = max(attributions)
    resolved_share_alikes = max(share_alikes)
    return resolved_use_case, resolved_attribution, resolved_share_alikes


def map_license_criteria(data_summary, all_constants):

    # Unpack licenses for each dataset. {uid --> (license_name, license_url)}
    our_uid_to_license_infos = defaultdict(list)
    hf_uid_to_license_infos = defaultdict(list)
    github_uid_to_license_infos = defaultdict(list)
    pwc_uid_to_license_infos = defaultdict(list)
    # Same as ours, but excludes OpenAI Terms:
    our_uid_to_license_infos_no_openai = defaultdict(list)

    for row in data_summary:
        uid = row["Unique Dataset Identifier"]
        for license_info in row["Licenses"]:
            license_name = license_info["License"]
            license_url = license_info["License URL"]
            our_uid_to_license_infos[uid].append((license_name, license_url))
            if license_info["License"] != "OpenAI":
                our_uid_to_license_infos_no_openai[uid].append((license_name, license_url))
        # If OpenAI was the only license, we add Unspecified so there isn't nothing there.
        if len(our_uid_to_license_infos_no_openai[uid]) == 0:
            our_uid_to_license_infos_no_openai[uid].append(("Unspecified", None))

        gh_license = row.get("Inferred Metadata", {}).get("GitHub License", None)
        hfy_license = row.get("Inferred Metadata", {}).get("HF Yaml License", None)
        hfc_license = row.get("Inferred Metadata", {}).get("HF Config License", None)
        pwc_license = row.get("Inferred Metadata", {}).get("PwC License Name", None)
        if hfy_license:
            hf_uid_to_license_infos[uid].append((hfy_license, None))
        if hfc_license:
            hf_uid_to_license_infos[uid].append((hfc_license, None))
        if gh_license:
            github_uid_to_license_infos[uid].append((gh_license, None))
        if pwc_license:
            pwc_uid_to_license_infos[uid].append((pwc_license, None))

    # valid_licenses = list(all_constants["LICENSE_CLASSES"].keys())
    # print(set([v for vs in pwc_uid_to_license_infos.values() for (v, _) in vs]) - set(valid_licenses))
    # print(set([v for vs in github_uid_to_license_infos.values() for (v, _) in vs]) - set(valid_licenses))

    def classify_and_resolve_licenses(license_infos, all_constants):
        classified_licenses = []
        for (license_name, license_url) in license_infos:
            classifications = classify_license(license_name, license_url, all_constants)
            classified_licenses.append(classifications)
        resolved_criteria = resolve_multiple_licenses(classified_licenses)
        return resolved_criteria

    # classify and resolve licenses for each dataset and each aggregator
    ours_resolved, ours_openai_resolved, hf_resolved, gh_resolved, pwc_resolved = {}, {}, {}, {}, {}
    for uid in our_uid_to_license_infos.keys():
        ours_resolved[uid] = classify_and_resolve_licenses(our_uid_to_license_infos[uid], all_constants)
        ours_openai_resolved[uid] = classify_and_resolve_licenses(our_uid_to_license_infos_no_openai[uid], all_constants)
        hf_resolved[uid] = classify_and_resolve_licenses(hf_uid_to_license_infos[uid], all_constants)
        gh_resolved[uid] = classify_and_resolve_licenses(github_uid_to_license_infos[uid], all_constants)
        pwc_resolved[uid] = classify_and_resolve_licenses(pwc_uid_to_license_infos[uid], all_constants)

    def add_license_classes_to_summaries(data_summary, resolved_classes, aggregator):
        # update dataframe with columns for use, attribution, share_alike
        for row in data_summary:
            row[f'License Use ({aggregator})'] = resolved_classes[row['Unique Dataset Identifier']][0]
            row[f'License Attribution ({aggregator})'] = resolved_classes[row['Unique Dataset Identifier']][1]
            row[f'License Share Alike ({aggregator})'] = resolved_classes[row['Unique Dataset Identifier']][2]
        return data_summary

    data_summary = add_license_classes_to_summaries(data_summary, ours_resolved, "DataProvenance")
    data_summary = add_license_classes_to_summaries(data_summary, ours_openai_resolved, "DataProvenance IgnoreOpenAI")
    data_summary = add_license_classes_to_summaries(data_summary, hf_resolved, "HuggingFace")
    data_summary = add_license_classes_to_summaries(data_summary, gh_resolved, "GitHub")
    data_summary = add_license_classes_to_summaries(data_summary, pwc_resolved, "PapersWithCode")

    return data_summary


def apply_filters(
    df,
    all_constants,
    selected_collection,
    selected_licenses,
    selected_license_use,
    openai_license_override,
    selected_license_attribution,
    selected_license_sharealike,
    selected_languages,
    selected_task_categories,
    selected_domains,
    selected_start_time,
    selected_end_time,
):
    filtered_df = df

    # Some sanity checks:
    all_langs = set([v for vs in all_constants["LANGUAGE_GROUPS"].values() for v in vs])
    option_langs = set(
        [lang for langs in filtered_df["Languages"].tolist() for lang in langs]
    )
    assert all_langs >= option_langs, f"Missing Languages: {option_langs - all_langs}"

    all_tcats = set([v for vs in all_constants["TASK_GROUPS"].values() for v in vs])
    option_tcats = set(
        [tc for tcs in filtered_df["Task Categories"].tolist() for tc in tcs]
    )
    assert (
            all_tcats >= option_tcats
    ), f"Missing Task Categories: {option_tcats - all_tcats}"

    all_sources = set([v for vs in all_constants["DOMAIN_GROUPS"].values() for v in vs])
    option_sources = set(
        [src for sources in filtered_df["Text Sources"].tolist() for src in sources]
    )
    assert all_sources >= option_sources, f"Missing Text Sources: {option_sources - all_sources}"

    all_models = set([v.lower() for vs in all_constants["MODEL_GROUPS"].values() for v in vs])
    option_models = set(
        [model.lower() for models in filtered_df["Model Generated"].tolist() for model in models]
    )
    assert all_models >= option_models, f"Missing Models: {option_models - all_models}"


    # Apply filters:
    if selected_collection:
        filtered_df = filtered_df[filtered_df["Collection"] == selected_collection]

    if not filtered_df.empty and selected_licenses:
        license_strs = set(all_constants["LICENSE_CLASSES"].keys())
        filtered_df = filtered_df[
            filtered_df["Licenses"].apply(lambda xs: license_strs >= set([x["License"] for x in xs]))
        ]

    if not filtered_df.empty and selected_license_use:
        use_key = "License Use (DataProvenance IgnoreOpenAI)" if openai_license_override else "License Use (DataProvenance)"
        valid_license_use_idx = constants.LICENSE_USE_TYPES.index(selected_license_use)
        valid_license_uses = constants.LICENSE_USE_TYPES[:valid_license_use_idx+1]
        filtered_df = filtered_df[
            filtered_df[use_key].apply(lambda x: x in valid_license_uses)
        ]

    if not filtered_df.empty and selected_license_attribution:
        filtered_df = filtered_df[
            filtered_df["License Attribution (DataProvenance)"].apply(lambda x: x <= int(selected_license_attribution))
        ]

    if not filtered_df.empty and selected_license_sharealike:
        filtered_df = filtered_df[
            filtered_df["License Share Alike (DataProvenance)"].apply(lambda x: x <= int(selected_license_sharealike))
        ]

    if not filtered_df.empty and selected_languages:
        lang_strs = set(
            [
                lang_str
                for k in selected_languages
                for lang_str in all_constants["LANGUAGE_GROUPS"][k]
            ]
        )
        filtered_df = filtered_df[
            filtered_df["Languages"].apply(lambda x: len(lang_strs.intersection(set(x))) > 0)
        ]

    if not filtered_df.empty and selected_task_categories:
        taskcat_strs = set(
            [
                taskcat_str
                for k in selected_task_categories
                for taskcat_str in all_constants["TASK_GROUPS"][k]
            ]
        )
        filtered_df = filtered_df[
            filtered_df["Task Categories"].apply(lambda x: len(taskcat_strs.intersection(set(x))) > 0)
        ]
    if not filtered_df.empty and selected_domains:
        text_source_strs = set(
            [
                source_str
                for k in selected_domains
                for source_str in all_constants["DOMAIN_GROUPS"][k]
            ]
        )
        filtered_df = filtered_df[
            filtered_df["Text Sources"].apply(lambda x: len(text_source_strs.intersection(set(x))) > 0)
        ]
    if not filtered_df.empty and (selected_start_time or selected_end_time):

        def get_min_date(metadata):
            date_columns = ["S2 Date", "HF Date", "GitHub Date"]
            dates = [metadata.get(col, "") for col in date_columns]
            valid_dates = [pd.to_datetime(date, format='%Y-%m-%d', errors='coerce') for date in dates if date]
            if valid_dates:
                return min(valid_dates)
            return pd.NaT

        filtered_df['Estimated Creation Date'] = filtered_df['Inferred Metadata'].apply(get_min_date)
        if selected_start_time:
            filtered_df = filtered_df[filtered_df['Estimated Creation Date'] >= pd.to_datetime(selected_start_time)]
        if selected_end_time:
            filtered_df = filtered_df[filtered_df['Estimated Creation Date'] <= pd.to_datetime(selected_end_time)]

    return filtered_df

def get_collection_to_uid_and_filter_ids(data_summary):
    filter_df_info = data_summary[[
        "Collection", "Unique Dataset Identifier", "Dataset Filter IDs"]]
    def extract_uids_and_filter_ids(group):
        return [(row["Unique Dataset Identifier"], row["Dataset Filter IDs"]) for _, row in group.iterrows()]
    collection_to_keys = dict(filter_df_info.groupby("Collection").apply(extract_uids_and_filter_ids))
    return {k: dict(v) for k, v in collection_to_keys.items()}

if __name__ == "__main__":
    """
    Example commands:

    python src/download_and_filter.py -c src/configs/default.yaml

    Process:
        1. Load summaries of all the datasets (a mapping from their ID to metadata)
        2. Filter the dataset summaries by their properties, according to your passed in filters
        3. Iteratively load each data collection remaining, normalize and save them.
    """
    ALL_CONSTANTS = io.read_all_constants()

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
        "-sd", "--domains", required=False,
        nargs='*', default=[],
        choices=list(ALL_CONSTANTS["DOMAIN_GROUPS"].keys()),
        help=f"A list of source domains we would confine our datasets to.")
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
        choices=["messages", "supervised"],
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

    data_summary = io.read_data_summary_json("data_summaries/")

    data_summary = map_license_criteria(
        data_summary, ALL_CONSTANTS)

    data_summary_df = pd.DataFrame(data_summary).fillna("")
    filtered_data_summary = apply_filters(
        data_summary_df,
        ALL_CONSTANTS,
        args.collection,
        args.licenses,
        args.license_use,
        int(args.openai_license_override),
        args.license_attribution,
        args.license_sharealike,
        args.languages,
        args.tasks,
        args.domains,
        args.start_time,
        args.end_time,
    )
    n_collections = set(filtered_data_summary["Collection"])
    n_datasets = len(filtered_data_summary)
    print(f"{n_datasets} datasets from {n_collections} after filtering.")

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
        downloader.run_and_save(
            flat_task_keys,
            limit=args.data_limit,
            reformat=args.output_format,
            savedir=args.savedir,
            debug=args.debug
        )
