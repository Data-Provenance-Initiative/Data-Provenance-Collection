from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd


# import src.helpers.constants as constants
from . import io, constants


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

        row["GitHub License"] = gh_license
        row["HF Yaml License"] = hfy_license
        row["HF Config License"] = hfc_license
        row["PwC License"] = pwc_license

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
    selected_license_use,  # sources from where the license information should be retrieved
    openai_license_override,
    selected_license_attribution,
    selected_license_sharealike,
    selected_languages,
    selected_task_categories,
    selected_domains,
    no_synthetic_data,
    text_source_allow_list,
    selected_start_time,
    selected_end_time,
    selected_license_sources,
    dpi_undefined_license_override  # flag to use GitHub license information if not available ("undefined") for our Data Provenance source
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
    # assert all_sources >= option_sources, f"Missing Text Sources: {option_sources - all_sources}" # :TODO: we need to check this here!

    all_models = set([v.lower() for vs in all_constants["MODEL_GROUPS"].values() for v in vs])
    option_models = set(
        [model.lower() for models in filtered_df["Model Generated"].tolist() for model in models]
    )
    assert all_models >= option_models, f"Missing Models: {option_models - all_models}"

    # Load text sources allow list if available
    if text_source_allow_list:
        text_source_allow_list = io.read_txt(text_source_allow_list)

    # Apply filters:
    if selected_collection:
        filtered_df = filtered_df[filtered_df["Collection"] == selected_collection]

    if not filtered_df.empty and selected_licenses:
        license_strs = set(all_constants["LICENSE_CLASSES"].keys())
        filtered_df = filtered_df[
            filtered_df["Licenses"].apply(lambda xs: license_strs >= set([x["License"] for x in xs]))
        ]

    if not filtered_df.empty and selected_license_use:
        valid_license_use_idx = constants.LICENSE_USE_TYPES.index(selected_license_use)
        valid_license_uses = [x.lower() for x in constants.LICENSE_USE_TYPES[:valid_license_use_idx + 1]]  # ["academic-only", ...]

        # check if openai license override is selected, if so, remove DataProvenance from sources and add DataProvenance IgnoreOpenAI
        if openai_license_override:
            # remove "DataProvenance" from selected_license_sources if openai_license_override is selected and add "DataProvenance IgnoreOpenAI" to selected_license_sources
            if "DataProvenance" in selected_license_sources:
                selected_license_sources.remove("DataProvenance")
                selected_license_sources.append("DataProvenance IgnoreOpenAI")

            # Check that DataProvenance is not in selected_license_sources if openai_license_override is selected
            assert "DataProvenance" not in selected_license_sources, f"DataProvenance should not be in selected_license_sources: {selected_license_sources}"

        # we have a flag to indicate if we want to use the GitHub license information if the DataProvenance or DataProvenance license is unspecified
        if dpi_undefined_license_override:
            # Check if "DataProvenance" is included in the selected license sources
            # If so, apply the GitHub license information (if it is not empty) to the DataProvenance license information
            if "DataProvenance" in selected_license_sources:
                filtered_df["License Use (DataProvenance)"] = filtered_df.apply(
                    lambda row: row["License Use (GitHub)"]  # check the existing license from GitHub
                    if row["License Use (DataProvenance)"] == 'unspecified' and row["License Use (GitHub)"] != ''  # check if the existing DataProvenance license is unspecified and a possible GitHub license is not empty ''
                    else row["License Use (DataProvenance)"],  # if no alternative GitHub license is available or empty keep using the unspecified DataProvenance license
                    axis=1
                )
                # check that all DataProvenance license which are undefined are replaced with the DataProvenance license information
                assert len([(dpi, git) for dpi, git in zip(df["License Use (DataProvenance)"], df["License Use (GitHub)"]) if dpi == "unspecified" and git != ""]) == 0, "Remaining DataProvenance license which are undefined"

            # Check if "DataProvenance IgnoreOpenAI" is included in the selected license sources
            # If so, apply the GitHub license information (if it is not empty) to the DataProvenance IgnoreOpenAI license information
            if "DataProvenance IgnoreOpenAI" in selected_license_sources:
                filtered_df["License Use (DataProvenance IgnoreOpenAI)"] = filtered_df.apply(
                    lambda row: row["License Use (GitHub)"]  # check the existing license from GitHub
                    if row["License Use (DataProvenance IgnoreOpenAI)"] != 'unspecified' and row["License Use (GitHub)"] != ''  # check if the existing DataProvenance IgnoreOpenAI license is unspecified and a possible GitHub license is not empty ''
                    else row["License Use (DataProvenance IgnoreOpenAI)"],  # if no alternative GitHub license is available or empty keep using the unspecified DataProvenance IgnoreOpenAI license
                    axis=1
                )
                # check that all DataProvenance IgnoreOpenAI license which are undefined are replaced with the DataProvenance IgnoreOpenAI license information
                assert len([(dpi, git) for dpi, git in zip(df["License Use (DataProvenance IgnoreOpenAI)"], df["License Use (GitHub)"]) if dpi == "unspecified" and git != ""]) == 0, "Remaining DataProvenance IgnoreOpenAI license which are undefined"

        # for all license sources ["DataProvenance", "DataProvenance IgnoreOpenAI", "HuggingFace", "GitHub"] add the license use types to the filtered_df depending on valid_license_uses ["academic-only", ...]
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: any(  # if any of the License Use of HuggingFace | GitHub  | ...  is in valid_license_uses ["academic-only",...]
                    row[f"License Use ({key})"] in valid_license_uses  # ["academic-only", ...]
                    for key in selected_license_sources  # ["DataProvenance", "DataProvenance IgnoreOpenAI", "HuggingFace", "GitHub"]
                ), axis=1
            )
        ]

        # Check if the filtered_df is smaller than the original df for those licenses which are not present in the selected_license_sources
        # i.e we expect that the filtered_df is smaller than the original df for those licenses which are not present
        for key in ["DataProvenance", "DataProvenance IgnoreOpenAI", "GitHub", "HuggingFace"]:
            if key not in selected_license_sources:
                assert len(df[f"License Use ({key})"]) >= len(filtered_df[f"License Use ({key})"]), f"Lengths don't match: {len(df[f'License Use ({key})'])} != {len(filtered_df[f'License Use ({key})'])}"

    # apply license attribution filter if selected and the license is present in selected_license_sources
    if not filtered_df.empty and selected_license_attribution:
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: all(
                    row[f"License Attribution ({key})"] <= int(selected_license_attribution)
                    for key in selected_license_sources
                    if isinstance(row[f"License Attribution ({key})"], int)
                ), axis=1
            )
        ]

    # apply license sharealike filter if selected and the license is present in selected_license_sources
    if not filtered_df.empty and selected_license_sharealike:
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda row: all(
                    row[f"License Share Alike ({key})"] <= int(selected_license_sharealike)
                    for key in selected_license_sources
                    if isinstance(row[f"License Share Alike ({key})"], int)
                ), axis=1
            )
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

    if not filtered_df.empty and no_synthetic_data:
        filtered_df = filtered_df[
            filtered_df["Model Generated"].apply(lambda x: len(x) == 0)
        ]

    if not filtered_df.empty and text_source_allow_list:
        filtered_df = filtered_df[
            filtered_df["Text Sources"].apply(lambda x: len(x) == 0 or set(x) <= set(text_source_allow_list))
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