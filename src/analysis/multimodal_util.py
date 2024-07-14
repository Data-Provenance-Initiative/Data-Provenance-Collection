import sys
import os
import datetime
import logging
import numpy as np
import pandas as pd
import altair as alt

from collections import defaultdict, Counter
from vega_datasets import data
from iso3166 import countries
from helpers import io, filters
from analysis import analysis_util
import typing

def invert_dict_of_lists(d: typing.Dict[str, typing.List[str]]) -> typing.Dict[str, str]:
    """Useful for mapping constants, paraphrases, etc.
    These are normally in the form:
        { "Category": ["item1", "item2", … ] }
    Whereas we want to invert it to:
        { "item1": "Category", "item2": "Category", … }
    """
    inverted = {}
    for k, v in d.items():
        for item in v:
            inverted[item] = k
    return inverted

def remap_licenses_with_paraphrases(
        summaries: typing.List[typing.Dict[str, typing.Any]],
        paraphrases: typing.Dict[str, str]
    ) -> typing.Dict[str, typing.Any]:
    """Map inconsistent license names to shared paraphrases using the constants.
    E.g. "CC-BY-SA 4.0", "CC BY SA 4.0" -> "CC BY-SA 4.0"
    """

    for i, summary in enumerate(summaries):
        for j, license in enumerate(summary["Licenses"]):
            license = license["License"]
            summaries[i]["Licenses"][j]["License"] = paraphrases.get(
                license,
                license
            )
        if "Inferred Metadata" in summary:
            license_keys = [
                "GitHub License",
                "HF Yaml License",
                "HF Config License",
                "PwC License"
            ]
            for key in license_keys:
                if key in summary["Inferred Metadata"]:
                    license = summary["Inferred Metadata"][key]
                    summary["Inferred Metadata"][key] = paraphrases.get(
                        license,
                        license
                    )

    return summaries

def classify_and_resolve_licenses(
    license_infos: typing.List[typing.Tuple[str, str]],
    all_constants: typing.Dict[str, typing.Dict[str, typing.List[str]]]
) -> typing.List[str]:
    """Function taken from `text_ft_plots.ipynb`"""
    classified_licenses = []
    for (license_name, license_url) in license_infos:
        # Classify an individual license
        classifications = filters.classify_license(license_name, license_url, all_constants)
        classified_licenses.append(classifications)

    # By default, multiple licenses yield to the most restrictive one
    resolved_criteria = filters.resolve_multiple_licenses(classified_licenses)
    return resolved_criteria


def add_license_classes_to_summaries(
    data_summary: typing.List[typing.Dict[str, typing.Any]],
    resolved_classes: typing.Dict[str, typing.List[str]],
    aggregator: str
):
    """Function taken from `text_ft_plots.ipynb`"""
    # Update DataFrame with columns for use, attribution, share_alike
    for row in data_summary:
        row[f"License Use ({aggregator})"] = resolved_classes[row["Unique Dataset Identifier"]][0]
        row[f"License Attribution ({aggregator})"] = resolved_classes[row["Unique Dataset Identifier"]][1]
        row[f"License Share Alike ({aggregator})"] = resolved_classes[row["Unique Dataset Identifier"]][2]
    return data_summary


def map_license_criteria_multimodal(
    data_summary: typing.List[typing.Dict[str, typing.Any]],
    all_constants: typing.Dict[str, typing.Dict[str, typing.List[str]]]
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Variant of `map_license_criteria` that works with multimodal datasets.
    Simplified to only include `Licenses` (not HF, etc.).

    Function adapted from `text_ft_plots.ipynb`.
    """

    # Unpack licenses for each dataset. {uid --> (license_name, license_url)}
    our_uid_to_license_infos = defaultdict(list)

    # Same as ours, but excludes OpenAI Terms:
    our_uid_to_license_infos_no_openai = defaultdict(list)

    for row in data_summary:
        uid = row["Unique Dataset Identifier"]
        for license_info in row["Licenses"]:
            license_name = license_info["License"]
            license_url = license_info.get("License URL", None) # FOR NOW
            our_uid_to_license_infos[uid].append((license_name, license_url))
            if license_info["License"] != "OpenAI":
                our_uid_to_license_infos_no_openai[uid].append((license_name, license_url))

        # If OpenAI was the only license, we add Unspecified so there isn't nothing there.
        if len(our_uid_to_license_infos_no_openai[uid]) == 0:
            our_uid_to_license_infos_no_openai[uid].append(("Unspecified", None))


    # classify and resolve licenses for each dataset and each aggregator
    ours_resolved, ours_openai_resolved = {}, {}
    for uid in our_uid_to_license_infos.keys():
        ours_resolved[uid] = classify_and_resolve_licenses(our_uid_to_license_infos[uid], all_constants)
        ours_openai_resolved[uid] = classify_and_resolve_licenses(our_uid_to_license_infos_no_openai[uid], all_constants)


    data_summary = add_license_classes_to_summaries(data_summary, ours_resolved, "DataProvenance")
    data_summary = add_license_classes_to_summaries(data_summary, ours_openai_resolved, "DataProvenance IgnoreOpenAI")

    return data_summary

countries_replace = { # These names need to be remapped from the original set to ISO3166
    "South Korea": "KOREA, REPUBLIC OF",
    "United Kingdom": "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND",
    "Czech Republic": "CZECHIA",
    "Vietnam": "VIET NAM",
    "Iran": "IRAN, ISLAMIC REPUBLIC OF",
    "Russia": "RUSSIAN FEDERATION",
    "UAE": "UNITED ARAB EMIRATES",
    "United States": "UNITED STATES OF AMERICA",
    "Scotland": "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND",
    "Turkey": "TÜRKIYE",
    "International/Other/Unknown": ""
}

# Text annotations contain "African Continent" in several cases
# This is a list of ISO3166 codes for the countries in the African Continent, for mapping purposes
african_continent_iso_codes = [12, 24, 204, 72, 86, 854, 108, 132, 120, 140, 148, 174, 178, 180, 384, 262, 818, 226, 232, 748, 231, 260, 266, 270, 288, 324, 624, 404, 426, 430, 434, 450, 454, 466, 478, 480, 175, 504, 508, 516, 562, 566, 638, 646, 654, 678, 686, 690, 694, 706, 710, 728, 729, 834, 768, 788, 800, 732, 894, 716]

def get_country(x: str) -> typing.List[int]:
    """Get the ISO3166 code for a country name. Returns a list for compatibility with x == "African Continent".

    Will log warnings for any countries not found.
    """
    if x == "African Continent":
        return african_continent_iso_codes
    try:
        return [countries.get(countries_replace.get(x, x))[-2]]
    except KeyError:
            logging.warning("Could not find country for %s" % x)
            return []


# Get year released for text datasets
def get_year_for_text(metadata: typing.Dict[str, typing.Any]):
    """Earliest year in the inferred metadata, for text."""
    if not isinstance(metadata, dict):
        return "Unknown"
    years = [
        datetime.datetime.strptime(metadata[c], "%Y-%m-%d").year
        for c in [
            "HF Date",
            "Github Date",
            "PwC Date",
            "S2 Date"
        ] if len(metadata.get(c, "")) > 0
    ]
    return min(years) if len(years) > 0 else "Unknown"

def plot_license_terms_stacked_bar_chart(
    df, license_palette, license_order, plot_width, plot_height, save_dir=None, plot_ppi=None
):
    chart_licenses = alt.Chart(df).mark_bar().encode(
        x=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="Pct. Datasets"
        ),
        y=alt.X("Modality:N"),
        color=alt.Color(
            "License Type:N",
            scale=alt.Scale(range=license_palette),
            title="License Use",
            sort=license_order
        ),
        order="order:Q"
    ).properties(
        title="License Use by Modality",
        width=plot_width,
        height=plot_height
    )

    if save_dir:
        chart_licenses.save(os.path.join(save_dir, "license_use_by_modality.png"), ppi=plot_ppi)

    return chart_licenses


def load_terms_metadata(data_dir):
    text_terms = pd.read_csv(os.path.join(data_dir, "text.csv")).fillna("")
    speech_terms = pd.read_csv(os.path.join(data_dir, "speech.csv")).fillna("")
    video_terms = pd.read_csv(os.path.join(data_dir, "video.csv")).fillna("")


    def interpret_row(row):
        src_verdict = row["Sources prohibit scraping?"].lower()
        src_ai_verdict = row["Sources restrict AI use?"].lower()
        src_license = row["Are the sources Noncommercial, Unspecified, or Unrestricted?"].lower()
        model_license = row.get("Model prohibits scraping?", "").lower()
        model_tos = row.get("Model ToS restrictions?", "").lower()

        # if any prohibited, prohibit, noncommercial.
        # elif any non-compete in model, model prohibits
        # elif any non-prohibited assertsions
        # elif any unspecify or "Cannot find website", unspecified
        src_rules = [src_verdict, src_ai_verdict, src_license]
        model_rules = [model_license, model_tos]
        prohibit_starts = [
            "prohibited", "likely prohibited", "noncommercial", "non-compete"
        ]
        nonprohibit_starts = [
            "not prohibited"
        ]
        unspecified_starts = [
            "unspecified", "cannot find website",
        ]
        final_verdict = None
        if any(x.startswith(w) for x in src_rules for w in prohibit_starts):
            final_verdict = "Source Closed"
        elif any(x.startswith(w) for x in model_rules for w in prohibit_starts):
            final_verdict = "Model Closed"
        elif any(x.startswith(w) for x in src_rules for w in nonprohibit_starts):
            final_verdict = "Unrestricted"
        elif any(x.startswith(w) for x in src_rules for w in unspecified_starts):
            final_verdict = "Unspecified"
        else:
            # print(row["Collection"])
            final_verdict = "Not Prohibited"
        return final_verdict

    # print(text_terms.columns)
    dset_to_value = {}
    for i, row in text_terms.iterrows():
        dset_to_value[row["Collection"]] = interpret_row(row)
    for i, row in speech_terms.iterrows():
        dset_to_value[row["Collection"]] = interpret_row(row)
    for i, row in video_terms.iterrows():
        dset_to_value[row["Collection"]] = interpret_row(row)

    return dset_to_value


def prep_summaries_for_visualization(
    text_summaries,
    speech_summaries,
    video_summaries,
    all_constants,
    collection_to_terms_mapper,
    year_categories,
    license_order,
    license_map={
        "academic-only": "NC/Acad",
        "non-commercial": "NC/Acad",
        "unspecified": "Unspecified",
        "commercial": "Commercial"
    }
):
    license_paraphrases = invert_dict_of_lists(all_constants["LICENSE_PARAPHRASES"])
    creator_groupmap = invert_dict_of_lists(all_constants["CREATOR_GROUPS"])
    creator_countrymap = invert_dict_of_lists(all_constants["CREATOR_COUNTRY_GROUPS"])
    domain_groupmap = invert_dict_of_lists(all_constants["DOMAIN_GROUPS"])
    domain_typemap = invert_dict_of_lists(all_constants["DOMAIN_TYPES"])

    text_summaries = filters.map_license_criteria(
        remap_licenses_with_paraphrases(
            text_summaries,
            license_paraphrases
        ),
        all_constants
    )

    speech_summaries = map_license_criteria_multimodal(
        remap_licenses_with_paraphrases(
            speech_summaries,
            license_paraphrases
        ),
        all_constants
    )

    video_summaries = map_license_criteria_multimodal(
        remap_licenses_with_paraphrases(
            video_summaries,
            license_paraphrases
        ),
        all_constants
    )

    df_text = pd.DataFrame(text_summaries).assign(Modality="Text")
    df_text["Data Terms"] = df_text["Collection"].apply(lambda x: collection_to_terms_mapper[x])
    df_speech = pd.DataFrame(speech_summaries).assign(Modality="Speech").rename(columns={"Location": "Countries"})
    df_speech["Data Terms"] = df_speech["Collection"].apply(lambda x: collection_to_terms_mapper[x])
    df_video = pd.DataFrame(video_summaries).assign(Modality="Video").rename(columns={"Video Sources": "Source Category"})
    df_video["Data Terms"] = df_video["Dataset Name"].apply(lambda x: collection_to_terms_mapper[x])

    df_text["Year Released"] = df_text["Inferred Metadata"].map(get_year_for_text)
    # Combine modalities
    df = pd.concat([df_text, df_speech, df_video])
    df["Model Generated"] = df["Model Generated"].fillna("")

    df["Year Released"] = pd.Categorical(
        df["Year Released"].map(
            lambda x : "<2013" if (isinstance(x, int) and x < 2013) else str(x)
        ),
        categories=year_categories
    )

    df["License Type"] = df["License Use (DataProvenance IgnoreOpenAI)"].map(license_map)
    df['License | Terms'] = df['License Type'] + ' | ' + df['Data Terms']
    # df["License Type"] = pd.Categorical(
    #     df["License Type"],
    #     categories=license_order,
    #     ordered=True
    # )
    # df = df.sort_values(by="License Type")

    # Map creators to categories (all modalities from constants, for this)
    df["Creator Categories"] = df["Creators"].map(lambda c: [creator_groupmap[ci] for ci in c])

    # For Text, we can infer the country from the creator group using the constants
    # For other modalities, they're taken from the summaries (annotated independently)
    df.loc[df["Modality"] == "Text", "Countries"] = df.loc[df["Modality"] == "Text", "Creators"].map(
        lambda x: [creator_countrymap[ci] for ci in x]
    )
    # For Text, we can infer the domain from the text sources using the constants
    # For other modalities, they're taken from the summaries (renamed columns)
    df.loc[df["Modality"] == "Text","Source Category"] = df.loc[df["Modality"] == "Text","Text Sources"].map(
        lambda x: [domain_groupmap[ci] for ci in x]
    )
    return df



def plot_stacked_temporal_license_categories(
    df,
    license_palette,
    year_categories,
    license_order,
    plot_width,
    plot_height,
    save_dir=None,
    plot_ppi=None,
    label_limit=1000,
):

    df["Year Released"] = pd.Categorical(
        df["Year Released"],
        categories=year_categories,
        ordered=True
    )

    base = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            "Year Released:N",
            title="Year Released",
            sort=year_categories
        ),
        y=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="Pct. Datasets"
        ),
        color=alt.Color(
            "License Type:N",
            scale=alt.Scale(range=license_palette),
            title="License Use",
            sort=license_order
        ),
        order="order:Q"
    ).properties(
        width=plot_width,
        height=plot_height // 2
    )

    text = alt.Chart(df).mark_text(
        dy=-68,
        align="center",
        baseline="top",
        fontSize=12
    ).encode(
        x=alt.X(
            "Year Released:N",
            sort=year_categories
        ),
        text="count():Q"
    )

    chart_licensesyears = (base + text).facet(
        row="Modality:N"
    ).properties(
        title="License Use by Modality and Dataset Release Year"
    )

    if save_dir:
        chart_licensesyears.save(os.path.join(save_dir, "license_use_by_modality+year.png"), ppi=plot_ppi)

    return chart_licensesyears



def categorize_creators(df, order):
    df_categories = df.explode("Creator Categories")
    df_categories["Creator Categories"] = df_categories["Creator Categories"].fillna("Unspecified")
    df_categories["Creator Categories"] = pd.Categorical(
        df_categories["Creator Categories"],
        categories=order,
        ordered=True
    )
    df_categories = df_categories.sort_values(by="Creator Categories")
    return df_categories

def categorize_sources(df, order, domain_typemap):

    def map_domaingroup(row) -> str:
        source_category = row["Source Category"]
        model_generated = row["Model Generated"]
        if isinstance(model_generated, list) and model_generated:
            return "Synthetic"
        if source_category in domain_typemap:
            return domain_typemap[source_category]
        if not pd.isna(source_category):
            logging.warning("Could not find domain for %s" % source_category)
        return "Other"

    # Unlist to have one row per source category (atomic components)
    df_sources = df.explode("Source Category")

    # Apply the updated map_domaingroup function to each row
    df_sources["Source Category"] = df_sources.apply(map_domaingroup, axis=1).fillna("Other")

    df_sources["Source Category"] = pd.Categorical(
        df_sources["Source Category"],
        categories=order,
        ordered=True
    )
    df_sources = df_sources.sort_values(by="Source Category")
    return df_sources

def plot_stacked_creator_categories(
    df, order, palette, pwidth, pheight, save_dir
):
    df_categories = categorize_creators(df, order)

    chart_categories = alt.Chart(df_categories).mark_bar().encode(
        x=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="Pct. Datasets"
        ),
        y=alt.X("Modality:N"),
        color=alt.Color(
            "Creator Categories:N",
            scale=alt.Scale(range=palette),
            title="Creator Category",
            sort=order
        ),
        order="order:Q"
    ).properties(
        title="Creator Categories by Modality",
        width=pwidth,
        height=pheight
    )

    if save_dir:
        chart_categories.save(os.path.join(save_dir, "creator_categories_by_modality.png"), ppi=300)

    return chart_categories


def plot_donut_creator_categories(
    df, order, palette, pheight, save_dir
):
    df_categories = categorize_creators(df, order)

    # Donut chart as alternate, to test
    chart_categoriesalt = alt.Chart(df_categories).mark_arc(innerRadius=40).encode(
        theta="count():Q",
        color=alt.Color(
            "Creator Categories:N",
            scale=alt.Scale(range=palette),
            title="Creator Category",
            sort=order
        ),
        order="order:Q"
    ).properties(
        title="Creator Categories by Modality",
        width=pheight, # Use height as width for square aspect ratio
        height=pheight
    ).facet(
        "Modality:N",
        columns=3
    )

    if save_dir:
        chart_categoriesalt.save(os.path.join(save_dir, "creator_categories_by_modality-alt.png"), ppi=300)

    return chart_categoriesalt

def plot_stacked_temporal_creator_categories(
    df, year_categories, order, palette, pwidth, pheight, save_dir
):
    df_categories = categorize_creators(df, order)

    base = alt.Chart(df_categories).mark_bar().encode(
        x=alt.X(
            "Year Released:N",
            title="Year Released",
            sort=year_categories
        ),
        y=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="Pct. Datasets"
        ),
        color=alt.Color(
            "Creator Categories:N",
            scale=alt.Scale(range=palette),
            title="Creator Category",
            sort=order
        ),
        order="order:Q"
    ).properties(
        width=pwidth,
        height=pheight // 2
    )

    text = alt.Chart(df_categories).mark_text(
        dy=-68,
        align="center",
        baseline="top",
        fontSize=12
    ).encode(
        x=alt.X(
            "Year Released:N",
            sort=year_categories
        ),
        text="count():Q"
    )

    chart_categoriesyears = (base + text).facet(
        row="Modality:N"
    ).properties(
        title="Creator Categories by Modality and Dataset Release Year"
    )

    if save_dir:
        chart_categoriesyears.save(os.path.join(save_dir, "creator_categories_by_modality+year.png"), ppi=300)

    return chart_categoriesyears


def plot_altair_worldmap(
    df,
    countries_src,
    modality_colors,
    plot_dim,
    save_dir,
):
    df_countries = df.explode("Countries").dropna(subset=["Countries"]) # Drop rows with no country for the moment
    df_countries = df_countries[["Countries", "Modality"]].value_counts().reset_index(name="Count")
    df_countries["Country ID"] = df_countries["Countries"].map(get_country)
    df_countries = df_countries.explode("Country ID").dropna(subset=["Country ID"]) # If couldn't be found (see any logged warnings), drop it

    base = alt.Chart(
        alt.topo_feature(countries_src, "countries")
    ).mark_geoshape(
        stroke="white"
    ).project(
        type="equalEarth"
    )

    charts = []

    for modality, color in modality_colors.items():
        modality_data = df_countries[df_countries["Modality"] == modality]
        chart = base.encode(
            color=alt.Color(
                "Count:Q",
                # log scale
                scale=alt.Scale(scheme=color, type="symlog"),
                title="Datasets"
            ),
            tooltip=["Countries:N", "Count:Q", "Modality:N"]
        ).properties(
            width=plot_dim,
            height=plot_dim//2
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(modality_data, "Country ID", ["Count", "Modality", "Countries"])
        ).transform_calculate(
            Count="isValid(datum.Count) ? datum.Count : 0",
            Modality="isValid(datum.Modality) ? datum.Modality : ''",
            Countries="isValid(datum.Countries) ? datum.Countries : ''"
        ).properties(
            title=modality
        )
        charts.append(chart)

    chart_map = alt.vconcat(
        *charts
    ).resolve_scale(
        color="independent"
    ).properties(
        title="Dataset Count by Country and Modality"
    )

    if save_dir:
        chart_map.save(os.path.join(save_dir, "dataset_count_by_country_and_modality.png"), ppi=300)

    return chart_map

def plot_source_domain_stacked_chart(
    df, domain_typemap, order, pwidth, pheight, save_dir
):
    df_sources = categorize_sources(df, order, domain_typemap)
    chart_sources = alt.Chart(df_sources).mark_bar().encode(
        x=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="Pct. Datasets"
        ),
        y=alt.X("Modality:N"),
        color=alt.Color(
            "Source Category:N",
            title="Source Category",
            sort=order
        ),
        order="order:Q"
    ).properties(
        title="Source Categories by Modality",
        width=pwidth,
        height=pheight
    )

    if save_dir:
        chart_sources.save(os.path.join(save_dir, "source_categories_by_modality.png"), ppi=300)

    return chart_sources


def plot_stacked_temporal_source_categories(
    df, year_categories, order, domain_typemap, pwidth, pheight, save_dir,
):

    df_sources = categorize_sources(df, order, domain_typemap)

    base = alt.Chart(df_sources).mark_bar().encode(
        x=alt.X(
            "Year Released:N",
            title="Year Released",
            sort=year_categories
        ),
        y=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="Pct. Datasets"
        ),
        color=alt.Color(
            "Source Category:N",
            title="Source Category",
            sort=order
        ),
        order="order:Q"
    ).properties(
        width=pwidth,
        height=pheight // 2
    )

    text = alt.Chart(df_sources).mark_text(
        dy=-68,
        align="center",
        baseline="top",
        fontSize=12
    ).encode(
        x=alt.X(
            "Year Released:N",
            sort=year_categories
        ),
        text="count():Q"
    )

    chart_sourcesyears = (base + text).facet(
        row="Modality:N"
    ).properties(
        title="Source Categories by Modality and Dataset Release Year"
    )

    if save_dir:
        chart_sourcesyears.save(os.path.join(save_dir, "source_categories_by_modality+year.png"), ppi=300)

    return chart_sourcesyears

def text_groupby_collection(df, mode_column, fn):

    df_text = df[df["Modality"] == "Text"].copy()
    df_nontext = df[df["Modality"] != "Text"]

    df_text.loc[:, mode_column] = df_text.groupby("Collection")[mode_column].transform(fn)

    df_text = df_text.drop_duplicates(subset="Collection")
    new_df = pd.concat([df_nontext, df_text], ignore_index=True)
    return new_df


def plot_source_domain_stacked_chart_collections(
    df, domain_typemap, order, pwidth, pheight, save_dir
):

    df_sources = categorize_sources(df, order, domain_typemap)
    df_sources = text_groupby_collection(df_sources, "Source Category",
        fn=lambda x: x.mode()[0] if not x.mode().empty else "Unspecified")

    logging.warning("Aggregating to %d collections" % len(df_sources.loc[df_sources["Modality"] == "Text", "Collection"].unique()))

    df_sources = df_sources.sort_values(by="Source Category")
    chart_sourcesaggregated = alt.Chart(df_sources).mark_bar().encode(
        x=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="Pct. Datasets"
        ),
        y=alt.X("Modality:N"),
        color=alt.Color(
            "Source Category:N",
            title="Source Category",
            sort=order
        ),
        order="order:Q"
    ).properties(
        title="Source Categories by Modality (Aggregated Collections)",
        width=pwidth,
        height=pheight
    )

    if save_dir:
        chart_sourcesaggregated.save(os.path.join(save_dir, "source_categories_by_modality-aggregated.png"), ppi=300)

    return chart_sourcesaggregated


# 'Non-Commercial/Academic', 'Unspecified', 'Commercial'
def license_rank_fn(license_list):
    if not isinstance(license_list, list):
        license_list = license_list.tolist()
    if "NC/Acad" in license_list:
        return "NC/Acad"
    elif "Unspecified" in license_list:
        return "Unspecified"
    else:
        return "Commercial"

def terms_rank_fn(terms_list):
    if not isinstance(terms_list, list):
        terms_list = terms_list.tolist()
    if "Model Closed" in terms_list:
        return "Model Closed"
    elif "Source Closed" in terms_list:
        return "Source Closed"
    elif "Unspecified" in terms_list:
        return "Unspecified"
    else:
        return "Unrestricted" 

def license_terms_rank_fn(license_list):
    ll = license_list.tolist()
    # TODO: Fix.
    if not all(isinstance(x, str) for x in ll):
        print(ll)
        return "NC/Acad | Source Closed"
    prefix = license_rank_fn([x.split(" | ")[0] for x in ll])
    suffix = terms_rank_fn([x.split(" | ")[1] for x in ll])
    return prefix + " | " + suffix


def plot_license_terms_stacked_bar_chart_collections(
    df, 
    license_key,
    license_palette, 
    license_order, 
    plot_width, 
    plot_height,
    save_dir=None, 
    plot_ppi=None,
    font_size=15,
):
    if license_key == "License Type":
        hierarchy_fn = license_rank_fn
    else:
        hierarchy_fn = license_terms_rank_fn

    df[license_key] = pd.Categorical(
        df[license_key],
        categories=license_order,
        ordered=True
    )
    df = df.sort_values(by=license_key)
    df = text_groupby_collection(df, license_key, fn=hierarchy_fn,)

    # modality_specific = df[df["Modality"] == "Speech"]
    # print(modality_specific.columns)
    # print(modality_specific[['Modality', 'License | Terms']])
    # print(modality_specific[license_key].value_counts())

    # print(df.groupby(['Modality', license_key]).size().reset_index(name='counts'))
    # grouped_data = df.groupby(['Modality', license_key]).size().reset_index(name='counts')
    # print(grouped_data)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.Y(
            "count():Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            # title="Pct. Datasets",
            title="",
        ),
        y=alt.X("Modality:N", title=""),
        color=alt.Color(
            f"{license_key}:N",
            scale=alt.Scale(domain=license_order, range=license_palette),  # Map each license to a specific color
            title=license_key,
            sort=license_order,
        ),
        order=alt.Order("order:Q", sort="ascending")  # Ensures correct order of the bars
    )
    
    chart = chart.properties(
        width=plot_width,
        height=plot_height
    ).configure_axis(
        labelFontSize=font_size,
        titleFontSize=font_size,
    ).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
        orient='bottom',
        columns=4,
        labelLimit=200,
    )

    if save_dir:
        chart.save(os.path.join(save_dir, "license_use_by_modality_collections.png"), ppi=plot_ppi)

    table = generate_multimodal_license_terms_latex(df)
    return chart, table




def gini(array: np.ndarray) -> float:
    """Calculate the Gini coefficient of a numpy array.

    Implementation taken from: https://github.com/oliviaguest/gini
    """
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def bootstrap_cis_for_gini(
    data: np.ndarray,
    n_samples: int = 1000,
    alpha: float = 0.05
) -> typing.Tuple[float, float]:
    """Estimate the confidence interval for the Gini coefficient using bootstrapping.
    """

    ginis = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        ginis.append(gini(sample))

    ginis = np.array(ginis)
    lower_bound = np.percentile(ginis, alpha / 2 * 100)
    upper_bound = np.percentile(ginis, (1 - alpha / 2) * 100)

    return np.mean(ginis), lower_bound, upper_bound


def factor_year(
    df: pd.DataFrame,
    column: str = "Year Released",
    min_year: int = 2013
) -> pd.DataFrame:
    """Transform the year column into a categorical column.

    Years before `min_year` are grouped into a category, i.e. "<`min_year`" (e.g. )
    """
    df = df.copy()

    min_yeartext = "<%d" % min_year
    max_year = df[column].max()

    df[column] = df[column].map(
        lambda x: min_yeartext if (x < min_year) else str(x)
    )

    order = [min_yeartext, *map(str, range(min_year, max_year + 1))]

    df[column] = pd.Categorical(
        df[column],
        categories=order,
        ordered=True
    )

    return df, order


def order_by_grouped_permisiveness(
    df: pd.DataFrame,
    group_column: str,
    licensetype_column: str = "License Type",
    permissive_licensetypes: typing.List[str] = ["Commercial"]
) -> pd.Series:
    """Given a DataFrame, group it by `group_column` and calculate the permissiveness of each group.

    Permisiveness is calculated as the proportion of licenses that are in `permissive_licensetypes`.
    """
    permisiveness = df.groupby(group_column).apply(
        lambda x: (x[licensetype_column].isin(permissive_licensetypes)).mean()
    ).reset_index(name="Permissiveness")

    permisiveness_order = permisiveness.sort_values(by="Permissiveness")[group_column].tolist()

    return permisiveness_order


def reduce_categories_to_topk(
    df: pd.DataFrame,
    column: str,
    k: int = 6
) -> pd.DataFrame:
    """Reduce the number of categories in a column to the top `k` categories.

    The rest are grouped into an "Other" category.
    """
    df = df.copy()
    topk = df[column].value_counts().head(k).index.tolist()
    df[column] = df[column].map(
        lambda x: x if x in topk else "Other"
    )

    return df


def generate_multimodal_license_terms_latex(df):
    dfx = df.groupby(['Modality', 'License | Terms']).size().reset_index(name='counts')

    # Apply the function and assign the results to new columns
    dfx['License Info'] = dfx['License | Terms'].apply(lambda x: x.split(' | ')[0].strip())
    dfx['Terms Info'] = dfx['License | Terms'].apply(lambda x: x.split(' | ')[1].strip())

    # List of all modalities
    modalities = dfx['Modality'].unique()
    
    # Initialize an empty string to store all LaTeX tables
    latex_outputs = {}
    
    # Process each modality
    for modality in modalities:
        # Filter the dataframe for the current modality
        modality_df = dfx[dfx['Modality'] == modality]
        
        # Get unique License Info and Terms Info
        license_info = modality_df['License Info'].unique()
        terms_info = modality_df['Terms Info'].unique()
        
        # Create an empty dictionary to hold counts
        counts_dict = {license: {term: 0 for term in terms_info} for license in license_info}
        
        # Add 'Total' key with a dictionary initialized for terms and total
        counts_dict['Total'] = {term: 0 for term in terms_info}
        counts_dict['Total']['Total'] = 0
        
        # Ensure each license has a 'Total' key
        for license in counts_dict:
            counts_dict[license]['Total'] = 0
        
        # Fill the dictionary with counts
        for _, row in modality_df.iterrows():
            license = row['License Info']
            term = row['Terms Info']
            count = row['counts']
            counts_dict[license][term] += count
            counts_dict[license]['Total'] += count
            counts_dict['Total'][term] += count
            counts_dict['Total']['Total'] += count
        
        # Calculate the percentage of each cell relative to the total counts
        total_counts = counts_dict['Total']['Total']
        for license in counts_dict:
            for term in counts_dict[license]:
                counts_dict[license][term] = round((counts_dict[license][term] / total_counts) * 100, 1)
        
        # Generate LaTeX table
        latex_table = "\\begin{table*}[t!]\n\\centering\n\\begin{adjustbox}{width=0.98\\textwidth}\n"
        latex_table += "\\begin{tabular}{l|" + "r" * len(terms_info) + "|r}\n\\toprule\n"
        joined_col_headers = " & ".join(["\\textsc{" + ti + "}" for ti in terms_info])
        latex_table += "\\textsc{License / Terms} & " + joined_col_headers + " & \\textsc{Total} \\\\\n"
        latex_table += "\\midrule\n"
        for license in counts_dict:
            lic_vals = [str(counts_dict[license][term]) for term in terms_info]
            total_vals = str(counts_dict[license]['Total'])
            if license == "Total":
                latex_table += "\\midrule\n"
            latex_table += "\\textsc{" + license + "} & " + " & ".join(lic_vals) + " & " + total_vals + " \\\\\n"
        mod_label = modality.lower() + "_license_terms_breakdown"
        unspec_or_open = counts_dict["Commercial"]["Unrestricted"] + counts_dict["Commercial"]["Unspecified"] + counts_dict["Unspecified"]["Unrestricted"] + counts_dict["Unspecified"]["Unspecified"]
        closed_pct = round(100 - unspec_or_open, 1)
        total_nc_license = round(counts_dict["NC/Acad"]["Total"], 1)
        total_restrictive_terms = round(counts_dict["Total"]["Source Closed"] + counts_dict["Total"]["Model Closed"], 1)
        caption = "\\textbf{A breakdown of " + modality + " Dataset licenses, and the Terms attached to their sources.} "
        caption += f"Among {modality} datasets, while only {total_nc_license}\% are Non-Commerically licensed, or have {total_restrictive_terms}\% have restrictive terms, a full {closed_pct}\% of datasets have either a restrictive license or terms."
        latex_table += "\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\caption{" + caption + "}\n\\label{tab:" + mod_label + "}\n\\end{table*}\n"
        
        latex_outputs[modality] = latex_table
    
    return latex_outputs


def tokens_calculation(df):
    # Extract and aggregate Text Metrics
    df = df.copy()
    text_metrics = df[['Unique Dataset Identifier', 'Text Metrics']].groupby('Unique Dataset Identifier').first().reset_index()

    # Normalize the 'Text Metrics' column
    metrics_df = pd.json_normalize(text_metrics['Text Metrics'])
    metrics_df['Unique Dataset Identifier'] = text_metrics['Unique Dataset Identifier']

    # Columns to check for NaN
    columns_to_check = [
        'Num Dialogs', 'Mean Inputs Length', 'Mean Targets Length',
        'Max Inputs Length', 'Max Targets Length', 'Min Inputs Length',
        'Min Targets Length', 'Min Dialog Turns', 'Max Dialog Turns',
        'Mean Dialog Turns'
    ]

    # Filter out rows where all specified columns are NaN
    metrics_df = metrics_df[metrics_df[columns_to_check].notna().any(axis=1)]

    # Calculate 'Total Tokens' using a lambda function
    metrics_df['Total Tokens'] = metrics_df.apply(lambda row: (
        row['Num Dialogs'] * row['Mean Dialog Turns'] * (row['Mean Inputs Length'] + row['Mean Targets Length'])
    ), axis=1)

    # Select relevant columns and merge with original DataFrame
    df_token = metrics_df[['Unique Dataset Identifier', 'Total Tokens']]
    tokens_output = pd.merge(df, df_token, on='Unique Dataset Identifier', how='left')

    return tokens_output

def chart_creation(df: pd.DataFrame, max_count: int, x_field: str, labels: list, ratio: float, title: str, width: int, height: int, color: str):
    chart1 = alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X(f'{x_field}:N', sort=labels, axis=alt.Axis(labelAngle=45)),
        y=alt.Y('Number of datasets:Q', scale=alt.Scale(domain=[0, max_count * ratio]), axis=alt.Axis(grid=True, tickCount=5)),
        tooltip=[x_field, 'Number of datasets']
    ).properties(
        title=title,
        width=width,
        height=height
    )
    return chart1

def combined_charts(*charts):
    # Concatenate the two charts horizontally with different scales for the y-axes
    combined_chart = alt.hconcat(*charts).configure(
        title={"font": "Times New Roman"},
        axis={
            "labelFont": "Times New Roman",
            "titleFont": "Times New Roman",
            "labelFontSize": 16,
            "titleFontSize": 16,
            "labelAngle": 0,
            "grid": True,
            "gridOpacity": 0.7,
            "tickColor": "grey",
            "tickSize": 10,
            "tickWidth": 2,
            "tickOpacity": 0.8
        },
        legend={
            "labelFont": "Times New Roman",
            "titleFont": "Times New Roman",
            "titleFontSize": 16,
            "labelFontSize": 16
        }
    )
    return combined_chart

def data_aggregation_for_chart(df, modality: str, bins, labels,  measure_column: str, group_column: str, by_collection=False):
    df_modality = df[df['Modality'] == modality]
    if by_collection:
        # Aggregate data by 'Collection' if specified
        df_modality = df_modality.groupby('Collection')[measure_column].sum().reset_index()
        df_modality[group_column] = pd.cut(df_modality[measure_column], bins=bins, labels=labels, right=False)
    else:
        # Normal token grouping
        df_modality[group_column] = pd.cut(df_modality[measure_column], bins=bins, labels=labels, right=False)

    group_distribution = df_modality[group_column].value_counts().sort_index()

    df1 = group_distribution.reset_index()
    df1.columns = [group_column, 'Count']

    df1[group_column] = df1[group_column].astype(str)
    df1['Number of datasets'] = df1['Count'].astype(int)

    max_count1 = df1['Number of datasets'].max()

    return df1, max_count1