import sys
import os
import datetime
import logging
import json
import numpy as np
import pandas as pd
import altair as alt
import langcodes

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


def read_continent_country_iso_codes():
    """
    Read continent, country and ISO codes mapping from the JSON file.
    """

    try:
        with open('../../constants/continent_country_iso.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error("The file '../../constants/continent_country_iso.json' was not found.")
        return []
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the file '../../constants/continent_country_iso.json'.")
        return []

    continent_country_iso_list = []

    for continent_data in data['continents']:
        continent = continent_data['name']
        countries = continent_data['countries']
        for country_data in countries:
            country = country_data['name']
            iso_numeric = country_data['iso_code']
            continent_country_iso_list.append({'continent': continent, 'country': country, 'iso_code': iso_numeric})
    return continent_country_iso_list

def get_continent(x: str, continent_country_iso_list: list) -> typing.List[int]:
    """
    Get the continent for a given country name, case-insensitive.

    """
    df_continent_country_iso = pd.DataFrame(continent_country_iso_list)
    continent_set = set()
    for country in x:
        if country == "African Continent":
            continent_set.update(['Africa'])
            continue
        if country == "International/Other/Unknown":
            # logging.warning(f"Country '{country}' not found in the data.")
            continue
        country = countries_replace.get(country, country)
        filtered_country= df_continent_country_iso[df_continent_country_iso['country'].str.lower() == country.lower()]

        if len(filtered_country) > 0:
            continent_set.update(filtered_country['continent'].unique().tolist())
        else:
            logging.warning(f"Country '{country}' not found in the data.")

    return list(continent_set)


def get_continent_id(x: str, continent_country_iso_list: list) -> typing.List[int]:
    """
    Get list of ISO numeric codes of countriues for a given continent.

    """

    df_continent_country_iso = pd.DataFrame(continent_country_iso_list)
    continent_iso_ids = df_continent_country_iso[df_continent_country_iso['continent'] == x]['iso_code'].apply(int).unique().tolist()
    if len(continent_iso_ids) > 0:
        return continent_iso_ids
    else:
        logging.warning(f"Country '{x}' not found in the data.")
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
            "not prohibited", "unrestricted"
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
            final_verdict = "Unspecified"
        return final_verdict

    # print(text_terms.columns)
    dset_to_value = {}
    for i, row in text_terms.iterrows():
        # if row["Collection"] == "Glaive Code Assistant":
        #     print(row, interpret_row(row))
        dset_to_value[row["Collection"]] = interpret_row(row)
    for i, row in speech_terms.iterrows():
        val = interpret_row(row)
        # print(val)
        dset_to_value[row["Collection"]] = val
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
    lang_typmap = invert_dict_of_lists(all_constants["LANGUAGE_GROUPS"])

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
    # print(df_text[df_text['Text Sources'].apply(lambda x: 'wikipedia,org' in x)])
    df_text = tokens_calculation(df_text)
    df_text = df_text[df_text["Collection"].isin(collection_to_terms_mapper.keys())]
    df_text["Data Terms"] = df_text["Collection"].apply(lambda x: collection_to_terms_mapper[x])
    df_text["Language Families"] = df_text["Languages"].map(lambda c: [lang_typmap[ci] for ci in c])
    df_speech = pd.DataFrame(speech_summaries).assign(Modality="Speech").rename(columns={"Location": "Countries"})
    df_speech["Data Terms"] = df_speech["Collection"].apply(lambda x: collection_to_terms_mapper[x])
    df_video = pd.DataFrame(video_summaries).assign(Modality="Video").rename(columns={
        "Video Sources": "Source Category", "Video Hours": "Hours"})
    df_video["Data Terms"] = df_video["Dataset Name"].apply(lambda x: collection_to_terms_mapper[x])

    df_text["Year Released"] = df_text["Inferred Metadata"].map(get_year_for_text)
    # Combine modalities
    df = pd.concat([df_text, df_speech, df_video])
    df["Model Generated"] = df["Model Generated"].fillna("")

    df["Year Released Category"] = pd.Categorical(
        df["Year Released"].map(
            lambda x : "<2013" if (isinstance(x, int) and x < 2013) else str(x)
        ),
        categories=year_categories
    )

    df["License Type"] = df["License Use (DataProvenance IgnoreOpenAI)"].map(license_map)
    df['License | Terms'] = df['License Type'] + ' | ' + df['Data Terms']

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
    df["Source Category"] = df.apply(
        lambda row: row["Source Category"] + ["Synthetic"]
        if len(row["Model Generated"]) > 0 and "templates" not in [x.lower() for x in row["Model Generated"]]
        and "other" not in [x.lower() for x in row["Model Generated"]]
        else row["Source Category"],
        axis=1
    )
    df["Source Category"] = df.apply(lambda row: [x for x in row["Source Category"] if x != "Unsure"], axis=1)

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
            print(row["Unique Dataset Identifier"], row["Modality"])
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
    df_categories, order, modality_order, palette, pwidth, pheight,
    save_dir=None, collection_level=False,
):
    if collection_level:
        def unpack_list(cats):
            cat_list = cats.tolist()
            num_vars = len(cat_list)
            flatten_cats = list(set([cat for catsll in cat_list for cat in catsll]))
            # print(flatten_cats)
            replicated_list = [flatten_cats[:] for _ in range(num_vars)]
            return replicated_list

        df_categories = text_groupby_collection(df_categories, "Creator Categories",
            fn=unpack_list)
    df_categories = categorize_creators(df_categories, order)

    # Add counts for calculating percentages
    df_categories['count'] = 1
    df_grouped = df_categories.groupby(['Modality', "Creator Categories"]).size().reset_index(name='count')

    # Add percentage column based on counts
    df_grouped['percentage'] = df_grouped.groupby('Modality')['count'].transform(lambda x: (x / x.sum()) * 100)

    def calculate_midpoints(points):
        midpoints = []
        for i in range(len(points)):
            trailing_quant = points.iloc[:i]
            midpoints.append((points.iloc[i]/2 + sum(trailing_quant))/100)

        return midpoints

    df_grouped['midpoints'] = df_grouped.groupby('Modality')['percentage'].transform(calculate_midpoints)

    # Base chart
    base = alt.Chart(df_grouped).encode(
        x=alt.Y(
            "percentage:Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="",
        ),
        y=alt.X("Modality:N", title="", sort=modality_order)
    ).properties(
        # title="Creator Categories by Modality",
        width=pwidth,
        height=pheight
    )

    # Create bars
    bars = base.mark_bar().encode(
        color=alt.Color(
            "Creator Categories:N",
            # scale=alt.Scale(range=palette),
            title="Creator Category",
            sort=order
        ),
        order="order:Q"
    )

    # Text annotations inside bars for percentages > 10%
    text = bars.mark_text(
        align='center',
        fontSize=14,
    ).encode(
        x='midpoints',
        text=alt.condition(
            alt.datum.percentage > 5,
            alt.Text('percentage:Q', format='.1f'),
            alt.value('')
        ),
        color=alt.value('white')
    )

    chart_categories = bars + text
    chart_categories = chart_categories.configure_axis(
            labelFontSize=15,
            titleFontSize=15,
    ).configure_legend(
        labelFontSize=14,
        titleFontSize=15,
        orient='bottom',
        columns=8,
        labelLimit=200,
    )
    if save_dir:
        chart_categories.save(os.path.join(save_dir, "creator_categories_by_modality.svg"), format='svg')

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


def plot_altair_worldmap_country(
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
                title="Datasets",
                legend=None,
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

    chart_map = alt.hconcat(
        *charts,
        spacing=-400,
    ).resolve_scale(
        color="independent"
    ).properties(
        # title="Dataset Count by Country and Modality",
        # width=600,
        # height=200,
    )

    # chart_map.properties(width=400, height=100)


    if save_dir:
        chart_map.save(os.path.join(save_dir, "dataset_count_by_country_and_modality.png"), ppi=300)

    return chart_map


def map_country_to_continent(df):
    df_countries = df.copy()
    continent_country_iso_list = read_continent_country_iso_codes()
    df_countries["Continent"] = df_countries["Countries"].map(lambda x: get_continent(x, continent_country_iso_list))
    df_countries = df_countries.explode("Continent").dropna(subset=["Continent"])
    df_countries = df_countries[["Continent", "Modality", "Total Tokens", "Hours"]]
    df_countries['Total Hours'] = df_countries.apply(lambda row: row['Total Tokens'] if row['Modality'] == 'Text' else row['Hours'], axis=1)

    # Now you can group and sum based on Continent and Modality
    df_countries = df_countries.groupby(["Continent", "Modality"]).agg(
        Count=('Modality', 'size'),
        Total_Hours=('Total Hours', 'sum')
    ).reset_index()

    # print(df_countries_hrs)
    # df_countries = df_countries[["Continent", "Modality"]].value_counts().reset_index(name="Count")
    # print(df_countries)
    percent_df = pd.DataFrame(0, index=df_countries["Modality"].unique(), columns=df_countries["Continent"].unique())
    percent_df_hrs = pd.DataFrame(0, index=df_countries["Modality"].unique(), columns=df_countries["Continent"].unique())
    modality_totals = df_countries.groupby('Modality')['Count'].sum()
    modality_total_hrs = df_countries.groupby('Modality')['Total_Hours'].sum()

    # Fill the percentage table by iterating over each row in the original dataframe
    for _, row in df_countries.iterrows():
        modality = row['Modality']
        continent = row['Continent']
        count = row['Count']
        dims = row["Total_Hours"]
        # dims = row["Total Tokens"] if modality == "Text" else row["Hours"]

        # Calculate the percentage
        total_for_modality = modality_totals[modality]
        percentage = (count / total_for_modality) * 100

        # Assign the percentage to the appropriate cell
        percent_df.at[modality, continent] = round(percentage, 1)
        percent_df_hrs.at[modality, continent] = round(100 * dims / modality_total_hrs[modality], 1)

    # Convert to LaTeX
    modalities_order = ["Text", "Speech", "Video"]
    continents_order = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
    percent_df_ordered = percent_df.reindex(index=modalities_order, columns=continents_order)
    latex_table_ordered = percent_df_ordered.to_latex(float_format="%.1f")

    percent_df_hrs_ordered = percent_df_hrs.reindex(index=modalities_order, columns=continents_order)
    latex_table2_ordered = percent_df_hrs_ordered.to_latex(float_format="%.1f")

    return latex_table_ordered, latex_table2_ordered
    # return df_countries

def plot_altair_worldmap_continent(
    df,
    countries_src,
    modality_colors,
    plot_dim,
    save_dir
):

    # if aggregate_level == "Country":
    #     df_countries = df.explode("Countries").dropna(subset=["Countries"]) # Drop rows with no country for the moment
    #     df_countries = df_countries[["Countries", "Modality"]].value_counts().reset_index(name="Count")
    #     df_countries["Country ID"] = df_countries["Countries"].map(get_country)
    #     df_countries = df_countries.explode("Country ID").dropna(subset=["Country ID"]) # If couldn't be found (see any logged warnings), drop it
    # else:
    continent_country_iso_list = read_continent_country_iso_codes()
    df_countries = df
    df_countries["Continent"] = df_countries["Countries"].map(lambda x: get_continent(x, continent_country_iso_list))
    df_countries = df_countries.explode("Continent").dropna(subset=["Continent"])
    # df_countries.to_csv('df_continent.csv')
    df_countries = df_countries[["Continent", "Modality"]].value_counts().reset_index(name="Count")
    df_countries["Continent ISO ID"] = df_countries["Continent"].map(lambda x: get_continent_id(x, continent_country_iso_list))
    df_countries = df_countries.explode("Continent ISO ID").dropna(subset=["Continent ISO ID"]) # If couldn't be found (see any logged warnings), drop it

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
            tooltip=["Continents:N", "Count:Q", "Modality:N"]
        ).properties(
            width=plot_dim,
            height=plot_dim//2
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(modality_data, "Continent ISO ID", ["Count", "Modality", "Continent"])
        ).transform_calculate(
            Count="isValid(datum.Count) ? datum.Count : 0",
            Modality="isValid(datum.Modality) ? datum.Modality : ''",
            Continents="isValid(datum.Continent) ? datum.Continent : ''"
        ).properties(
            title=modality
        )
        charts.append(chart)

    chart_map = alt.hconcat(
        *charts
    ).resolve_scale(
        color="independent"
    ).properties(
        title="Dataset Count by Continent and Modality"
    )

    if save_dir:
        chart_map.save(os.path.join(save_dir, "dataset_count_by_continent_and_modality.png"), ppi=300)

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

def text_groupby_collection(df, mode_column, fn, txt_mod_col="Text"):

    df_text = df[df["Modality"] == txt_mod_col].copy()
    df_nontext = df[df["Modality"] != txt_mod_col]

    # print(df_text[["Collection", "License | Terms"]])
    df_text.loc[:, mode_column] = df_text.groupby("Collection")[mode_column].transform(fn)
    # df_nontext = df_nontext[mode_column].transform(fn)

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

def merge_to_restricted(license_term):
    # Split the license and the term
    license_part, term_part = license_term.split(' | ')

    # If the term is "Source Closed" or "Model Closed", change it to "Restricted"
    if term_part in ['Source Closed', 'Model Closed']:
        return f"{license_part} | Restricted"

    # Otherwise, return the original term
    return license_term

def prepare_license_terms_temporal_plot(
    df,
    license_key,
    license_palette,
    license_order,
):
    if license_key == "License Type":
        hierarchy_fn = license_rank_fn
    else:
        hierarchy_fn = license_terms_rank_fn

    df = df.copy()
    df[license_key] = pd.Categorical(
        df[license_key],
        categories=license_order,
        ordered=True
    )
    df = df.sort_values(by=license_key)
    df = text_groupby_collection(df, license_key, fn=hierarchy_fn,)

    # df = df[["Modality", license_key]]
    return df

def plot_license_terms_stacked_bar_chart_collections(
    df,
    license_key,
    license_palette,
    license_order,
    modality_order,
    plot_width,
    plot_height,
    title="",
    no_legend=False,
    save_dir=None,
    plot_ppi=None,
    split_text_mod=False,
    pct_by_tokens=False,
    font_size=15,
    return_license_table=True,
    configure_chart=True,
):
    if license_key == "License Type":
        hierarchy_fn = license_rank_fn
    else:
        hierarchy_fn = license_terms_rank_fn

    df = df.copy()
    df = df.sort_values(by=license_key)

    if split_text_mod:
        df_text = df[df['Modality'] == 'Text'].copy()
        df_text_datasets = df_text.copy()
        df_text_datasets['Modality'] = 'Text (Datasets)'
        df_text_collections = df_text.copy()
        df_text_collections['Modality'] = 'Text (Collections)'
        df_non_text = df[df['Modality'] != 'Text']
        df = pd.concat([df_non_text, df_text_datasets, df_text_collections], ignore_index=True)
        modality_name = "Text (Collections)"
    else:
        modality_name = "Text"

    df = text_groupby_collection(df, license_key, fn=hierarchy_fn, txt_mod_col=modality_name)
    df[license_key] = df[license_key].apply(merge_to_restricted)
    df[license_key] = pd.Categorical(
        df[license_key],
        categories=license_order,
        ordered=True
    )
    # Modify the order of the modality
    df['Modality'] = pd.Categorical(df['Modality'], categories=modality_order, ordered=True)

    # Add counts for calculating percentages
    df['count'] = 1
    # print(df_grouped)
    if pct_by_tokens:
        df_grouped = df.groupby(['Modality', license_key]).agg({
            'Hours': 'sum',
            'Total Tokens': 'sum'
        }).reset_index()

        # Create a new column to hold the quantity based on the Modality
        df_grouped['quantity'] = np.where(df_grouped['Modality'].isin(['Speech', 'Video']),
                                        df_grouped['Hours'],
                                        df_grouped['Total Tokens'])

        # Now calculate the percentage based on the 'quantity' column
        df_grouped['percentage'] = df_grouped.groupby('Modality')['quantity'].transform(lambda x: (x / x.sum()) * 100)
        df_melted = df_grouped.melt(id_vars=['Modality', license_key, 'percentage'], value_vars=['quantity'], var_name='metric', value_name='value')
    else:
        df_grouped = df.groupby(['Modality', license_key]).size().reset_index(name='count')
        df_grouped['percentage'] = df_grouped.groupby('Modality')['count'].transform(lambda x: (x / x.sum()) * 100)
        df_melted = df_grouped.melt(id_vars=['Modality', license_key, 'percentage'], value_vars=['count'], var_name='metric', value_name='value')

    # Calculate midpoints for text annotations
    def calculate_midpoints(points):
        midpoints = []
        for i in range(len(points)):
            trailing_quant = points.iloc[:i]
            midpoints.append((points.iloc[i]/2 + sum(trailing_quant))/100)

        return midpoints
    df_melted['midpoints'] = df_melted.groupby('Modality')['percentage'].transform(calculate_midpoints)

    # Base chart
    base = alt.Chart(df_melted).encode(
        x=alt.Y(
            "percentage:Q",
            stack="normalize",
            axis=alt.Axis(format="%"),
            title="",
        ),
        y=alt.X("Modality:N", title="", sort=modality_order)
    ).properties(
        title=title,
        width=plot_width,
        height=plot_height
    )

    # Create bars
    if no_legend:
        colors = alt.Color(
                f"{license_key}:N",
                scale=alt.Scale(domain=license_order, range=license_palette),
                title=license_key,
                legend=None
            )
    else:
        colors = alt.Color(
                    f"{license_key}:N",
                    scale=alt.Scale(domain=license_order, range=license_palette),
                    title=license_key,
                )

    bars = base.mark_bar().encode(
        color=colors,
        order=alt.Order(
            "order:Q",
            sort='ascending'
        )
    )

    # Text annotations inside bars for percentages > 10%
    text = bars.mark_text(
        align='center',
        fontSize=font_size,
    ).encode(
        x='midpoints',
        text=alt.condition(
            alt.datum.percentage > 5,
            alt.Text('percentage:Q', format='.1f'),
            alt.value('')
        ),
        color=alt.value('white')
    )

    chart = bars + text

    if configure_chart:
        chart = chart.configure_axis(
            labelFontSize=font_size,
            titleFontSize=font_size+2,
        ).configure_legend(
            labelFontSize=font_size+1,
            titleFontSize=font_size+1,
            orient='bottom',
            columns=3,
            labelLimit=400,
        ).configure_title(
            fontSize=font_size + 1  # Increase title font size
        )

    # Save chart to file if a directory is provided
    if save_dir:
        chart.save(os.path.join(save_dir, "license_use_by_modality_collections.svg"), format='svg')

    # Return the chart and the generated LaTeX table if requested
    if return_license_table:
        table = generate_multimodal_license_terms_latex(df)
        return chart, table

    return chart

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
    # print(df)
    dfx = df.groupby(['Modality', 'License Type', 'Data Terms']).size().reset_index(name='counts')
    # print(dfx)

    dfx['License Info'] = dfx["License Type"]
    dfx['Terms Info'] = dfx["Data Terms"]
    # Apply the function and assign the results to new columns
    # dfx['License Info'] = dfx['License | Terms'].apply(lambda x: x.split(' | ')[0].strip())
    # dfx['Terms Info'] = dfx['License | Terms'].apply(lambda x: x.split(' | ')[1].strip())

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
        # terms_info = modality_df['Terms Info'].unique()
        terms_info = ["Model Closed", "Source Closed", "Unspecified", "Unrestricted"]

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

        # Ensure "Source Closed" and "Model Closed" are present in 'Total'
        if 'Source Closed' not in counts_dict['Total']:
            counts_dict['Total']['Source Closed'] = 0
        if 'Model Closed' not in counts_dict['Total']:
            counts_dict['Total']['Model Closed'] = 0

        # Calculate the percentage of each cell relative to the total counts
        total_counts = counts_dict['Total']['Total']
        if total_counts > 0:
            for license in counts_dict:
                for term in counts_dict[license]:
                    counts_dict[license][term] = round((counts_dict[license][term] / total_counts) * 100, 1)

        # Generate LaTeX table
        latex_table = "\\begin{table*}[t!]\n\\centering\n\\begin{adjustbox}{width=0.98\\textwidth}\n"
        latex_table += "\\begin{tabular}{l|" + "r" * len(terms_info) + "|r}\n\\toprule\n"
        joined_col_headers = " & ".join(["\\textsc{" + ti + "}" for ti in terms_info])
        latex_table += "\\textsc{License / Terms} & " + joined_col_headers + " & \\textsc{Total} \\\\\n"
        latex_table += "\\midrule\n"
        for license in ["NC/Acad", "Unspecified", "Commercial", "Total"]:
            lic_vals = [str(counts_dict[license][term]) for term in terms_info]
            total_vals = str(counts_dict[license]['Total'])
            if license == "Total":
                latex_table += "\\midrule\n"
            latex_table += "\\textsc{" + license + "} & " + " & ".join(lic_vals) + " & " + total_vals + " \\\\\n"

        # Adding calculations for percentages
        mod_label = modality.lower() + "_license_terms_breakdown"
        unspec_or_open = counts_dict["Commercial"].get("Unrestricted", 0) + counts_dict["Commercial"].get("Unspecified", 0) + counts_dict["Unspecified"].get("Unrestricted", 0) + counts_dict["Unspecified"].get("Unspecified", 0)
        closed_pct = round(100 - unspec_or_open, 1)
        total_nc_license = round(counts_dict["NC/Acad"]["Total"], 1)
        total_restrictive_terms = round(counts_dict["Total"]["Source Closed"] + counts_dict["Total"]["Model Closed"], 1)
        caption = "\\textbf{A breakdown of " + modality + " Dataset licenses, and the Terms attached to their sources.} "
        caption += f"Among {modality} datasets, {total_nc_license}\% are Non-Commercially licensed, and {total_restrictive_terms}\% have restrictive terms, but a full {closed_pct}\% of datasets have either a restrictive license or terms."
        latex_table += "\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\caption{" + caption + "}\n\\label{tab:" + mod_label + "}\n\\end{table*}\n"

        latex_outputs[modality] = latex_table

    return latex_outputs


# Function to categorize tasks based on their domain and modality
def categorize_tasks(df, order, domain_typemap, tasks_column, modality, collections_datasets_flag):
    def map_taskgroup(row, modality) -> str:
        task = row[tasks_column]

        # If the modality is "Text" and the task is in the domain_typemap, return the mapped domain
        if modality == "Text":
            if task in domain_typemap:
                return domain_typemap[task]
            if not pd.isna(task):
                logging.warning("Could not find domain for %s" % task)
            return "Other"
        # If the modality is "Speech", return the task without domain mapping
        elif modality == "Speech":
            return task
        elif modality == "Video":
            if "misc" in task:
                return "Other"
            return task
        else:
            logging.warning("Unsupported modality for task categorization: %s" % modality)
            return "Other"

    # Define a mapping function for text tasks
    task_categories_mapper_text = {
        'Code': 'Code',
        'Translation': 'Translation',
        'Summarization': 'Summarization',
        'Response Ranking': 'Resp. Ranking',
        'Bias & Toxicicity Detection': 'Bias Detection',
        'Question Answering': 'Q&A',
        'Dialog Generation': 'Generation',
        'Miscellaneous': 'Other',
        'Short Text Generation': 'Generation',
        'Open-form Text Generation': 'Generation',
        'Brainstorming': 'Creativity',
        'Creative Writing': 'Creativity',
        'Creativity': 'Creativity',
        'Explanation': 'Reasoning',
        'Commonsense Reasoning': 'Reasoning',
        'Logical and Mathematical Reasoning': 'Reasoning',
        'Chain-of-Thought': 'Reasoning',
        'Natural Language Inference': 'Classification',
        'Text Classification': 'Classification',
        'Sequence Tagging': 'Classification',
        'Language Style Analysis': 'Classification'
    }

    # Define a mapping function for speech tasks
    task_categories_mapper_speech = {
        'Text to Speech': 'Text-To-Speech',
        'Text-To-Speech': 'Text-To-Speech',
        'Translation and Retrieval': 'Translation',
        'Speech Translation': 'Translation',
        'Machine Translation': 'Translation',
        'Speaker Identification': 'Speaker ID',
        'Speaker Identification (mono/multi)': 'Speaker ID',
        'Speaker Recognition': 'Speaker ID',
        'Speaker Verification': 'Speaker ID',
        'Speaker Diarization': 'Speaker ID',
        'Speech Recognition/Translation': 'Translation',
        'Speech Language Identification': 'Language ID',
        'Language Identification': 'Language ID',
        'Bias in Speech Recognition (Accents)': 'Bias Detection',
        'Speech Synthesis': 'Speech Synthesis',
        'Query By Example': 'Query by Ex',
        'Keyword Spotting': 'Keyword Spotting',
        'Speech Recognition': 'Other',  # other will be filtered out in the end
    }

    task_categories_mapper_video = {
        'Video Classification': 'Classification',
        'Video Captioning': 'Captioning',
        'Video Summarization': 'Summarization',
        'Video Q&A': 'Q&A',
        'Temporal Action Localization': 'Localization',
        'Group Activity Recognition': 'Action Recognition',
        'Video Segmentation': 'Segmentation',
        'Action Segmentation': 'Segmentation',
        'Action Localization': 'Localization',
        'Pose Estimation': 'Pose Estimation',
        'Temporal Action Segmentation': 'Segmentation',
        'Interaction understanding via ordering': 'Action Recognition',
        'Video Question Answering': 'Q&A',
        'Temporal Action Detection': 'Action Detection',
        'Action Recognition': 'Action Recognition',
        'Visual Interaction Prediction': 'Action Recognition',
        'Temporal Localization': 'Localization',
        'Spatial-Temporal Action Localization': 'Localization',
        'Video Object Detection': 'Object Detection',
        'Other': 'Other',
    }


    # Choose the appropriate mapping function based on the modality and collections_datasets_flag (needs to be updated if the flag is updated)

    task_categories_mapper = None
    if modality == "Text":
        task_categories_mapper = task_categories_mapper_text
    elif modality == "Speech":
        task_categories_mapper = task_categories_mapper_speech
    elif modality == "Video":
        task_categories_mapper = task_categories_mapper_video

    # Explode the tasks column to have one row per task
    if modality == "Text":
        if collections_datasets_flag == 'Datasets':
            df_tasks = df.explode(tasks_column)
        elif collections_datasets_flag == 'Collections':
            df_agg = df.groupby('Collection')[tasks_column].apply(lambda x: list(set.union(*map(set, x)))).reset_index()
            df_tasks = df.merge(df_agg, on='Collection', suffixes=('', '_agg'))
            df_tasks[tasks_column] = df_tasks[f'{tasks_column}_agg']
            df_tasks.drop(columns=[f'{tasks_column}_agg'], inplace=True)
            df_tasks = df_tasks.explode(tasks_column)
    else:
        df_tasks = df.explode(tasks_column)

    # Apply the mapping function to each row
    df_tasks[tasks_column] = df_tasks.apply(lambda row: map_taskgroup(row, modality), axis=1).fillna("Other")

    # Sort the tasks in the desired order
    df_tasks = df_tasks.sort_values(by=tasks_column)

    # Map the tasks to their corresponding categories using the chosen mapping function
    df_tasks[tasks_column] = df_tasks[tasks_column].apply(lambda x: task_categories_mapper.get(x, "Other"))

    # Filter out the 'Other' tasks
    df_tasks = df_tasks[df_tasks[tasks_column] != 'Other']

    # Add a new row for the 'Recognition' task in the Speech modality as every dataset should have a 'Recognition' task by default
    if modality == 'Speech':
        new_rows = pd.DataFrame({tasks_column: ['Recognition'] * len(df_tasks), 'Recognition': range(len(df_tasks) + 1, len(df_tasks) + len(df_tasks) + 1)})
        df_tasks = pd.concat([df_tasks, new_rows], ignore_index=True)

    return df_tasks


# Function to plot the conceptual bar chart
def concatenate_task_charts(chart1, chart2, chart3, font_size):
    combined_chart = alt.hconcat(chart1, chart2, chart3).configure_axis(
        grid=False,  # remove grid lines
        domain=False
    ).configure_view(
        strokeOpacity=0
    ).configure_title(
        fontSize=font_size+2,  # increase font size for the title by 2
        anchor='start'
    ).configure_legend(
        titleFontSize=font_size+2,  # increase font size for the legend title by 2
        labelFontSize=font_size,
        symbolSize=100
    )

    return combined_chart


# Function to plot the tasks chart
def plot_tasks_chart(
    df, task_typemap, order, pwidth, pheight, save_dir, font_size, modality, tasks_column, collections_datasets_flag
):
    # Filter the dataframe to only include the specified modality
    df = df[df['Modality'] == modality]

    # Categorize the tasks into their respective groups
    df_sources = categorize_tasks(df, order, task_typemap, tasks_column, modality, collections_datasets_flag)

    # Filter out the tasks that are not having a category
    df_sources = df_sources[df_sources[tasks_column].notnull()]

    # Prepare the aggregated counts for each task
    df_tasks_aggregated = df_sources[tasks_column].value_counts().reset_index()
    df_tasks_aggregated.columns = [tasks_column, 'count']

    # Prepare the aggregated counts for each dataset
    df_datasets_aggregated = df_sources['Dataset Name'].value_counts().reset_index()
    df_datasets_aggregated.columns = ['Dataset Name', 'count']

    # Calculate percentage for each task and dataset in the aggregated dataframe using the count of datasets as denominator
    df_tasks_aggregated['percentage'] = ((df_tasks_aggregated['count'] / df_datasets_aggregated['count'].sum()) * 100).round().astype(int)

    # Sort the tasks based on the count
    sorted_order = df_tasks_aggregated.sort_values('count', ascending=False)[tasks_column].tolist()

    # Assign a position to each task based on the sorted order
    df_tasks_aggregated['position'] = df_tasks_aggregated[tasks_column].apply(lambda x: sorted_order.index(x))

    # Assign a color to each task based on the position
    color_scale = alt.Scale(domain=list(range(len(sorted_order))), scheme='tableau20')

    bar_chart = alt.Chart(df_tasks_aggregated).mark_bar(size=12).encode(
        y=alt.Y(
            field=f"{tasks_column}",
            type="nominal",
            title=None,  # remove axis title
            sort=sorted_order,
            axis=alt.Axis(labelFontSize=font_size, titleFontSize=font_size+2)
        ),
        x=alt.X(
            field="percentage",
            type="quantitative",
            title=None,  # remove axis title
            axis=alt.Axis(format='.0f', labelExpr="datum.value + '%'", labelFontSize=font_size, titleFontSize=font_size+2)  # format percentage
        ),
        color=alt.Color(
            field="position",
            type="ordinal",
            title=None,  # remove legend title
            scale=color_scale,
            legend=None  # remove legend
        ),
        tooltip=[
            alt.Tooltip(f"{tasks_column}", title=f"{tasks_column}"),
            alt.Tooltip('count', title='Count'),
            alt.Tooltip('percentage', title='Percentage', format='.2f')
        ]
    ).properties(
        title={
            "text": [modality],
            "align": "center",  # center the title
            "anchor": "middle"  # anchor the title in the middle
        },
        width=pwidth,  # set width
        height=pheight  # set height
    )

    return bar_chart

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

def chart_creation(
    df: pd.DataFrame, max_count: int, x_field: str, labels: list, ratio: float, title: str, width: int, height: int, color: str):
    chart1 = alt.Chart(df).mark_bar(color=color).encode(
        # y=alt.Y(f'{x_field}:N', sort=labels, axis=alt.Axis(labelAngle=45)),
        # , domain=False, ticks=False, grid=False
        y=alt.Y(f'{x_field}:N', sort=labels, axis=alt.Axis(title=None, grid=False, domain=False, ticks=False)),
        x=alt.X('Number of datasets:Q', scale=alt.Scale(domain=[0, max_count * ratio]), axis=alt.Axis(grid=False, domain=False, tickCount=5)),
        tooltip=[x_field, 'Number of datasets']
    ).properties(
        title=title,
        width=width,
        height=height
    )
    return chart1

def combined_dim_charts(*charts):
    # Concatenate the two charts horizontally with different scales for the y-axes
    combined_chart = alt.hconcat(*charts, spacing=1).resolve_scale(
        color="independent",  # Ensures independent color scale for each chart
        shape="independent",  # Ensures independent shape scale for each chart (if applicable)
        size="independent"    # Ensures independent size scale for each chart (if applicable)
    ).resolve_legend(
        color="independent",  # Ensures each chart has an independent legend for color
        shape="independent",  # Ensures each chart has an independent legend for shape (if applicable)
        size="independent"    # Ensures each chart has an independent legend for size (if applicable)
    ).configure(
        title={"font": "Times New Roman", "fontSize": 16},
        axis={
            "labelFont": "Times New Roman",
            "titleFont": "Times New Roman",
            "labelFontSize": 14,
            "titleFontSize": 14,
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


def plot_temporal_cumulative_sources(
    df, modality, top_n, cumulative_measurement, earliest_year=2015, plotw=400, ploth=200,
    save_png=False
):

    def prep_df(df, modality, top_n, cumulative_measurement, earliest_year=2015):
        df_mod = df[df["Modality"] == modality]
        df_modsourceyears = df_mod.explode("Source Category")
        df_modsourceyears = reduce_categories_to_topk(df_modsourceyears, "Source Category", top_n)
        df_modsourceyears['Source Category'] = df_modsourceyears['Source Category'].apply(lambda x: x.title())
        # return df_modsourceyears

        source_cat_mapper = {
            "Crowdsourced": "Human Partic.",
            "Human": "Human Partic.",
            "Human Participants": "Human Partic.",
            "Getty-Images": "Getty Images",
            "Entertainment": "General Web",
            "News": "News Sites",
            "Encyclopedias": "Encyclopedias",
            "Governments": "Government",
            "Undisclosed Web": "General Web",
            "Calling Platform": "Calling Plat.",
            "Videoblocks": "VideoBlocks",
            "Tv": "TV",
            "Ml Datasets": "Other",
            "Others": "Other",
            "Other": "Other",
            "Unclear": "Other",
        }

        df_modsourceyears['Source Category'].replace(source_cat_mapper, inplace=True)
        if earliest_year > 2013:
            rep_map = {str(yr): f"<{earliest_year}" for yr in range(2013, earliest_year)}
            rep_map.update({"<2013": f"<{earliest_year}"})
            df_modsourceyears['Year Released Category'].replace(rep_map, inplace=True)

        df_modsourceyears = df_modsourceyears[df_modsourceyears['Year Released Category'] != "Unknown"]
        df_modsourceyears['Year Released Category'] = df_modsourceyears['Year Released Category'].cat.remove_unused_categories()

        df_modsourcecumulativeyears = df_modsourceyears.groupby(
            ["Year Released Category", "Source Category"]
        )[cumulative_measurement].sum().groupby(
            "Source Category"
        ).cumsum().reset_index(name="Cumulative Hours")

        # return df_modsourcecumulativeyears

        df_modsourcecumulativeyears = df_modsourcecumulativeyears.sort_values(by="Year Released Category")
        # Assuming your dataframe is named df
        df_modsourcecumulativeyears = df_modsourcecumulativeyears[df_modsourcecumulativeyears['Source Category'] != "Other"]
        # print(df_modsourcecumulativeyears)
        return df_modsourcecumulativeyears[["Year Released Category", "Source Category", "Cumulative Hours"]]
    df_cumyears = prep_df(df, modality, top_n, cumulative_measurement, earliest_year=earliest_year)
    # return df_cumyears

    df_final_values = df_cumyears.groupby('Source Category').apply(lambda x: x.loc[x['Year Released Category'].idxmax()])
    df_final_values = df_final_values[['Source Category', 'Cumulative Hours']].sort_values('Cumulative Hours', ascending=False)
    # Use the sorted categories for the legend order
    sorted_categories = df_final_values['Source Category'].tolist()
    # print(sorted_categories)
    # print(df_cumyears)

    YEARS_ORDER = [f"<{earliest_year}"] + [str(year) for year in range(earliest_year, 2025)]
    if modality in ["Speech", "Video"]:
        domain_max = 1000000
        yaxis_vals = [0, 1, 1000, 10000, 100000, 1000000]
        symlog_constant = 1000
    else:
        domain_max = 1e13
        yaxis_vals = [1e7, 1e9, 1e10, 1e11, 1e12, 1e13]
        symlog_constant = 1e7
    chart_sourceyearhours = alt.Chart(
        df_cumyears
    ).mark_line().encode(
        x=alt.X(
            "Year Released Category:N",
            title="",
            sort=YEARS_ORDER,
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "Cumulative Hours:Q",
            title="",
            scale=alt.Scale(
                type="symlog",
                constant=symlog_constant,
                domain=[1, domain_max]
            ),
            axis=alt.Axis(
                values=yaxis_vals,
                labelExpr="datum.value >= 1000000000000 ? datum.value / 1000000000000 + 'T' : datum.value >= 1000000000 ? datum.value / 1000000000 + 'B' : datum.value >= 1000000 ? datum.value / 1000000 + 'M' : datum.value >= 1000 ? datum.value / 1000 + 'K' : datum.value"
                # labelExpr="datum.value >= 1000000 ? datum.value / 1000000 + 'M' : datum.value >= 1000 ? datum.value / 1000 + 'K' : datum.value"
            )
        ),
    color=alt.Color(
        "Source Category:N",
        title="Source Category",
        legend=alt.Legend(
            orient="bottom",
            labelFontSize=15,  # Adjust the font size for the legend
            columns=3  # Wrap every K entries (replace K with the number of entries per row)
        ),
        sort=sorted_categories  # Sort legend by the final value
    )
    ).properties(
        width=plotw,
        height=ploth,
        title=f"{modality}"
    )

    if save_png:
        chart_sourceyearhours.save(
            os.path.join(PLOT_DIR, f"{modality}_sourcecategories-cumulativehours.png"),
            ppi=PLOT_PPI
        )
    return chart_sourceyearhours


def extract_lang_fam_mappers():

    lang_codes_to_families = {}
    lang_codes_to_names = {}
    iso_codes_to_langs = {}
    lang_id_to_iso_codes = {}
    lang_families = {}

    df_langsglottolog = pd.read_csv("data/speech_supporting_data/languages.csv")
    for i, row in df_langsglottolog.iterrows():
        lang_id = row["ID"]

        iso_code = row["Closest_ISO369P3code"]

        iso_codes_to_langs[iso_code] = lang_id
        lang_id_to_iso_codes[lang_id] = iso_code
        lang_codes_to_names[lang_id] = row["Name"]

        if row["Level"] == "family":
            lang_families[lang_id] = row["Name"]

        if pd.isna(iso_code):
            continue

        family_id = row["Family_ID"]
        if pd.isna(family_id):
            continue

        lang_codes_to_families[lang_id] = family_id
    return lang_codes_to_families, lang_codes_to_names, iso_codes_to_langs, lang_families, lang_id_to_iso_codes

def get_langfamily(
    lang: str,
    lang_fam_infos,
    code_langs,
) -> str:
    lang_codes_to_families, lang_codes_to_names, iso_codes_to_langs, lang_families, lang_id_to_iso_codes = lang_fam_infos
    # if not isinstance(lang, str):
    #     print(lang)
    lang = lang.split("-")[0].split("_")[0]
    try:
        if lang in code_langs:
            return "Code"
        # Need to iteratively seek to the top level
        lang = langcodes.get(langcodes.standardize_tag(lang, macro=True)).to_alpha3()
        lang = iso_codes_to_langs.get(lang, lang)

        while lang in lang_codes_to_families:
            lang = lang_codes_to_families[lang]

        lang = lang_families[lang]

    except:
        lang = "Other"

    return lang

# Creating a dictionary with the given languages mapped to their ISO 369-3 codes
language_iso_additional_mapping = {
    "Greek": "ell",
    "Northern Sotho": "nso",
    "Jingpho": "kac",
    "Pashto": "pus",
    "Sardinian": "srd",
    "Interlingue": "ile",
    'Arabic': 'arb',
    'Chinese': 'zho',
    'Hebrew': 'heb',
    'Persian': 'fas',
    'Indonesian': 'ind',
    'Portugese (Brazilian)': 'por',
    'Azerbaijani': 'aze',
    'Kyrgyz': 'kir',
    'Oromo': 'orm',
    'Nigerian Pidgin English': 'pcm',
    'Punjabi': 'pan',
    'Gaelic': 'gla',
    'Serbian': 'srp',
    'Serbian (Latin script)': 'srp',
    'Sinhalese': 'sin',
    'Kiswahili': 'swa',
    'Flemish': 'nld',
    'Croatian': 'hrv',
    'Bosnian': 'bos',
    'Guarani': 'grn',
    'Armenian': 'hye',
    'Interlingua': 'ina',
    'Ilocano': 'ilo',
    'Khmer': 'khm',
    'Luxembourgish': 'ltz',
    'Malay': 'msa',
    'Quechua': 'que',
    'Serbo-Croatian': 'hbs',
    'Yiddish': 'yid',
    'Farsi': 'fas',
    'Acehnese (Arabic script)': 'ace',
    'Acehnese (Latin script)': 'ace',
    'Mesopotamian Arabic': 'acm',
    'Ta’izzi-Adeni Arabic': 'acq',
    'North Levantine Arabic': 'apc',
    'Modern Standard Arabic': 'arb',
    'Modern Standard Arabic (Romanized)': 'arb',
    'Bemba': 'bem',
    'Banjar (Arabic script)': 'bjn',
    'Banjar (Latin script)': 'bjn',
    'Standard Tibetan': 'bod',
    'Nigerian Fulfulde': 'fuv',
    'Haitian Creole': 'hat',
    'Kamba': 'kam',
    'Kashmiri (Arabic script)': 'kas',
    'Kashmiri (Devanagari script)': 'kas',
    'Central Kanuri (Arabic script)': 'knc',
    'Central Kanuri (Latin script)': 'knc',
    'Kabiyè': 'kbp',
    'Kikongo': 'kon',
    'Limburgish': 'lim',
    'Lingala': 'lin',
    'Latgalian': 'ltg',
    'Luba-Kasai': 'lua',
    'Luo': 'luo',
    'Minangkabau (Arabic script)': 'min',
    'Minangkabau (Latin script)': 'min',
    'Meitei (Bengali script)': 'mni',
    'Western Persian': 'pes',
    'Tosk Albanian': 'als',
    'Tamasheq (Latin script)': 'taq',
    'Tamasheq (Tifinagh script)': 'taq',
    'Central Atlas Tamazight': 'tzm',
    'Uyghur': 'uig',
    'Chinese (Simplified)': 'zho',
    'Chinese (Traditional)': 'zho',
    'Iranian Persian': 'fas',
    'Panjabi': 'pan',
    'Pashto (Southern)': 'pbt',
    'Albanian (Tosk)': 'als',
    'Dayak': 'day',
    'Chinese (Hong Kong)': 'zho',
    'Divehi': 'div',
    'Hmong': 'hmn',
    'Hassaniya Arabic': 'mey',
    'Malagasy': 'mlg',
    'Myanmar': 'mya',
    'Northern Ndebele': 'nde',
    'Chichewa': 'nya',
    'Shilha': 'shi',
    'Tonga': 'ton',
    'Zhuang': 'zha'
}


def get_hours_for_dataset_and_language(row: pd.Series) -> float:
    df_yodashours = pd.read_csv("data/speech_supporting_data/yodas_splithours.csv").rename({"hours": "Hours"}, axis=1)
    df_yodashours["Language (ISO)"] = df_yodashours["split"].map(lambda x : x[:2])
    df_yodashours["Language (Name)"] = df_yodashours["Language (ISO)"].map(
        lambda x: langcodes.Language.make(language=langcodes.standardize_tag(x, macro=True)).language_name()
    )
    df_commonvoicehours = pd.read_json("data/speech_supporting_data/commonvoice_splithours.json").T

    language_codes_to_aggregate = {}
    for langcode in df_commonvoicehours.index:
        if "-" in langcode or "_" in langcode:
            langcode_simplified = langcode.split("-")[0].split("_")[0]
            # print("Will aggregate language %s to %s" % (langcode, langcode_simplified))
            language_codes_to_aggregate.setdefault(langcode_simplified, [])
            language_codes_to_aggregate[langcode_simplified].append(langcode)

    for langcode_simplified, langcode_data_to_aggregate in language_codes_to_aggregate.items():
        df_commonvoicehours.loc[langcode_simplified, "total_clips_duration"] = df_commonvoicehours.loc[langcode_data_to_aggregate, "total_clips_duration"].sum()


    df_commonvoicehours["Hours"] = df_commonvoicehours["total_clips_duration"] / 60 / 60 / 1000
    df_commonvoicehours = df_commonvoicehours.rename(columns={'Languages (ISO)': 'Language (ISO)'})
    df_commonvoicehours = df_commonvoicehours.reset_index(names=["Language (ISO)"])

    df_multilinguallibrispeechhours = pd.read_csv("data/speech_supporting_data/multilinguallibrispeech_splithours.csv")
    df_multilinguallibrispeechhours["Hours"] = df_multilinguallibrispeechhours[df_multilinguallibrispeechhours.columns[1:]].sum(axis=1)
    df_multilinguallibrispeechhours["Language (ISO)"] = df_multilinguallibrispeechhours.language.map(lambda x: langcodes.find(x).to_tag())

    df_bloomspeechhours = pd.read_csv("data/speech_supporting_data/bloomspeech_splithours.csv")
    df_bloomspeechhours["Hours"] = df_bloomspeechhours[df_bloomspeechhours.columns[2:]].sum(axis=1) / 60
    df_bloomspeechhours["Language (ISO)"] = df_bloomspeechhours["ISO-639-3"].map(lambda x: langcodes.standardize_tag(x, macro=True))

    df_fleursspeechhours = pd.read_csv("data/speech_supporting_data/fleurs_splithours.csv")
    df_fleursspeechhours = df_fleursspeechhours.rename(columns={'Languages (ISO)': 'Language (ISO)'})

    dataset_hoursmapping = {
        "yodas": df_yodashours,
        "common-voice-corpus-170": df_commonvoicehours,
        "multilingual-librispeech": df_multilinguallibrispeechhours,
        "bloom-speech": df_bloomspeechhours,
        "fleurs": df_fleursspeechhours
    }

    special_cases = {
        "yodas": {
            "sr-Latn": "sr", # YODAS metadata doesn't specify the script
            "he": "iw" # YODAS appears to use the old ISO639-1 code
        },
        "fleurs": {
            "no": "nb" # FLEURS specifies locale code
        }
    }

    dataset = row["Unique Dataset Identifier"]
    language = row["Language (ISO)"]
    language = langcodes.standardize_tag(language)

    if dataset in dataset_hoursmapping:
        if language in special_cases.get(dataset, {}):
            language = special_cases[dataset][language]

        hours_df = dataset_hoursmapping[dataset]
        hours = hours_df[hours_df["Language (ISO)"] == language]["Hours"].sum()
        if hours == 0:
            print("Hours not found for language code %s in dataset %s" % (language, dataset))
        return hours

    return row["Hours"]



def plot_temporal_ginis(df_gini, df_spec, domain_cats, columns):
    FONT_SIZE = 16
    EARLIEST_YEAR = 2013
    YEARS_ORDER = [f"<{EARLIEST_YEAR}"] + [str(year) for year in range(EARLIEST_YEAR, 2025)]

    df_gini = df_gini.sort_values(by="Year Released Category")

    # print(df_gini)
    domains = df_gini["Type"].unique()
    df_gini["Modality"] = df_gini["Type"]
    y_axis_min, y_axis_max = 0, 1
    chart_meanlangf = alt.Chart(
        df_gini
    ).mark_line().encode(
        x=alt.X(
            "Year Released Category:N",
            title="",
            sort=YEARS_ORDER,
            # axis=alt.Axis(labelAngle=-30),
            axis=alt.Axis(labelAngle=0),
            scale=alt.Scale(padding=0)
        ),
        y=alt.Y(
            "Gini Mean:Q",
            title="Gini (Cumulative)",
            # scale=alt.Scale(zero=False)
            scale=alt.Scale(zero=False, domain=[y_axis_min, y_axis_max])
        ),
        color=alt.Color(
            "Type:N",
            title="Modality",
            scale=alt.Scale(
                domain=domains,
                # range=color_range,
                # domain=["Text Language (ISO)", "Text Language Family", "Speech Language (ISO)", "Speech Language Family"],
                # range=["#82b5cf", "#ff7fde"]
            )
        )
    )

    chart_meanpointslangf = alt.Chart(
        df_gini
    ).mark_point().encode(
        x=alt.X(
            "Year Released Category:N",
            title="",
            sort=YEARS_ORDER,
            axis=alt.Axis(labelAngle=0),
            # axis=alt.Axis(labelAngle=-30),
            scale=alt.Scale(padding=0)
        ),
        y=alt.Y(
            "Gini Mean:Q",
            title="Gini (Cumulative)",
            # scale=alt.Scale(zero=False),
            scale=alt.Scale(zero=False, domain=[y_axis_min, y_axis_max])
        ),
        color="Type:N"
    )

    chart_cislangf = alt.Chart(
        df_gini
    ).mark_area(
        opacity=0.25
    ).encode(
        x=alt.X(
            "Year Released Category:N",
            title="",
            sort=YEARS_ORDER,
            axis=alt.Axis(labelAngle=0)
            # axis=alt.Axis(labelAngle=-30)
        ),
        y="Gini Lower:Q",
        y2="Gini Upper:Q",
        color="Type:N"
    )

    chart_langf = (chart_cislangf + chart_meanlangf + chart_meanpointslangf).configure_axis(
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE,
        grid=False
    ).configure_legend(
        orient="bottom",
        direction="horizontal",
        padding=15,
        # opacity=1.0,
        # cornerRadius=5,
        # fillColor="white",
        # strokeColor="lightgray",
        offset=-100,
        # legendX=-100,
        # legendY=-50,
        columns=columns,
        titleFontSize=15,
        labelFontSize=15,
    ).properties(
        width=500,
        height=200,
    )
    modalities = df_spec["Modality"].unique()
    for modality in modalities:
        for domain in domain_cats:
            print(f"{modality} | {domain} | {df_spec[df_spec['Modality'] == modality][domain].nunique()}")

    return chart_langf

def compute_temporal_gini_bounds(df_spec, measure_key, cumulation_key):
    # Get the cumulative hours by language over time
    df_spec = df_spec[df_spec['Year Released Category'] != "Unknown"]
    df_spec['Year Released Category'] = df_spec['Year Released Category'].cat.remove_unused_categories()

    df_spec_cum = df_spec.groupby(
        ["Year Released Category", measure_key]
    )[cumulation_key].sum().groupby(
        measure_key
    ).cumsum().reset_index(name="Cumulative Hours")
    df_spec_cum = df_spec_cum[df_spec_cum['Year Released Category'] != "Unknown"]

    # Calculate Gini coefficient and CIs for cumulative hours by language
    df_spec_cum_gini = df_spec_cum.groupby("Year Released Category")["Cumulative Hours"]
    # print(df_spec_cum_gini.groups)
    df_spec_cum_gini = df_spec_cum_gini.apply(
        lambda x: bootstrap_cis_for_gini(x.values)
    ).reset_index(
        name="Gini"
    )
    df_spec_cum_gini = df_spec_cum_gini[df_spec_cum_gini['Year Released Category'] != "Unknown"]

    df_spec_cum_gini["Gini Mean"] = df_spec_cum_gini["Gini"].map(lambda x: x[0])
    df_spec_cum_gini["Gini Lower"] = df_spec_cum_gini["Gini"].map(lambda x: max(0, x[1]))
    df_spec_cum_gini["Gini Upper"] = df_spec_cum_gini["Gini"].map(lambda x: min(x[2], 1))
    df_spec_cum_gini["Type"] = measure_key
    return df_spec_cum_gini


def prepare_speech_for_gini(df):
    df_speech = df[df["Modality"] == "Speech"]
    df_speech = df_speech.rename(columns={'Languages (ISO)': 'Language (ISO)'})
    df_speechlanguagenames = df_speech.explode("Language (ISO)")
    df_speechlanguagenames["Language (Name)"] = df_speechlanguagenames["Language (ISO)"].map(
        lambda x: langcodes.Language.make(language=langcodes.standardize_tag(x.split("-")[0].split("_")[0], macro=True)).language_name()
    )
    df_speechlanguagesn = df_speechlanguagenames.copy()

    df_speechlanguagesn["Language (ISO)"] = df_speechlanguagesn["Language (ISO)"].map(lambda x : x.split("_")[0].split("-")[0])
    lang_fam_infos = extract_lang_fam_mappers()
    df_speechlanguagesn["Language Family"] = df_speechlanguagesn["Language (ISO)"].map(lambda x: get_langfamily(x, lang_fam_infos, []))

    # Subdivide hours evenly across the languages given in each dataset
    df_speechlanguagesn["Hours"] = df_speechlanguagesn.groupby(["Unique Dataset Identifier", "Language (ISO)"])["Hours"].transform(
        lambda x: x / x.count()
    )
    df_speechlanguagesn["Hours"] = df_speechlanguagesn.apply(get_hours_for_dataset_and_language, axis=1)

    df_speechlanguagesn = df_speechlanguagesn.sort_values(by="Year Released Category")

    # Ensure that, for each of those datasets, we have heterogenous language hours
    for dataset in ["yodas", "common-voice-corpus-170", "multilingual-librispeech", "bloom-speech", "fleurs"]:
        assert df_speechlanguagesn[
                df_speechlanguagesn["Unique Dataset Identifier"] == dataset
            ]["Hours"].nunique() > 1, "Dataset %s has homogenous language hours" % dataset

    # # Gini coefficient for hours across languages
    # speechlanguages_totalhours = df_speechlanguagesn.groupby("Languages (ISO)")["Hours"].sum().reset_index(name="Total Hours")

    # multimodal_util.gini(speechlanguages_totalhours["Total Hours"].values)

    # # Gini coefficient for hours across language-families
    # speechlanguagesf_totalhours = df_speechlanguagesn.groupby("Language Family")["Hours"].sum().reset_index(name="Total Hours")

    # multimodal_util.gini(speechlanguagesf_totalhours["Total Hours"].values)
    speech_df_spec_cum_gini_langs = compute_temporal_gini_bounds(df_speechlanguagesn, "Language (ISO)", "Hours")
    speech_df_spec_cum_gini_langfams = compute_temporal_gini_bounds(df_speechlanguagesn, "Language Family", "Hours")
    df_speechlanguagesfamilycumulativehoursgini = pd.concat(
        [speech_df_spec_cum_gini_langs, speech_df_spec_cum_gini_langfams]
    )
    return df_speechlanguagesfamilycumulativehoursgini, df_speechlanguagesn[["Year Released Category", "Hours", "Language (ISO)", "Language Family"]]


def prep_text_for_lang_gini(df, all_constants):
    LANG_GROUP_MAPPER = invert_dict_of_lists(all_constants["LANGUAGE_GROUPS"])
    df_text = df[df["Modality"] == "Text"] #[["Year Released", "Total Tokens", "Languages", "Language Families"]]

    # TODO: Undo this when we have all metrics in.
    df_text = df_text[df_text["Total Tokens"].notna()]

    df_text = df_text[df_text['Year Released Category'] != "Unknown"]
    df_text['Year Released Category'] = df_text['Year Released Category'].cat.remove_unused_categories()

    df_text_lang_explode = df_text.explode("Languages")
    df_text_lang_explode["Language Families"] = df_text_lang_explode["Languages"].map(lambda c: LANG_GROUP_MAPPER[c])
    # Subdivide tokens evenly across the languages given in each dataset
    df_text_lang_explode["Tokens"] = df_text_lang_explode.groupby(["Unique Dataset Identifier", "Languages"])["Total Tokens"].transform(
        lambda x: x / x.count()
    )
    df_text_lang_explode = df_text_lang_explode[["Collection", "Dataset Name", "Year Released Category", "Tokens", "Languages", "Language Families"]]

    code_languages = df_text_lang_explode[df_text_lang_explode["Language Families"] == "Code"]["Languages"].unique()

    lang_fam_infos = extract_lang_fam_mappers()
    lang_name_to_id = {v: k for k, v in lang_fam_infos[1].items()}
    lang_id_to_isocode = lang_fam_infos[4]

    lang_iso_mapper = {}
    for langname in df_text_lang_explode["Languages"].unique():
        if langname in code_languages:
            lang_iso_mapper[langname] = langname
        elif langname in language_iso_additional_mapping:
            lang_iso_mapper[langname] = language_iso_additional_mapping[langname]
        elif langname in lang_name_to_id:
            xx = lang_name_to_id[langname]
            lang_iso_mapper[langname] = lang_id_to_isocode[xx]
        else:
            print(f"{langname} not found")

    df_text_lang_explode['Language (ISO)'] = df_text_lang_explode['Languages'].map(lang_iso_mapper)
    # print(df_text_lang_explode["Language (ISO)"].value_counts())
    # print(all([isinstance(ll, str) for ll in df_text_lang_explode["Language (ISO)"].unique()]))
    df_text_lang_explode["Language Family"] = df_text_lang_explode["Language (ISO)"].map(lambda x: get_langfamily(x, lang_fam_infos, code_languages))

    # df_text = df_text.rename(columns={'Languages': 'Language (ISO)', "Language Families": "Language Family"})

    text_df_spec_cum_gini_langs = compute_temporal_gini_bounds(df_text_lang_explode, "Language (ISO)", "Tokens")
    text_df_spec_cum_gini_langfams = compute_temporal_gini_bounds(df_text_lang_explode, "Language Family", "Tokens")
    return pd.concat([text_df_spec_cum_gini_langs, text_df_spec_cum_gini_langfams]), df_text_lang_explode


def prepare_geo_gini_data(df):
    df_locs = df[df['Countries'].apply(lambda x: len(x) > 0)]
    # print(len(df_locs))
    df_loc_explode = df_locs.explode("Countries")
    # print(len(df_loc_explode))
    df_text_locs = df_loc_explode[df_loc_explode["Modality"] == "Text"]
    # print(len(df_text_locs))
    df_speech_locs = df_loc_explode[df_loc_explode["Modality"] == "Speech"]
    df_video_locs = df_loc_explode[df_loc_explode["Modality"] == "Video"]
    df_text_locs = df_text_locs[df_text_locs["Total Tokens"].notna()]
    # print(len(df_text_locs))
    df_text_locs["Dimension"] = df_text_locs["Total Tokens"]
    df_speech_locs["Dimension"] = df_speech_locs["Hours"]
    df_video_locs["Dimension"] = df_video_locs["Hours"]


    df_text_locs["Dimension"] = df_text_locs.groupby(["Unique Dataset Identifier", "Countries"])["Dimension"].transform(
        lambda x: x / x.count()
    )
    df_video_locs["Dimension"] = df_video_locs.groupby(["Unique Dataset Identifier", "Countries"])["Dimension"].transform(
        lambda x: x / x.count()
    )
    df_speech_locs["Dimension"] = df_speech_locs.groupby(["Unique Dataset Identifier", "Countries"])["Dimension"].transform(
        lambda x: x / x.count()
    )
    df_spec_locs = pd.concat([df_text_locs, df_speech_locs, df_video_locs])[["Unique Dataset Identifier", "Countries", "Dimension", "Modality", "Year Released Category"]]

    text_df_spec_cum_gini_locs = compute_temporal_gini_bounds(df_text_locs, "Countries", "Dimension")
    text_df_spec_cum_gini_locs["Type"] = "Text"
    speech_df_spec_cum_gini_locs = compute_temporal_gini_bounds(df_speech_locs, "Countries", "Dimension")
    speech_df_spec_cum_gini_locs["Type"] = "Speech"
    video_df_spec_cum_gini_locs = compute_temporal_gini_bounds(df_video_locs, "Countries", "Dimension")
    video_df_spec_cum_gini_locs["Type"] = "Video"
    df_gini_locs = pd.concat(
        [text_df_spec_cum_gini_locs, speech_df_spec_cum_gini_locs, video_df_spec_cum_gini_locs]
    )

    return df_gini_locs, df_spec_locs


def prepare_data_cum_barchart(df_spec_locs, target):
    EARLIEST_YEAR = 2013
    YEARS_ORDER = [f"<{EARLIEST_YEAR}"] + [str(year) for year in range(EARLIEST_YEAR, 2025)]

    # Ensure that 'Year Released' is treated as an ordered categorical variable
    df_spec_locs["Year Released Category"] = pd.Categorical(
        df_spec_locs["Year Released Category"],
        categories=YEARS_ORDER,
        ordered=True
    )

    unique_countries_per_modality_year = defaultdict(lambda: defaultdict(set))
    cumulative_count = []  # This will store the result
    # Step 3: Iterate over sorted data and calculate cumulative unique country count
    for i, entry in df_spec_locs.iterrows():
        modality = entry["Modality"]
        country = entry[target]
        year = entry["Year Released Category"]

        # Add the country to the set for the current modality
        unique_countries_per_modality_year[modality][year].add(country)

    df_spec_src = []
    for modality, year_countries in unique_countries_per_modality_year.items():
        observed = set()
        for year in YEARS_ORDER:
            observed = observed.union(year_countries[year])
            # Append the result to the cumulative_count list
            df_spec_src.append({
                "Modality": modality,
                "First Year Released": year,
                f"Total {target}": len(observed)
            })
    return pd.DataFrame(df_spec_src)


def plot_cum_barchart(df_spec_src, target, domains):
    colors = {
        "Text": "salmon",
        "Speech": "skyblue",
        "Video": "forestgreen",
    }
    EARLIEST_YEAR = 2013
    YEARS_ORDER = [f"<{EARLIEST_YEAR}"] + [str(year) for year in range(EARLIEST_YEAR, 2025)]
    color_scale = alt.Scale(
        domain=domains,  # Sorted as requested
        range=[colors[d] for d in domains]
    )

    chart = alt.Chart(df_spec_src).mark_bar(
        width=6  # Adjust this value to control bar width
    ).encode(
        x=alt.X(
            "First Year Released:N",
            title="",
            sort=YEARS_ORDER,
            axis=alt.Axis(labelAngle=0, domain=False, ticks=False, grid=False)
        ),
        y=alt.Y(
            f"Total {target}:Q",
            axis=alt.Axis(grid=False, domain=False),
            title=f"Total {target} Represented"
        ),
        xOffset=alt.XOffset("Modality:N", sort=domains),  # Sorted as requested
        color=alt.Color("Modality:N", scale=color_scale, title="Modality"),
        # opacity=alt.Opacity("Unique Dataset Identifier:N", legend=None, scale=alt.Scale(range=[0.6, 1.0])),
    )

    # Control the width and height of the figure
    chart = chart.configure_axis(
        labelFontSize=15,
        titleFontSize=15,
        grid=False
    ).properties(
        width=500,  # Adjust this value to control the width
        height=200  # Adjust this value to control the height
    ).configure_view(
        strokeWidth=0  # This removes the frame around the plot
    ).configure_legend(
        # columns=columns,
        orient='top-left',
        titleFontSize=15,
        labelFontSize=15,
    )


    # ).configure_legend(
    #     orient="bottom",
    #     direction="horizontal",
    #     padding=15,
    #     # opacity=1.0,
    #     # cornerRadius=5,
    #     # fillColor="white",
    #     # strokeColor="lightgray",
    #     offset=-100,
    #     # legendX=-100,
    #     # legendY=-50,
    #     columns=columns,
    #     titleFontSize=15,
    #     labelFontSize=15,
    # )

    return chart
