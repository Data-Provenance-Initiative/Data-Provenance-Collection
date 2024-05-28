import os
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency

from helpers import io
from . import analysis_constants


############################################################
###### Text Pretraining Analysis Functions
############################################################



def extract_url_annotations(dirpaths):
    """Convert raw annotations into url --> info dictionaries."""
    def extract_row_info(row):
        """
        Extract available information from a row of the DataFrame.
        Returns a dictionary with the available information.
        """
        row_info = {}
    
        # Extract information from columns if present
        for col in ["Website Issue", "User Content", "Terms of Use Link 1", "Terms of Use Link 2", "Terms of Use Link 3", 
            "Terms of Use Link 4", "Terms of Use Link 5", "Paywall", "Content Modalities: Text", "Content Modalities: Images", 
            "Content Modalities: Video", "Content Modalities: Audio", "Advertisements", "Website Issue User Content", "Website Description", 
            "Content Domain I", "Content Domain II", "Content Domain III", "Content Domain (other)", "Type of service", "Type of service (Other)",
            "Sensitive content: Nudity", "Sensitive content: Pornography", "Sensitive content: Drugs", "Sensitive content: Violence",
            "Sensitive content: Illegal Activities", "Sensitive content: Hate Speech"]:
            if col in row:
                row_info[col] = row[col]
    
        return row_info
    
    url_to_info = defaultdict(dict)
    url_to_issue = []
    all_fpaths = [fp for dirpath in dirpaths for fp in io.listdir_nohidden(dirpath)]
    for fpath in all_fpaths:
        df = pd.read_csv(fpath)
        if 'User Content' in df:
            df['User Content'] = df['User Content'].fillna('None')
        df = df.fillna("")
        overwrite_attempts = 0
        for _, row in df.iterrows():
            domain = row["Domain"]
            row_info = extract_row_info(row)
            if domain in url_to_issue or row_info["Website Issue"]:
                url_to_issue.append(domain)
                continue
            if domain in url_to_info:
                overwrite_attempts += 1
                url_to_info[domain].update(row_info)
            else:
                url_to_info[domain] = row_info
        # print(f"{fpath}: + {len(df)} = {len(url_to_info)} | overwritten: {overwrite_attempts}")
                
    print(f"{len(url_to_info)} rows before filtering.")
    # filter out incomplete rows:
    url_to_rows = {}
    issue_counter, unannotated_counter = 0, 0
    for url, infos in url_to_info.items():
        if infos["Website Issue"]:
            issue_counter += 1
            continue
        elif not infos.get('Website Description', "") or not infos.get("Paywall", ""):
            unannotated_counter += 1
            continue
        url_to_rows[url] = infos
    print(f"{len(url_to_rows)} rows after filtering. {issue_counter} issues, {unannotated_counter} unannotated.")
    return url_to_rows


def categorize_domain_annotations(url_to_info, cols, mapper):
    """Categorize domains or services annotations into their categories."""
    url_to_categories = defaultdict(list)
    other_vals, only_other = [], []
    for url, infos in url_to_info.items():
        if infos["Website Issue"] or not infos.get('User Content', ""):
            continue
        categories = set()
        for col in cols:
            if col in infos:
                val = infos[col].lower().strip()
                if val != "":
                    category = mapper.get(val, "Other")
                    if category == "Other":
                        other_vals.append(val)
                    categories.add(category)

        if len(categories) == 1 and "Other" in categories:
            only_other.append(val)
        url_to_categories[url] = sorted(list(categories))
    return url_to_categories, Counter(other_vals), Counter(only_other)


def process_url_annotations(url_to_info):
    """
    domains, services, paywall, advertisements, user_content, modalities, sensitive_content
    """

    all_domain_cols = ["Content Domain I", "Content Domain II","Content Domain III", "Content Domain (other)"]
    url_to_domains, other_domains, od2 = categorize_domain_annotations(url_to_info, all_domain_cols, analysis_constants.CONTENT_DOMAIN_INVERSE_MAPPING)
    # domain_counter = Counter(["-".join(vals) for vals in url_to_domains.values()])

    all_typ_cols = ["Type of service", "Type of service (Other)"]
    url_to_services, other_services, os2 = categorize_domain_annotations(url_to_info, all_typ_cols, analysis_constants.WEBSITE_SERVICE_INVERSE_MAPPING)
    # service_counter = Counter(["-".join(vals) for vals in url_to_services.values()])
    return od2

    def make_domains_services_compatible(domains, services):
        merge_fields = [
            ["News/Periodicals"],
            ["E-Commerce"],
            ["Blogs"],
            ["Academic"],
            ["Social Media/Forums"]
        ]
        for merge_list in merge_fields:
            for d in domains:
                if d in merge_list:
                    services.append(d)

            for s in services:
                if s in merge_list:
                    domains.append(s)

        return list(set(domains)), list(set(services))

    # Logic to propgate domains and services judgements to one another
    for url in url_to_domains:
        # If [ecommerce, blog, news, academic, social media/forums] for domains or services, then same for the other.
        url_to_domains[url], url_to_services[url] = make_domains_services_compatible(url_to_domains[url], url_to_services[url])

    # Compress domain categories to even fewer, high-level categories
    for url, ds in url_to_domains.items():
        url_to_domains[url] = list(set([analysis_constants.INVERSE_CONTENT_DOMAIN_CATEGORY_COMPRESSION[dd] for dd in ds]))

    sensitive_content_cols = [
        "Sensitive content: Nudity", "Sensitive content: Pornography", "Sensitive content: Drugs", "Sensitive content: Violence",
        "Sensitive content: Illegal Activities", "Sensitive content: Hate Speech"
    ]
    url_results = []
    for url, infos in url_to_info.items():
        url_results.append({
            "URL": url,
            "User Content": infos["User Content"] == "Weak Moderation",
            "Domains": url_to_domains[url],
            "Services": url_to_services[url],
            "Paywall": infos["Paywall"] != "No",
            "Ads": infos["Advertisements"],
            "Modality: Image": infos["Content Modalities: Images"] not in ["None", "", False],
            "Modality: Video": infos["Content Modalities: Video"] not in ["None", "", False], 
            "Modality: Audio": infos["Content Modalities: Audio"] not in ["None", "", False], 
            "Sensitive Content": any([infos[col] for col in sensitive_content_cols]),
        })
    return url_results
    # return url_results, Counter(other_domains), Counter(other_services)

def encode_size_columns(df, url_token_lookup):
    c4_url_to_counts = url_token_lookup.get_url_to_token_map("c4")
    rf_url_to_counts = url_token_lookup.get_url_to_token_map("rf")
    dolma_url_to_counts = url_token_lookup.get_url_to_token_map("dolma")
    random_10k_urls = url_token_lookup.get_10k_random_sample()
    
    count_df = pd.DataFrame(list(c4_url_to_counts.items()), columns=['URL', 'c4 tokens'])
    count_df = count_df.sort_values('c4 tokens', ascending=False)
    count_df['c4 rank'] = range(1, len(count_df) + 1)
    df = df.merge(count_df, on='URL', how='left')

    count_df = pd.DataFrame(list(rf_url_to_counts.items()), columns=['URL', 'rf tokens'])
    count_df = count_df.sort_values('rf tokens', ascending=False)
    count_df['rf rank'] = range(1, len(count_df) + 1)
    df = df.merge(count_df, on='URL', how='left')

    count_df = pd.DataFrame(list(dolma_url_to_counts.items()), columns=['URL', 'dolma tokens'])
    count_df = count_df.sort_values('dolma tokens', ascending=False)
    count_df['dolma rank'] = range(1, len(count_df) + 1)
    df = df.merge(count_df, on='URL', how='left')

    df['sample'] = df['URL'].apply(lambda x: 'random' if x in random_10k_urls else 'top')
    return df

def analyze_url_variable_correlations(df, top_n_list, corpus_key="c4"):
    """
    corpus_key: 'c4', 'rf' or 'dolma'
    """
    binary_vars = ['User Content', 'Paywall', 'Ads', 'Modality: Image', 'Modality: Video', 'Modality: Audio', 'Sensitive Content']
    domain_vars = df['Domains'].apply(pd.Series).columns
    service_vars = df['Services'].apply(pd.Series).columns

    # Expand the 'tags' lists into a DataFrame of True/False values
    domains_expanded = pd.get_dummies(df['Domains'].explode()).groupby(level=0).max()
    domains_expanded.columns = ['domain_' + col for col in domains_expanded.columns]
    services_expanded = pd.get_dummies(df['Services'].explode()).groupby(level=0).max()
    services_expanded.columns = ['services_' + col for col in services_expanded.columns]

    # Combine with the original DataFrame
    df = df.join(domains_expanded).join(services_expanded)
    # return df

    all_vars = binary_vars + list(domains_expanded.columns) + list(services_expanded.columns)

    # Sort the dataframe by 'c4 tokens' in descending order
    df = df.sort_values(by=f"{corpus_key} tokens", ascending=False)

    # Create an empty dataframe to store the results
    results_df = pd.DataFrame(index=all_vars, columns=[f"Top {n}" for n in top_n_list] + ['Random', 'Chi-Squared Stat', 'P-value'])

    def compute_percentages(df, cols):
        result = {}
        for col in cols:
            count = df[col].sum()
            total = len(df)
            result[col] = round(100 * count / total, 2)
        return result
    
    for top_n in top_n_list:
        top_n_df = df[df[f"{corpus_key} rank"] <= top_n]
        print(f"Num URLs in Top-{top_n}: {len(top_n_df)}")
        results_df[f'Top {top_n}'] = compute_percentages(top_n_df, all_vars)
    random_df = df[df["sample"] == "random"]
    print(f"Num URLs in random sample: {len(random_df)}")
    results_df['Random'] = compute_percentages(random_df, all_vars)

    # Perform Chi-Squared test and populate the results
    for var in all_vars:
        observed = pd.crosstab(random_df[var], pd.qcut(df[f"{corpus_key} tokens"], 4))
        chi2, p, dof, expected = chi2_contingency(observed)
        results_df.loc[var, 'Chi-Squared Stat'] = f"{chi2:.2f}"
        results_df.loc[var, 'P-value'] = f"{p:.2f}"

    return results_df

############################################################
###### Text Finetuning Analysis Functions
############################################################

def check_datasummary_in_constants(rows, all_constants):
    """Tests your data summary rows to see if all the values are in the constants.

    If not, it will print out the missing values, and which data collections they came from.
    """
    CREATOR_TO_COUNTRY = {v: k for k, vs in all_constants["CREATOR_COUNTRY_GROUPS"].items() for v in vs}
    CREATOR_TO_GROUP = {v: k for k, vs in all_constants["CREATOR_GROUPS"].items() for v in vs}
    TASK_TO_GROUP = {v: k for k, vs in all_constants["TASK_GROUPS"].items() for v in vs}
    LANG_TO_GROUP = {v: k for k, vs in all_constants["LANGUAGE_GROUPS"].items() for v in vs}
    SOURCE_TO_GROUP = {v: k for k, vs in all_constants["DOMAIN_GROUPS"].items() for v in vs}
    LICENSE_CLASSES = list(all_constants["LICENSE_CLASSES"].keys()) + list(all_constants["CUSTOM_LICENSE_CLASSES"].keys())

    def check_entities(collection_id, vals, const_map, miss_dict):
        for v in vals:
            if v not in const_map:
                miss_dict[v].add(collection_id)
                
    # Category --> missing entry --> list of Collection IDs where this comes from.
    missing_metadata = defaultdict(lambda: defaultdict(set))
    for row in rows:

        row_licenses = [lic["License URL"] if lic["License"] == "Custom" else lic["License"] for lic in row["Licenses"]]
        check_entities(row["Collection"], row_licenses, 
                       LICENSE_CLASSES, missing_metadata["License Classes"])

        check_entities(row["Collection"], row.get("Creators", []), 
                       CREATOR_TO_GROUP, missing_metadata["Creator Groups"])

        check_entities(row["Collection"], row.get("Creators", []), 
                       CREATOR_TO_COUNTRY, missing_metadata["Creator Countries"])

        check_entities(row["Collection"], row.get("Task Categories", []), 
                       TASK_TO_GROUP, missing_metadata["Task Categories"])

        check_entities(row["Collection"], row.get("Text Sources", []), 
                       SOURCE_TO_GROUP, missing_metadata["Text Sources"])

        check_entities(row["Collection"], row.get("Languages", []), 
                       LANG_TO_GROUP, missing_metadata["Languages"])
        
    for category, missing_info in missing_metadata.items():
        if len(missing_info) == 0:
            print(f"No missing info for {category}!")
            # print()
        else:
            print(f"There is metadata missing from the constants/ files for {category}:")
            print("Please check if you can modify the name of the entity (in data summary) to exactly match the entity as written in the constants files -- so we don't have multiple versions.")
            print("If it is not in the constants file in any form, then add it to the constants file.")
            print()
            for x, collections in missing_info.items():
                print(x + f"   |   Appears in: {collections}")
        print()


def extract_info(rows, all_constants):
    """Interpret the categories across all data summary rows.

    Returns:
        Dict: {dataset_uid --> {attribute --> value}}

        The value can be a list (e.g. tasks/sources/creators), float (e.g. num exs), or string (license class)
    """
    CREATOR_TO_COUNTRY = {v: k for k, vs in all_constants["CREATOR_COUNTRY_GROUPS"].items() for v in vs}
    CREATOR_TO_GROUP = {v: k for k, vs in all_constants["CREATOR_GROUPS"].items() for v in vs}
    TASK_TO_GROUP = {v: k for k, vs in all_constants["TASK_GROUPS"].items() for v in vs}
    LANG_TO_GROUP = {v: k for k, vs in all_constants["LANGUAGE_GROUPS"].items() for v in vs}
    SOURCE_TO_GROUP = {v: k for k, vs in all_constants["DOMAIN_GROUPS"].items() for v in vs}

    # {dataset_uid --> {attribute --> value}}
    dataset_infos = {}
    for row in rows:
        dataset_uid = row["Unique Dataset Identifier"]
        if dataset_uid not in dataset_infos:
            dataset_infos[dataset_uid] = {
                "Name": row["Dataset Name"],
                "Sources": set(),
                "Domains": set(),
                "Synthetic": False,
                "Licenses": set(),
                "License Use (DataProvenance)": None,
                "License Attribution (DataProvenance)": None,
                "License Share Alike (DataProvenance)": None,
                "License Use (HuggingFace)": None,
                "License Use (GitHub)": None,
                "License Use (PapersWithCode)": None,
                "Creators": set(),
                "Creator Groups": set(),
                "Creator Countries": set(),
                "Input Text Lengths": 0,
                "Target Text Lengths": 0,
                "Num Exs": 0,
                "Dialog Turns": 0,
                "Tasks": set(),
                "Task Groups": set(),
                "Languages": set(),
                "Language Groups": set(),
                "Preparation Times": None
            }
        
        info = dataset_infos[dataset_uid]

        # Update sets
        info["Sources"].update(row.get("Text Sources", []))
        info["Domains"].update({SOURCE_TO_GROUP[x] for x in row.get("Text Sources", [])})
        info["Licenses"].update([lic_info["License"] for lic_info in row.get("Licenses", [])])
        info["Creators"].update(row.get("Creators", []))
        info["Creator Groups"].update([CREATOR_TO_GROUP[x] for x in row.get("Creators", [])])
        info["Creator Countries"].update([CREATOR_TO_COUNTRY[x] for x in row.get("Creators", [])])
        info["Tasks"].update(row.get("Task Categories", []))
        info["Task Groups"].update([TASK_TO_GROUP[x] for x in row.get("Task Categories", [])])
        info["Languages"].update(row.get("Languages", []))
        info["Language Groups"].update([LANG_TO_GROUP[x] for x in row.get("Languages", [])])

        # Update numeric values
        text_metrics = row.get("Text Metrics", {})
        info["Input Text Lengths"] += text_metrics.get("Mean Inputs Length", 0)
        info["Target Text Lengths"] += text_metrics.get("Mean Targets Length", 0)
        info["Num Exs"] += text_metrics.get("Num Dialogs", 0)
        info["Dialog Turns"] += text_metrics.get("Mean Dialog Turns", 0)

        # Synthetic data flag
        info["Synthetic"] = info["Synthetic"] or (len(row.get("Model Generated", [])) > 0)

        # Update single values, assuming they are consistent across duplicated datasets
        lic_keys = ["License Use (DataProvenance)", "License Use (DataProvenance)", "License Use (DataProvenance)", "License Use (DataProvenance)", \
                    "License Attribution (DataProvenance)", "License Share Alike (DataProvenance)", "Preparation Times"]
        for field in lic_keys:
            if row.get(field) is not None:
                if info[field] is not None and info[field] != row.get(field):
                    raise ValueError(f"Inconsistent values for {field} in dataset '{dataset_uid}'")
                info[field] = row.get(field)

        s2_time = row.get("Inferred Metadata", {}).get("S2 Date") or "3000"
        pwc_time = row.get("Inferred Metadata", {}).get("PwC Date") or "3000"
        hf_time = row.get("Inferred Metadata", {}).get("Github Date") or "3000"
        gh_time = row.get("Inferred Metadata", {}).get("HF Date") or "3000"
        earlier_time = min([s2_time, pwc_time, hf_time, gh_time])
        info["Preparation Times"] = None if earlier_time == "3000" else earlier_time

    # Convert set to list to finalize
    set_keys = ["Sources", "Domains", "Licenses", "Creators", "Creator Groups", "Creator Countries", "Tasks", "Task Groups", "Languages", "Language Groups"]
    for dataset_uid, info in dataset_infos.items():
        for key in set_keys:
            info[key] = list(info[key])
            
    return dataset_infos


#################################################################
############### Visualization Helpers
#################################################################


def plot_grouped_chart(
    info_groups, 
    group_names, 
    category_key, 
    name_remapper,
    exclude_groups,
    savename
):

    groups = defaultdict(list)
    for group_name in set(group_names) - set(exclude_groups):
        for license_group, dsets_info in info_groups.items():
            count = sum([1 if group_name in cat_to_vals[category_key] else 0 for cat_to_vals in dsets_info.values()])
            if name_remapper:
                groups[name_remapper.get(group_name, group_name)].append(count)
            else:
                groups[group_name].append(count)
    print(groups)

    total_dsets = sum([len(vs) for vs in info_groups.values()])
    custom_colors = ['#e04c71','#e0cd92','#82b5cf']
    groups = {trim_label(k): v for k, v in groups.items() if sum(v)}
    group_order = [k for k, v in sorted(groups.items(), key=lambda x: x[1][0] / sum(x[1]), reverse=False)]
    if len(groups) > 16:
        group_order = group_order[:8] + group_order[-8:]
    return plot_stackedbars(
        groups, None, list(info_groups.keys()),
        custom_colors, group_order, total_dsets, legend=None, savepath=f"paper_figures/altair/{savename}")


def plot_grouped_time_chart(
    info_groups,
    category_key,
    disallow_repeat_dsetnames,
    savename
):
    START_YEAR = 2013
    
    def bucket_time(t):
        if not t:
            return None
        if int(t.split("-")[0]) < START_YEAR:
            return f"< {START_YEAR}"
        else:
            return t.split("-")[0]
            
    ordered_tperiods = [f"< {START_YEAR}"] + [str(x) for x in range(START_YEAR, 2025)]
    groups = defaultdict(list)
    for group_name in ordered_tperiods:
        seenDsets = []
        for license_group, dsets_info in info_groups.items():
            vals = []
            for cat_to_vals in dsets_info.values():
                if disallow_repeat_dsetnames and cat_to_vals["Name"] in seenDsets:
                    continue
                seenDsets.append(cat_to_vals["Name"])
                
                vals.append(1 if group_name == bucket_time(cat_to_vals[category_key]) else 0)
            groups[group_name].append(sum(vals))
            # count = sum([1 if group_name == bucket_time(cat_to_vals[category_key]) else 0 for cat_to_vals in dsets_info.values()])
            # groups[group_name].append(count)
    print(groups)
    custom_colors = ['#e04c71','#e0cd92','#82b5cf']
    return plot_stackedbars(
        groups, None, list(info_groups.keys()),
        custom_colors, ordered_tperiods, 0, legend=None, savepath=f"paper_figures/altair/{savename}")


def plot_license_breakdown(
    infos, 
    license_classes,
    disallow_repeat_dsetnames,
    savename
):
    category_remapper = {
        "All": "Commercial",
        "NC": "Non-Commercial/Academic",
        "Acad": "Non-Commercial/Academic",
        "Custom": "Custom",
    }
    licenses_remapper = {
        "GNU General Public License v3.0": "GNU v3.0",
        "Microsoft Data Licensing Agreement": "Microsoft Data License",
        "Academic Research Purposes Only": "Academic Research Only",
        "Academic Free License v3.0": "AFL v3.0",
    }

    # list of license appearances
    if disallow_repeat_dsetnames:
        license_list = defaultdict(list)
        for cat_to_val in infos.values():
            license_list[cat_to_val["Name"]] = set(license_list[cat_to_val["Name"]]).union(set(cat_to_val["Licenses"]))
        license_list = [l for ll in license_list.values() for l in ll]
    else:
        license_list = [lic for cat_to_val in infos.values() for lic in cat_to_val["Licenses"]] 

    # Remove Unspecified
    license_list = [l for l in license_list if l != "Unspecified"]
    license_counts = Counter(license_list).most_common()
    # print(sum([v for (k, v) in license_counts]))
    # print(license_counts)
    
    def license_to_attributes(license):
        if license == "Custom":
            use_case, attr, sharealike = "Custom", 0, 0
        elif license_classes[license][1] == "?":
            use_case, attr, sharealike = "Non-Commercial/Academic", 1, 1
        else:
            use_case = category_remapper[license_classes[license][0]]
            attr = 1 if int(license_classes[license][1]) else 0
            sharealike = 1 if int(license_classes[license][2]) else 0
        return use_case, attr, sharealike

    license_infos = {}
    for license, count in dict(license_counts).items():
        use_case, attr, sharealike = license_to_attributes(license)
        license_infos[license] = {
            "Count": count, "Requires Attribution": attr, "Requires Share Alike": sharealike,
            "Allowed Use": use_case,
        }
    
    custom_colors = ['#82b5cf','#e04c71','#ded9ca']
    
    plot_seaborn_barchart(
        license_infos, "Licenses", "Count", "Requires Attribution", "Requires Share Alike",
        "Allowed Use", custom_colors, f"paper_figures/{savename}"
    )
    
    total_count = sum([vd["Count"] for vd in license_infos.values()])
    num_attr = sum([vd["Count"] for vd in license_infos.values() if vd["Requires Attribution"] == 1])
    num_sa = sum([vd["Count"] for vd in license_infos.values() if vd["Requires Share Alike"] == 1])
    print(f"Fraction of Total Licenses Requiring Attribution = {round(100 * num_attr / total_count, 2)}%")
    print(f"Fraction of Total Licenses Requiring Share Alike = {round(100 * num_sa / total_count, 2)}%")



# Splitting y-label into multiple lines:
def split_label(label, maxlen=24):
    words = label.split(' ')
    line = []
    new_label = []
    char_count = 0
    for word in words:
        char_count += len(word)
        if char_count > maxlen:
            new_label.append(' '.join(line))
            line = [word]
            char_count = len(word)
        else:
            line.append(word)
    new_label.append(' '.join(line))
    return '\n'.join(new_label)


def trim_label(label, maxlen=20):
    return label if len(label) < maxlen else label[:17] + "..."

def plot_stackedbars(
    data, 
    title, 
    category_names, 
    custom_colors,
    group_order, 
    total_dsets, 
    legend=True, 
    savepath=None
):
    
    # Ensure the color list matches the number of categories
    if len(custom_colors) != len(data[list(data.keys())[0]]):
        raise ValueError("Number of colors does not match number of categories!")
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data, columns=group_order, index=category_names)
    # print(df.columns)
    df = df[group_order].T
    # print(df.columns)
    # df = df[df.columns[bar_order]]
    df.index = df.index.map(split_label)
    
    # Calculate percentages for annotations
    # print(df)
    df_percentage = df.div(df.sum(axis=1), axis=0) * 100
    
    # Melt the dataframe for Altair
    df_melted = df.reset_index().melt(id_vars='index', var_name='category', value_name='value')
    df_melted_percentage = df_percentage.reset_index().melt(id_vars='index', var_name='category', value_name='percentage')
    df_melted['percentage'] = df_melted_percentage['percentage']
    
    order_mapping = {name: i for i, name in enumerate(category_names)}

    # Add an 'order' column based on the 'category' column and our mapping.
    df_melted['order'] = df_melted['category'].map(order_mapping)
    
    # Base chart for bars
    # print(bar_order)
    # print(df_melted.category)
    bars = alt.Chart(df_melted).mark_bar(width=50).encode(
        # y=alt.Y('percentage:Q', stack="normalize", axis=alt.Axis(format='%', labelFontSize=14, titleFontSize=16, title="Percentage (%)"), scale=alt.Scale(domain=[0,1]), order=bar_order),
        x=alt.X('index:N', sort=group_order, title=None, axis=alt.Axis(labelAngle=-25, labelFontSize=14)),
        y=alt.Y('percentage:Q', stack="normalize", sort=category_names, axis=alt.Axis(format='%', labelFontSize=14, titleFontSize=16, title="Percentage (%)", titleFontWeight='normal'), scale=alt.Scale(domain=[0,1])),
        color=alt.Color('category:N', sort=category_names, scale=alt.Scale(range=custom_colors), legend=alt.Legend(title=None) if legend else None),
        order='order:O' 
    )

    # Text annotations inside bars
    text = bars.mark_text(dx=0, dy=-7, align='center', baseline='middle', color='white', fontSize=14).encode(
        text=alt.condition(alt.datum.percentage > 0.05, alt.Text('percentage:Q', format='.1f'), alt.value(''))
    )
    
    # Calculate the totals for each bar
    df_totals = df.sum(axis=1).reset_index()
    df_totals.columns = ['index', 'total']
    df_totals['text_label'] = df_totals.apply(lambda row: f"({row['total']})", axis=1)

    # Totals text above bars
    totals_text = alt.Chart(df_totals).mark_text(dy=-32, align='center', baseline='top', fontSize=14).encode(
        x=alt.X('index:N', sort=category_names, title=None),
        y=alt.value(0),  # Positions text at the top of the bar
        text='text_label:N'
    )

    # Combine all layers
    chart = bars + text + totals_text
    chart = chart.properties(title="" if title is None else title, height=140, width=850)
    
    if savepath:
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        with open(savepath, 'w') as f:
            f.write(chart.to_json())
        # chart.save(savepath)#, format='svg')
    # else:
    return chart

def plot_seaborn_barchart(
    counts, 
    xlabel, 
    ylabel, 
    featureA, 
    featureB, 
    featureC, 
    custom_colors, 
    savepath=None
):
    plt.rcParams['font.family'] = 'Helvetica'
    # Convert counts to a DataFrame
    df = pd.DataFrame({
        xlabel: [split_label(k) for k in counts.keys()],
        ylabel: [v[ylabel] for v in counts.values()],
        featureA: [v[featureA] for v in counts.values()],
        featureB: [v[featureB] for v in counts.values()],
        featureC: [v[featureC] for v in counts.values()],
    })
    
    color_dict = dict(zip(df[featureC].unique(), custom_colors))
    df['color'] = df[featureC].map(color_dict)
    
    df['percentage'] = 100 * df[ylabel] / df[ylabel].sum()

    # sort the DataFrame and select the top categories
    df = df.sort_values(ylabel, ascending=False)[:21]

    # print (df)
    # Create the bar plot
    plt.figure(figsize=(20, 8))
    ax = sns.barplot(x=xlabel, y=ylabel, data=df, width=0.7)  # Adjust the width for increased spacing between bars
    
    # FeatureA edge color and FeatureB denser hatch pattern
    edge_color = 'purple'
    denser_hatch = '||'
    
    for idx, bar in enumerate(ax.patches):
        bar.set_facecolor(df.iloc[idx]['color'])
        if df.iloc[idx][featureA]:
            bar.set_edgecolor(edge_color)
            bar.set_linewidth(2)  # Set edge width for clarity
        if df.iloc[idx][featureB]:
            bar.set_hatch(denser_hatch)
    
    # Custom legend for edge colors and hatches
    legend_patches = [
        Patch(facecolor='gray', edgecolor=edge_color, linewidth=2, label=featureA),
        Patch(facecolor='gray', hatch=denser_hatch, label=featureB, edgecolor='purple'),
        # Rectangle((0, 0), 1, 1, facecolor='gray', hatch=denser_hatch, edgecolor='purple'),  # Custom patch for purple hatch

        # Patch(facecolor='gray', edgecolor=edge_color, linewidth=1.5, hatch=denser_hatch, label=f"{featureA} & {featureB}")
    ]
    # Adding patches for FeatureC colors
    for feature_value, color in color_dict.items():
        legend_patches.append(Patch(facecolor=color, label=f"{featureC}: {feature_value}"))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=20)
    
    # Remove the border around the legend
    legend = ax.get_legend()
    legend.set_frame_on(False)

    # Add text labels
    for idx, bar in enumerate(ax.patches):
        # Adjusted the text positions to display count and percentage values above one another
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.05 * df[ylabel].max()), 
                f"{df.iloc[idx][ylabel]}", 
                ha='center', va='center', color='black', fontsize=18)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.14 * df[ylabel].max()), 
                f"({df.iloc[idx]['percentage']:.1f}%)", 
                ha='center', va='center', color='black', fontsize=18)
        
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=18, rotation=65)  # Rotate x-axis labels to 65 degrees
    ax.yaxis.set_tick_params(labelsize=18)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, format='pdf', bbox_inches='tight')
    else:
        plt.show()