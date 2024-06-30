import os
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

from helpers import io
from . import analysis_constants
from . import visualization_util


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
    url_to_issue = set()
    all_fpaths = [fp for dirpath in dirpaths for fp in io.listdir_nohidden(dirpath)]
    for fpath in all_fpaths:
        df = pd.read_csv(fpath)
        if 'User Content' in df:
            df['User Content'] = df['User Content'].fillna('None')
        df = df.fillna("")
        overwrite_attempts = 0
        for _, row in df.iterrows():
            domain = row["Domain"]
            if not domain.startswith("www."):
                domain = "www." + domain
            row_info = extract_row_info(row)
            if domain in url_to_issue or row_info["Website Issue"]:
                url_to_issue.add(domain)
            if domain in url_to_info:
                overwrite_attempts += 1
                url_to_info[domain].update(row_info)
            else:
                url_to_info[domain] = row_info
        # print(f"{fpath}: + {len(df)} = {len(url_to_info)} | overwritten: {overwrite_attempts}")
                
    print(f"{len(url_to_info)} rows before filtering.")
    # filter out incomplete rows:
    url_to_rows = {}
    unannotated_urls = {}
    issue_counter, unannotated_counter = 0, 0
    for url, infos in url_to_info.items():
        if url in url_to_issue:
            issue_counter += 1
            continue
        elif not infos.get('Website Description', "") or not infos.get("Paywall", ""):
            unannotated_counter += 1
            unannotated_urls[url] = infos
            continue
        url_to_rows[url] = infos
    print(f"{len(url_to_rows)} rows after filtering. {issue_counter} issues, {unannotated_counter} unannotated.")
    return url_to_rows, unannotated_urls


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
    # print(od2)

    all_typ_cols = ["Type of service", "Type of service (Other)"]
    url_to_services, other_services, os2 = categorize_domain_annotations(url_to_info, all_typ_cols, analysis_constants.WEBSITE_SERVICE_INVERSE_MAPPING)
    # service_counter = Counter(["-".join(vals) for vals in url_to_services.values()])
    # print(os2)

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

    df = pd.DataFrame(url_results)

    domain_vars = df['Domains'].apply(pd.Series).columns
    service_vars = df['Services'].apply(pd.Series).columns

    # Expand the 'tags' lists into a DataFrame of True/False values
    domains_expanded = pd.get_dummies(df['Domains'].explode()).groupby(level=0).max()
    domains_expanded.columns = ['domain_' + col for col in domains_expanded.columns]
    services_expanded = pd.get_dummies(df['Services'].explode()).groupby(level=0).max()
    services_expanded.columns = ['services_' + col for col in services_expanded.columns]

    # Combine with the original DataFrame
    df = df.join(domains_expanded).join(services_expanded)

    return df
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

def analyze_url_variable_correlations(
    df, top_n_list, # corpus_key="c4"):
    c4_estimates,
    rf_estimates,
    dolma_estimates,
):
    """
    corpus_key: 'c4', 'rf' or 'dolma'
    """
    binary_vars = [
        'User Content', 'Paywall', 'Ads', 'Modality: Image', 'Modality: Video', 'Modality: Audio', 'Sensitive Content',
        'Restrictive Robots.txt', 'Restrictive Terms'
    ]

    # return df

    all_vars = c4_estimates.keys()

    # Sort the dataframe by 'c4 tokens' in descending order
    # df = df.sort_values(by=f"{corpus_key} tokens", ascending=False)

    # Create an empty dataframe to store the results
    # results_df = pd.DataFrame(index=all_vars, columns=[f"Top {n}" for n in top_n_list] + ['Random', 'Chi-Squared Stat', 'P-value'])
    results_df = pd.DataFrame(index=all_vars, columns=[f"Top {n}" for n in top_n_list] + ['Random', 'C4', 'RW', 'Dolma'])

    def compute_percentages(df, cols):
        result = {}
        for col in cols:
            count = df[col].sum()
            total = len(df)
            result[col] = round(100 * count / total, 2)
        return result
    
    for top_n in top_n_list:
        top_n_c4_df = df[df[f"c4 rank"] <= top_n]
        top_n_rf_df = df[df[f"rf rank"] <= top_n]
        top_n_dolma_df = df[df[f"dolma rank"] <= top_n]
        # print(f"Num URLs in Top-{top_n}: {len(top_n_df)}")
        c4_pct = compute_percentages(top_n_c4_df, all_vars)
        rf_pct = compute_percentages(top_n_c4_df, all_vars)
        dolma_pct = compute_percentages(top_n_c4_df, all_vars)
        mean_pcts = {}
        for var in all_vars:
            mean_pcts[var] = np.mean([c4_pct[var], rf_pct[var], dolma_pct[var]])
        results_df[f'Top {top_n}'] = mean_pcts
    random_df = df[df["sample"] == "random"]
    print(f"Num URLs in random sample: {len(random_df)}")
    results_df['Random'] = compute_percentages(random_df, all_vars)

    # Perform Chi-Squared test and populate the results
    # for var in all_vars:
    #     # print(var)
    #     observed = pd.crosstab(random_df[var], pd.qcut(df[f"{corpus_key} tokens"], 4))
    #     # print(observed)
    #     chi2, p, dof, expected = chi2_contingency(observed)
    #     results_df.loc[var, 'Chi-Squared Stat'] = f"{chi2:.2f}"
    #     results_df.loc[var, 'P-value'] = f"{p:.2f}"

    # Add in C4, RW, Dolma estimates:
    for var in all_vars:
        results_df.loc[var, 'C4'] = c4_estimates[var]["Estimated Tokens Pct"]
        results_df.loc[var, 'RW'] = rf_estimates[var]["Estimated Tokens Pct"]
        results_df.loc[var, 'Dolma'] = dolma_estimates[var]["Estimated Tokens Pct"]

    return results_df


############################################################
###### Population Estimation Analysis Functions
############################################################

def combine_samples(data_head, data_random):
    """Combine head and random samples into a single DataFrame."""
    return pd.concat([data_head, data_random])

def fit_logistic_regression(X, y):
    """Fit a logistic regression model to the data."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def predict_probabilities(model, X):
    """Predict probabilities of positive states using the fitted model."""
    return model.predict_proba(X)[:, 1]

def calculate_bucket_estimates(buckets, predicted_probs):
    """Calculate the expected summed magnitude and counts of points with a positive state for each bucket."""
    buckets['predicted_prob'] = predicted_probs
    buckets['expected_positive_magnitude'] = buckets['bucket_midpoint'] * buckets['predicted_prob'] * buckets['count']
    total_summed_magnitude = buckets['expected_positive_magnitude'].sum()
    
    total_positive_count = (buckets['predicted_prob'] * buckets['count']).sum()
    total_negative_count = buckets['count'].sum() - total_positive_count
    
    return total_summed_magnitude, total_positive_count, total_negative_count

def run_empirical_bayes(data_head, data_random, buckets):
    """Run the Empirical Bayes method for a single population."""
    # Combine head and random samples
    data_combined = combine_samples(data_head, data_random)
    
    # Fit logistic regression model
    X = data_combined[['magnitude']]
    y = data_combined['binary_state']
    model = fit_logistic_regression(X, y)
    
    # Predict probabilities for the magnitude buckets
    X_buckets = buckets[['bucket_midpoint']]
    X_buckets = X_buckets.rename(columns={'bucket_midpoint': 'magnitude'})
    # print(X_buckets)
    predicted_probs = predict_probabilities(model, X_buckets)
    
    # Calculate expected summed magnitude and binary variable counts
    total_summed_magnitude, total_positive_count, total_negative_count = calculate_bucket_estimates(buckets, predicted_probs)
    
    # Fill in the known head distribution stats
    data_head['predicted_prob'] = data_head['binary_state']
    # Create a combined DataFrame of head and bucket predictions
    head_sum = data_head['magnitude'].sum()
    head_positive_sum = data_head[data_head['binary_state'] == 1]['magnitude'].sum()
    head_positive_count = data_head['binary_state'].sum()
    head_negative_count = len(data_head) - head_positive_count

    # Add head stats to bucket stats
    total_summed_magnitude += head_positive_sum
    total_positive_count += head_positive_count
    total_negative_count += head_negative_count

    return total_summed_magnitude, total_positive_count, total_negative_count

def conservative_estimate(data_head, data_random, buckets):
    """Conservative estimate using known head distribution stats and predicted stats for the rest."""
    # Fit logistic regression model
    X = data_random[['magnitude']]
    y = data_random['binary_state']
    model = fit_logistic_regression(X, y)
    
    # Predict probabilities for the magnitude buckets
    X_buckets = buckets[['bucket_midpoint']]
    X_buckets = X_buckets.rename(columns={'bucket_midpoint': 'magnitude'})
    predicted_probs = predict_probabilities(model, X_buckets)
    
    # Fill in the known head distribution stats
    data_head['predicted_prob'] = data_head['binary_state']
    
    # Create a combined DataFrame of head and bucket predictions
    head_sum = data_head['magnitude'].sum()
    head_positive_sum = data_head[data_head['binary_state'] == 1]['magnitude'].sum()
    head_positive_count = data_head['binary_state'].sum()
    head_negative_count = len(data_head) - head_positive_count
    
    # Calculate expected summed magnitude and binary variable counts for buckets excluding head
    total_summed_magnitude, total_positive_count, total_negative_count = calculate_bucket_estimates(buckets, predicted_probs)
    
    # Add head stats to bucket stats
    total_summed_magnitude += head_positive_sum
    total_positive_count += head_positive_count
    total_negative_count += head_negative_count
    
    return total_summed_magnitude, total_positive_count, total_negative_count

def process_url_population(data, method='empirical_bayes'):
    """Process population and its binary variables."""
    results = {}
    
    data_head = data['head']
    data_random = data['random']
    buckets = data['buckets']
    
    for binary_var in data['binary_vars']:
        # Update binary state column
        data_head['binary_state'] = data_head[binary_var]
        data_random['binary_state'] = data_random[binary_var]
        
        # Run chosen method
        if method == 'empirical_bayes':
            print("bayes")
            total_summed_magnitude, total_positive_count, total_negative_count = run_empirical_bayes(data_head, data_random, buckets)
        elif method == 'conservative':
            total_summed_magnitude, total_positive_count, total_negative_count = conservative_estimate(data_head, data_random, buckets)
        

        results[binary_var] = {
            'total_summed_magnitude': round(total_summed_magnitude, 2),
            'total_positive_count': round(total_positive_count, 2),
            'total_negative_count': round(total_negative_count, 2),
        }
    
    return results

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



def map_services_to_urls(url_results_df):
    service_to_urls = defaultdict(list)
    for i, row in url_results_df.iterrows():
        for service in row.Services:
            service_to_urls[service.replace("/", "_").replace("-", "_")].append(row["URL"])
        service_to_urls["all"].append(row["URL"])
    for k, vs in service_to_urls.items():
        print(f"{k}: {len(vs)}")
    return service_to_urls


def analyze_url_variable_correlations(df, all_vars, top_n_list=[100, 500, 2000]):
    ret = {}
    for top_n in top_n_list:
        top_n_c4_df = df.loc[df['c4 rank'] <= top_n]
        top_n_rf_df = df.loc[df['rf rank'] <= top_n]
        top_n_dolma_df = df.loc[df['dolma rank'] <= top_n]

        ret[f'Top {top_n}'] = (
            top_n_c4_df[all_vars].mean() +
            top_n_rf_df[all_vars].mean() +
            top_n_dolma_df[all_vars].mean()
        ) / 3
    
    ret['Random'] = df.loc[df['sample'] == 'random', all_vars].mean()

    return pd.DataFrame(ret)


def run_population_analysis(
    url_results_df, 
    url_token_lookup,
    top_corpus_urls, 
    corpus_name,
    all_vars,
    verbose=False,
):
    
    data_head = url_results_df \
        .loc[url_results_df['URL'].isin(top_corpus_urls)] \
        .rename({f"{corpus_name} tokens": 'magnitude'}, axis=1) \
        [all_vars + ['magnitude', 'URL']] \
        .set_index('URL') \
        .astype(int)
    
    data_random = url_results_df \
        .loc[url_results_df['sample'] == 'random'] \
        .rename({f"{corpus_name} tokens": 'magnitude'}, axis=1) \
        [all_vars + ['magnitude', 'URL']] \
        .set_index('URL') \
        .astype(int)

    # print(data_random)
    total_urls = url_token_lookup._TOTAL_URLS[corpus_name]
    total_tokens = url_token_lookup._TOTAL_TOKENS[corpus_name]
    total_urls_rest = total_urls - data_head.shape[0] - data_random.shape[0]
    # total_tokens_rest = total_tokens - data_head['magnitude'].sum() - data_random['magnitude'].sum()
    total_tokens_rest = total_tokens - data_head['magnitude'].sum()

    final_results = {}
    for bvar in all_vars:
        pred_rest = data_random[bvar].mean()
        pred_head = data_head[bvar].mean()

        # Tokens for [2k-rand]
        
        # compute proportion of tokens out of 2k-rand
        # (X / total_tokens) * (Y / total_tokens)
    
        est_urls = data_head[bvar].sum() + data_random[bvar].sum() + (pred_rest * total_urls_rest).sum()
        est_tokens = (
            (data_head['magnitude'] * data_head[bvar]).sum() +
            (total_tokens_rest * ((data_random['magnitude'] * data_random[bvar]).sum() / data_random['magnitude'].sum()))
        )
    
        final_results[bvar] = {
            'est_url_pct': est_urls / total_urls,
            'est_tokens_pct': est_tokens / total_tokens,
        }
        if verbose and bvar == "Restrictive Terms": # and corpus_name == "c4":
            # print(total_tokens - total_tokens_rest)
            # print(total_tokens_rest)
            head_rate = (data_head['magnitude'] * data_head[bvar]).sum() / data_head['magnitude'].sum()
            rand_rate = ((data_random['magnitude'] * data_random[bvar]).sum() / data_random['magnitude'].sum())

            head_proportion = (data_head['magnitude'] * data_head[bvar]).sum() / total_tokens
            # head_proportion = (pred_head * (data_head['magnitude'].sum() / total_tokens))
            rand_proportion = (total_tokens_rest * ((data_random['magnitude'] * data_random[bvar]).sum() / data_random['magnitude'].sum())) / total_tokens

    
    return pd.DataFrame(final_results).T


# def run_population_analysis(
#     url_results_df, 
#     top_corpus_urls, 
#     corpus_name,
#     data_buckets_fpath,
#     url_token_lookup,
#     verbose=False,
# ):
#     total_tokens = url_token_lookup._TOTAL_TOKENS[corpus_name]
#     total_urls = url_token_lookup._TOTAL_URLS[corpus_name]
    
#     top_results_df = url_results_df[url_results_df['URL'].isin(top_corpus_urls)]
#     random_results_df = url_results_df[url_results_df['sample'] == "random"]
#     print(f"Head sample size: {len(top_results_df)}")
#     print(f"Rand sample size: {len(random_results_df)}")
#     head_tokens = list(top_results_df[f"{corpus_name} tokens"])
#     rand_tokens = list(random_results_df[f"{corpus_name} tokens"])

#     cols = ['User Content', 'Paywall', 'Ads','Modality: Image', 'Modality: Video', 'Modality: Audio',
#        'Sensitive Content', 'services_Academic', 'services_Blogs',
#        'services_E-Commerce', 'services_Encyclopedia/Database',
#        'services_Government', 'services_News/Periodicals',
#        'services_Organization/Personal Website', 'services_Other',
#        'services_Social Media/Forums', 'Restrictive Robots.txt', 'Restrictive Terms']
#     # var_name --> {head -> vals, rand -> vals}
#     vars_data = {}
#     for col in cols:
#         vars_data[col] = {
#             "head": [int(x) for x in top_results_df[col]],
#             "rand": [int(x) for x in random_results_df[col]],
#         }

#     head_info = {k: v["head"] for k, v in vars_data.items()}
#     head_info.update({'magnitude': head_tokens})
#     rand_info = {k: v["rand"] for k, v in vars_data.items()}
#     rand_info.update({'magnitude': rand_tokens})
#     population = {
#         'head': pd.DataFrame(head_info),
#         'random': pd.DataFrame(rand_info),
#         'buckets': pd.read_csv(data_buckets_fpath),
#         'binary_vars': cols,
#     }
#     results = process_url_population(population, method='conservative') # conservative, empirical_bayes

#     final_results = {}
#     for bvar, var_results in results.items():
#         head_pct = round(100 * np.mean(vars_data[bvar]["head"]), 2)
#         rand_pct = round(100 * np.mean(vars_data[bvar]["rand"]), 2)
#         pos = var_results["total_positive_count"]
#         pos_pct = round(100 * pos / total_urls, 2)
#         pos_t = var_results["total_summed_magnitude"]
#         pos_t_pct = round(100 * pos_t / total_tokens, 2)
#         if verbose:
#             print(f"{bvar} | Head = {head_pct} % | Rand = {rand_pct} %")
#             print(f"Estimated URLs = {pos} / {total_urls} = {pos_pct} %")
#             print(f"Estimated Tokens = {pos_t} / {total_tokens} = {pos_t_pct} %")
#             print()
#         final_results[bvar] = {
#             "Estimated URL Pct": pos_pct,
#             "Estimated Tokens Pct": pos_t_pct,
#         }
#     return final_results