import itertools
import json
import os
import re
import typing
from collections import Counter, defaultdict
from datetime import datetime
from urllib.parse import urlparse

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from analysis import analysis_constants, visualization_util

from . import parse_robots

############################################################
###### Robots.txt Bot Methods
############################################################

# Grouping bots by company and their usage
BOT_TRACKER = {
    "*All Agents*": {
        "train": ["*All Agents*"],
        "retrieval": ["*All Agents*"],
        # An aggregation of policies towards All Agents.
    },
    "*": {"train": ["*"], "retrieval": ["*"]},
    "OpenAI": {
        "train": ["GPTBot"],
        "retrieval": ["ChatGPT-User"],
        "search": ["OAI-SearchBot"],
        # https://platform.openai.com/docs/gptbot
        # https://platform.openai.com/docs/plugins/bot
    },
    "Google": {
        "train": ["Google-Extended"],
        "retrieval": ["Google-Extended"],
        # https://developers.google.com/search/docs/crawling-indexing/overview-google-crawlers
    },
    "Common Crawl": {
        "train": ["CCBot"],
        "retrieval": ["CCBot"],
        # https://commoncrawl.org/ccbot
    },
    "Amazon": {
        "train": ["Amazonbot"],
        "retrieval": ["Amazonbot"],
        # https://developer.amazon.com/amazonbot
    },
    "False Anthropic": {
        "train": ["anthropic-ai"],
        "retrieval": ["Claude-Web"],
        # https://support.anthropic.com/en/articles/8896518-does-anthropic-crawl-data-from-the-web-and-how-can-site-owners-block-the-crawler
    },
    "Anthropic": {
        "train": ["ClaudeBot", "CCBot"],
        "retrieval": ["ClaudeBot", "CCBot"],
        # https://support.anthropic.com/en/articles/8896518-does-anthropic-crawl-data-from-the-web-and-how-can-site-owners-block-the-crawler
    },
    "Cohere": {"train": ["cohere-ai"], "retrieval": ["cohere-ai"]},
    "Meta": {
        "train": ["FacebookBot"],
        "retrieval": ["FacebookBot"],
        # https://developers.facebook.com/docs/sharing/bot/
    },
    "Internet Archive": {"train": ["ia_archiver"], "retrieval": ["ia_archiver"]},
    "Google Search": {
        "train": ["Googlebot"],
        "retrieval": ["Googlebot"],
    },
    "Perplexity AI": {
        "train": ["PerplexityBot"],
        "retrieval": ["Perplexity-User"]
        # https://docs.perplexity.ai/guides/bots#perplexity-crawlers
    },
    # "Microsoft": {
    #     "train": [],
    #     "retrieval": ["Bingbot"]
    # },
}


def get_bot_groups(groups=BOT_TRACKER.keys()):
    ret = {}
    for group in groups:
        ret[group] = list(
            set([v[0] for v in BOT_TRACKER[group].values()]) 
        )
    return ret


def get_bots(company=None, setting=None):
    """
    Get bots by optional company and setting.
    :param company: str or None, company to filter bots
    :param setting: str or None, setting to filter bots ('train' or 'retrieval')
    :return: list of bots
    """
    if company:
        if setting:
            # Return bots for a specific company and setting
            x = BOT_TRACKER.get(company, {}).get(setting, [])
            return set(x)
        # Return all bots for a specific company
        return set([bot for role in BOT_TRACKER[company].values() for bot in role])
    if setting:
        # Return all bots for a specific setting across all companies
        return set(
            [
                bot
                for company in BOT_TRACKER.values()
                for bot in company.get(setting, [])
            ]
        )
    # Return all bots
    return set(
        [
            bot
            for company in BOT_TRACKER.values()
            for role in company.values()
            for bot in role
        ]
    )


############################################################
###### URL --> Token Lookup Methods
############################################################


def normalize_url(url):
    return url if url.startswith("www.") else f"www.{url}"


class URLTokenLookup:
    def __init__(self, file_path):
        """
        Initialize the URLTokenLookup object with data from a CSV file.

        Args:
        file_path (str): The path to the CSV file containing the URL and token data.
        """
        self.file_path = file_path
        self.lookup_map = self._create_lookup_map()
        self.index_map = {"c4": 0, "rf": 1, "dolma": 2}

        self._TOTAL_TOKENS = {
            "c4": 170005451386,
            "rf": 431169198316,
            "dolma": 1974278779400,
        }
        self._TOTAL_URLS = {
            "c4": 15928138,
            "rf": 33210738,
            "dolma": 45246789,
        }  # Total URLs in common = 10136147

    def _create_lookup_map(self):
        """
        Create a lookup map from the CSV file.

        Returns:
        dict: A dictionary with URLs as keys and tuples of token counts as values.
        """
        df = pd.read_csv(self.file_path)
        # x = df.set_index('url')[['c4_tokens', 'rf_tokens', 'dolma_tokens']]
        # return {row['url']: (row['c4_tokens'], row['rf_tokens'], row['dolma_tokens']) for index, row in df.iterrows()}
        # print(df)
        df["url"] = df["url"].apply(normalize_url)
        # df = df.drop_duplicates(subset='url')
        df = (
            df.groupby("url")
            .agg(
                {
                    "c4_tokens": "sum",
                    "rf_tokens": "sum",
                    "dolma_tokens": "sum",
                }
            )
            .reset_index()
        )
        # print(df)
        # www.apnews.com,8981058,19450645,8819143
        # apnews.com,14265514,15769066,143689592

        df.set_index("url", inplace=True)

        # Step 3: Convert the DataFrame to a dictionary with tuples as values
        lookup_map = df.to_dict("index")  # Converts to a dictionary of dictionaries

        # Convert inner dictionaries to tuples for a more compact and consistent data structure
        return {url: tuple(values.values()) for url, values in lookup_map.items()}

    def total_tokens(self, dataset_name):
        """
        Compute the total count of tokens for the specified dataset.

        Args:
        dataset_name (str): The name of the dataset ('c4', 'rf', or 'dolma').

        Returns:
        int: Total number of tokens for the specified dataset.
        """
        return self._TOTAL_TOKENS[dataset_name]

    def url_tokens(self, url, dataset_name):
        """
        Get the number of tokens for a specific URL and dataset.

        Args:
        url (str): The URL to lookup.
        dataset_name (str): The dataset name ('c4', 'rf', or 'dolma').

        Returns:
        int: Number of tokens for the specified URL and dataset, or 0 if URL is not found.
        """
        dataset_index = self.index_map[dataset_name]
        return self.lookup_map.get(url, (0, 0, 0))[dataset_index]

    def top_k_urls(self, dataset_name, k, verbose=True):
        """
        Return the list of URLs with the highest token counts for the specified dataset.

        Args:
        dataset_name (str): The name of the dataset ('c4', 'rf', or 'dolma').
        k (int): The number of top URLs to return.

        Returns:
        list: List of URLs corresponding to the top K token counts.
        """
        dataset_index = self.index_map[dataset_name]

        # Create a list of (URL, token_count) tuples sorted by token_count in descending order
        sorted_urls = sorted(
            self.lookup_map.items(),
            key=lambda item: item[1][dataset_index],
            reverse=True,
        )

        # Extract the top K URLs
        top_urls = [normalize_url(url) for url, tokens in sorted_urls[:k]]
        num_tokens = sum(
            [tokens[self.index_map[dataset_name]] for url, tokens in sorted_urls[:k]]
        )
        if verbose:
            print(
                f"Number of tokens in {k} URLs: {num_tokens} | {round(100*num_tokens / self._TOTAL_TOKENS[dataset_name], 2)}% of {dataset_name}"
            )
        return top_urls

    def get_10k_random_sample(self):
        top_urls = (
            self.top_k_urls("c4", 2000, False)
            + self.top_k_urls("rf", 2000, False)
            + self.top_k_urls("dolma", 2000, False)
        )
        return [normalize_url(url) for url in self.lookup_map if url not in top_urls]

    def get_url_to_token_map(self, dataset_name):
        dataset_index = self.index_map[dataset_name]
        dataset_lookup = {k: v[dataset_index] for k, v in self.lookup_map.items()}
        return dataset_lookup


def agent_and_operation(agent_statuses):
    """Given a list of agent statuses: all, some, none, and no_robots,
    return the strictest designation."""
    if "all" in agent_statuses:
        return "all"
    elif "some" in agent_statuses:
        return "some"
    elif "none" in agent_statuses:
        return "none"
    else:
        return "no_robots"


def agent_and_operation_detailed(agent_statuses):
    """Given a list of agent statuses, return the strictest designation."""
    if "all" in agent_statuses:
        return "all"
    elif any(status.startswith("some") for status in agent_statuses):
        some_categories = [
            status for status in agent_statuses if status.startswith("some")
        ]
        if "some_pattern_restrictions" in some_categories:
            return "some_pattern_restrictions"
        elif "some_disallow_file_types" in some_categories:
            return "some_disallow_file_types"
        elif "some_disallow_important_dir" in some_categories:
            return "some_disallow_important_dir"
        else:
            return "some_other"
    elif any(status.startswith("none") for status in agent_statuses):
        none_categories = [
            status for status in agent_statuses if status.startswith("none")
        ]
        if "none_crawl_delay" in none_categories:
            return "none_crawl_delay"
        elif "none_sitemap" in none_categories:
            return "none_sitemap"
        else:
            return "none"
    else:
        return "no_robots"


def find_closest_time_key(dates, target_period, direction):
    """
    Given a list of dates, and a specific target date, find the closest
    date in the list to the target date, in the specified direction.
    """

    # Extract the target start date and end date from the target period
    target_start_date = target_period.start_time.to_pydatetime().date()
    target_end_date = target_period.end_time.to_pydatetime().date()

    closest_key = None
    # Set initial comparison values based on search direction
    if direction == "backward":
        compare = lambda current_date, closest_date: current_date > closest_date
        target_date = target_end_date
    else:  # "forward"
        compare = lambda current_date, closest_date: current_date < closest_date
        target_date = target_start_date

    # Iterate through dictionary keys
    for key in dates:
        # Convert pandas timestamp to datetime.date
        key_date = key.to_pydatetime().date()

        if (
            key_date <= target_date
            if direction == "backward"
            else key_date >= target_date
        ):
            if closest_key is None or compare(
                key_date, closest_key.to_pydatetime().date()
            ):
                closest_key = key
    return closest_key


def print_out_robots_info(loaded_robots):
    print(f"Num robot URLs loaded: {len(loaded_robots)}")
    all_times = []
    for k, vs in loaded_robots.items():
        for time in vs:
            all_times.append(time)
    all_times = set(all_times)
    print(f"Earliest time: {min(all_times)}")
    print(f"Last time: {max(all_times)}")


def compute_url_date_agent_status(data, relevant_agents):
    """
    Args:
        data: {URL --> Date --> robots.txt raw text}
        relevant_agents: List of agent names to track

    Returns:
        status_summary: {URL --> Date --> Agent --> Status} (only for relevant_agents)
        agent_counter_df: DataFrame with columns [agent, observed, all, some, none]
    """
    # Convert relevant_agents to lowercase for matching
    relevant_agents_lower = [agent.lower() for agent in relevant_agents]

    # Status summary to be returned (only for relevant agents)
    status_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    # Counter to track statuses for all agents and their case variants
    agent_counter = Counter()
    status_counter = defaultdict(lambda: Counter({"all": 0, "none": 0, "some": 0}))
    case_variants = defaultdict(
        Counter
    )  # Track case variants {lowercase: {original_case: count}}

    for url, date_to_robots in data.items():
        url = normalize_url(url)
        if None in date_to_robots:
            print(url)
        _, parsed_result = parse_robots.analyze_robots(date_to_robots)

        for date_str, agent_to_status in parsed_result.items():
            date = pd.to_datetime(date_str)

            # Create case-insensitive lookup for current agents
            agent_to_status_lower = {
                agent.lower(): (status, agent)
                for agent, status in agent_to_status.items()
            }
            for agent in relevant_agents:
                agent_lower = agent.lower()
                if agent_lower in agent_to_status_lower:
                    status, original_case = agent_to_status_lower[agent_lower]
                    case_variants[agent_lower][original_case] += 1
                    status_summary[url][date][agent] = status
                else:
                    # Fall back to wildcard rules
                    status = agent_to_status.get("*", "none")
                    status_summary[url][date][agent] = status

            # Update counters using lowercase matching
            for agent, (status, original_case) in agent_to_status_lower.items():
                agent_counter[agent] += 1
                if status == "all":
                    status_counter[agent]["all"] += 1
                elif status == "none":
                    status_counter[agent]["none"] += 1
                else:
                    status_counter[agent]["some"] += 1

    # Use most common case variant for each agent in the final DataFrame
    agent_case_map = {
        agent_lower: max(variants.items(), key=lambda x: x[1])[0]
        for agent_lower, variants in case_variants.items()
    }

    # Create DataFrame using the most common case variants
    agent_counter_df = pd.DataFrame(
        [
            (
                agent_case_map.get(agent, agent),  # Use most common case variant
                agent_counter[agent],
                counts["all"],
                counts["some"],
                counts["none"],
            )
            for agent, counts in status_counter.items()
        ],
        columns=["agent", "observed", "all", "some", "none"],
    )

    return status_summary, agent_counter_df


# Example usage (assuming you have the data and relevant_agents variables)
# status_summary, agent_counter_df = compute_url_date_agent_status(data, relevant_agents)
# print(agent_counter_df)


# Example usage (assuming you have the data and relevant_agents variables)
# status_summary, agent_counter_df = compute_url_date_agent_status(data, relevant_agents)
# print(agent_counter_df)


def compute_url_date_agent_status_detailed(data, relevant_agents):
    """
    Args:
        data: {URL --> Date --> robots.txt raw text}
        relevant_agents: List of agent names to track

    Returns: {URL --> Date --> Agent --> Status}
    """
    status_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    for url, date_to_robots in data.items():
        url = normalize_url(url)
        _, parsed_result = parse_robots.analyze_robots(date_to_robots)
        for date_str, agent_to_status in parsed_result.items():
            date = pd.to_datetime(date_str)
            for agent in relevant_agents:
                status = agent_to_status.get(agent, agent_to_status.get("*", "none"))
                robots_txt = date_to_robots[date_str]

                if status == "some":
                    if re.search(r"Disallow:\s+/.*\?", robots_txt):
                        status = "some_pattern_restrictions"
                    elif re.search(
                        r"Disallow:\s*/(?:admin|private|confidential)", robots_txt
                    ):
                        status = "some_disallow_important_dir"
                    else:
                        status = "some_other"
                elif status == "none" or status == "*":
                    if re.search(r"Crawl-delay:", robots_txt):
                        status = "none_crawl_delay"
                    elif re.search(r"Sitemap", robots_txt) or re.search(
                        r"sitemap", robots_txt
                    ):
                        status = "none_sitemap"
                    else:
                        status = "none"
                status_summary[url][date][agent] = status
    return status_summary


def sanitize_url(url: str) -> str:
    """
    Sanitizes URL to be used as a folder name.
    """
    parsed_url = urlparse(url)
    sanitized_netloc = parsed_url.netloc.replace(".", "_")
    sanitized_path = "_".join(
        filter(None, re.split(r"\/+", parsed_url.path.strip("/")))
    )
    sanitized_url = f"{sanitized_netloc}_{sanitized_path}"
    return sanitized_url.replace(".", "_")


def read_start_dates(fpath, robots_urls):
    # Load the JSON data
    with open(fpath, "r") as file:
        start_dates = json.load(file)

    # Map the sanitized URLs back to the original URLs
    website_start_dates = {
        normalize_url(url): start_dates.get(
            sanitize_url(url), pd.to_datetime("1970-01-01")
        )
        for url in robots_urls
    }
    return website_start_dates


def prepare_robots_temporal_summary(
    url_robots_summary,
    group_to_agents,
    start_time,
    end_time,
    time_frequency="M",
    website_start_dates=None,
):
    """
    Fill in the missing weeks for each URL.

    Args:
        url_robots_summary: {URL --> Date --> Agent --> Status}
        group_to_agents: {group_name --> [agents]}
        start_time: YYYY-MM-DD
        end_time: YYYY-MM-DD
        time_frequency: "M" = Monthly, "W" = Weekly.
        website_start_dates: {URL --> start_date} (optional)

    Returns:
        {Period --> Agent --> Status --> set(URLs)}
    """
    start_date = pd.to_datetime(start_time)
    end_date = pd.to_datetime(end_time)
    date_range = pd.period_range(start_date, end_date, freq=time_frequency)
    filled_status_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for period in date_range:
        for url, start_date in website_start_dates.items():
            if pd.isnull(start_date):  # Skip URLs without a start date
                continue
            if (
                url not in url_robots_summary
            ):  # Skip URLs that don't exist in url_robots_summary
                continue
            if period.start_time >= start_date:
                date_agent_status = url_robots_summary[url]
                robots_time_keys = sorted(list(date_agent_status.keys()))
                time_key = find_closest_time_key(
                    robots_time_keys, period, direction="backward"
                )

                for group, agents in group_to_agents.items():
                    if time_key is None:
                        filled_status_summary[period][group]["no_robots"].add(
                            url
                        )  # Site exists but no robots.txt file for the period
                    else:
                        statuses = [
                            date_agent_status[time_key].get(agent, "no_robots")
                            for agent in agents
                        ]
                        group_status = agent_and_operation(statuses)
                        filled_status_summary[period][group][group_status].add(url)

    return filled_status_summary


def prepare_robots_temporal_summary_detailed(
    url_robots_summary,
    group_to_agents,
    start_time,
    end_time,
    time_frequency="M",
    website_start_dates=None,
):
    """
    Fill in the missing weeks for each URL.

    Args:
        url_robots_summary: {URL --> Date --> Agent --> Status}
        group_to_agents: {group_name --> [agents]}
        start_time: YYYY-MM-DD
        end_time: YYYY-MM-DD
        time_frequency: "M" = Monthly, "W" = Weekly.
        website_start_dates: {URL --> start_date} (optional)

    Returns:
        {Period --> Agent --> Status --> set(URLs)}
    """
    start_date = pd.to_datetime(start_time)
    end_date = pd.to_datetime(end_time)
    date_range = pd.period_range(start_date, end_date, freq=time_frequency)
    filled_status_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for period in date_range:
        for url, start_date in website_start_dates.items():
            if pd.isnull(start_date):  # Skip URLs without a start date
                continue
            if (
                url not in url_robots_summary
            ):  # Skip URLs that don't exist in url_robots_summary
                continue
            if period.start_time >= start_date:
                date_agent_status = url_robots_summary[url]
                robots_time_keys = sorted(list(date_agent_status.keys()))
                time_key = find_closest_time_key(
                    robots_time_keys, period, direction="backward"
                )

                for group, agents in group_to_agents.items():
                    if time_key is None:
                        filled_status_summary[period][group]["no_robots"].add(
                            url
                        )  # Site exists but no robots.txt file for the period
                    else:
                        statuses = [
                            date_agent_status[time_key].get(agent, "no_robots")
                            for agent in agents
                        ]
                        group_status = agent_and_operation_detailed(statuses)
                        filled_status_summary[period][group][group_status].add(url)

    return filled_status_summary


def robots_temporal_to_df(filled_status_summary, strictness_order, url_to_counts={}):
    """
    Args:
        filled_status_summary: {Period --> Agent --> Status --> set(URLs)}
        strictness_order: List of statuses ordered from least strict to most strict
        url_to_counts: {url -> num tokens}. If available, will sum the tokens for each URL in counts

    Returns:
        Dataframe: [Period, Agent, Status, Count, Tokens]
    """

    # Convert results to a DataFrame for easy viewing and manipulation
    summary_df_list = []
    combined_agent_summary = {}

    for period, agent_statuses in filled_status_summary.items():
        combined_agent_summary[period] = {}

        # Collect all unique URLs and their statuses across agents for the current period
        url_statuses = {}
        for agent, statuses in agent_statuses.items():
            for status, urls in statuses.items():
                for url in urls:
                    if url not in url_statuses:
                        url_statuses[url] = []
                    url_statuses[url].append((agent, status))

        # Determine the strictest status for each URL
        for url, agent_status_list in url_statuses.items():
            strictest_status = max(
                agent_status_list, key=lambda x: strictness_order.index(x[1])
            )[1]
            if strictest_status not in combined_agent_summary[period]:
                combined_agent_summary[period][strictest_status] = set()
            combined_agent_summary[period][strictest_status].add(url)

        for agent, statuses in agent_statuses.items():
            for status, urls in statuses.items():
                summary_df_list.append(
                    {
                        "period": period,
                        "agent": agent,
                        "status": status,
                        "count": len(urls),
                        "tokens": sum([url_to_counts.get(url, 0) for url in urls]),
                    }
                )

    # Add the "Combined Agent" data
    for period, statuses in combined_agent_summary.items():
        for status, urls in statuses.items():
            summary_df_list.append(
                {
                    "period": period,
                    "agent": "Combined Agent",
                    "status": status,
                    "count": len(urls),
                    "urls": urls,
                    "tokens": sum([url_to_counts.get(url, 0) for url in urls]),
                }
            )

    summary_df = pd.DataFrame(summary_df_list)
    return summary_df


def get_latest_url_robot_statuses(url_robots_summary, agents):
    # {URL --> Date --> Agent --> Status}
    # URL —> status
    print(type(url_robots_summary))
    result = {}
    for url, date_agent_status in url_robots_summary.items():
        if date_agent_status:
            final_date = max(date_agent_status.keys())
            # for agents in group_to_agents:
            statuses = [date_agent_status[final_date][agent] for agent in agents]
            if not statuses:
                print(date_agent_status[final_date])
            group_status = agent_and_operation(statuses)
            result[url] = group_status

            # assert agent in date_agent_status[final_date], f"{agent} not in {date_agent_status[final_date].keys()}"
            # result[url] = date_agent_status[final_date][agent]
    return result


############################################################
###### ToS Processing
############################################################


def switch_dates_yearly_to_monthly(nested_dict):
    for url, date_dict in nested_dict.items():
        for date, tos_dict in list(date_dict.items()):
            # Split the date and take the first two parts to form "YYYY-MM"
            new_date = "-".join(date.split("-")[:2])
            # Assign the new date to the same sub-dictionary
            nested_dict[url][new_date] = nested_dict[url].pop(date)
    return nested_dict


def tos_policy_merging(
    verdict_ai,
    verdict_license,
    compete_verdict,
):
    if verdict_license == "Non-Comercial Only":
        verdict_license = "NC Only"
    elif verdict_license == "Conditionally Commercial":
        verdict_license = "Conditional Use"

    if verdict_ai == "No Terms Pages":
        return verdict_ai
    if verdict_ai == verdict_license == compete_verdict:
        return verdict_ai

    if verdict_ai.startswith("No"):
        return verdict_ai
    elif verdict_license == "NC Only":
        return verdict_license
    elif compete_verdict in ["Non-Compete", "No Re-Distribution"]:
        return compete_verdict
    elif "Conditional" in verdict_ai or "Conditional" in verdict_license:
        return verdict_ai
    else:
        return verdict_ai


def determine_tos_verdicts(
    tos_time,
    ai_verdict_dict,
    license_verdict_dict,
    compete_verdict_dict,
):
    verdict = None
    if tos_time in license_verdict_dict and tos_time in compete_verdict_dict:
        verdict_ai_codes = [
            vinfo["verdict"] for vinfo in ai_verdict_dict[tos_time].values()
        ]
        verdict_license_codes = [
            vinfo["verdict"] if vinfo["verdict"] else 3
            for vinfo in license_verdict_dict[tos_time].values()
        ]
        verdict_compete_codes = [
            vinfo["verdict"] if vinfo["verdict"] else 4
            for vinfo in compete_verdict_dict[tos_time].values()
        ]
        ai_verdict = analysis_constants.TOS_AI_SCRAPING_VERDICT_MAPPER[
            min(verdict_ai_codes)
        ]
        license_verdict = analysis_constants.TOS_LICENSE_VERDICT_MAPPER[
            min(verdict_license_codes)
        ]
        compete_verdict = analysis_constants.TOS_COMPETE_VERDICT_MAPPER[
            min(verdict_compete_codes)
        ]
        verdict = tos_policy_merging(ai_verdict, license_verdict, compete_verdict)
    return verdict


def get_tos_url_time_verdicts(
    tos_policies,
    tos_license_policies,
    tos_compete_policies,
    manual_annotated_urls,
    website_start_dates,
):
    """
    Input:
        URL --> time --> ToS subpage --> {verdict: <code>, evidence: <evidence>}
        URL --> time --> ToS subpage --> {verdict: <code>, evidence: <evidence>}
        URL --> time --> ToS subpage --> {verdict: <code>, evidence: <evidence>}
        List of manually annotated URLs
        URL --> Start date.

    Returns:
        URL --> time --> ToS verdict string.
    """

    url_to_time_to_verdict = {}
    misses, hits = 0, 0
    for url, time_to_subpage_to_verdicts in tos_policies.items():
        for tos_time, subpage_verdict in time_to_subpage_to_verdicts.items():
            verdict = determine_tos_verdicts(
                tos_time,
                time_to_subpage_to_verdicts,
                tos_license_policies[url],
                tos_compete_policies[url],
            )
            if verdict is None:
                misses += 1
            else:
                hits += 1
                if url not in url_to_time_to_verdict:
                    url_to_time_to_verdict[url] = {}
                url_to_time_to_verdict[url].update({tos_time: verdict})

    # print(len(url_to_time_to_verdict))
    for url in manual_annotated_urls:
        if url in website_start_dates and url not in url_to_time_to_verdict:
            url_to_time_to_verdict[url] = {
                website_start_dates[url].strftime("%Y-%m-%d"): "No Terms Pages"
            }
    # print(len(url_to_time_to_verdict))

    print(f"{misses} / {misses + hits} dates missed due to time mismatches.")
    return url_to_time_to_verdict


def prepare_tos_temporal_summary(
    tos_time_verdicts,
    start_time,
    end_time,
    time_frequency="M",
    website_start_dates=None,
):
    """
    Fill in the missing weeks for each URL.

    Args:
        tos_time_verdicts: {URL --> Date --> Verdict}
        start_time: YYYY-MM-DD
        end_time: YYYY-MM-DD
        time_frequency: "M" = Monthly, "W" = Weekly.
        website_start_dates: {URL --> start_date} (optional)

    Returns:
        {Period --> Status --> set(URLs)}
    """
    start_date = pd.to_datetime(start_time)
    end_date = pd.to_datetime(end_time)
    date_range = pd.period_range(start_date, end_date, freq=time_frequency)
    filled_status_summary = defaultdict(lambda: defaultdict(set))

    for period in date_range:
        for url, start_date in website_start_dates.items():
            if pd.isnull(start_date):  # Skip URLs without a start date
                continue
            if (
                url not in tos_time_verdicts
            ):  # Skip URLs that don't exist in url_robots_summary
                continue
            if period.start_time >= start_date:
                date_to_verdict = tos_time_verdicts[url]
                tos_time_keys = sorted(
                    [pd.to_datetime(date_str) for date_str in date_to_verdict.keys()]
                )
                time_key = find_closest_time_key(
                    tos_time_keys, period, direction="backward"
                )

                # for group, agents in group_to_agents.items():
                if time_key is None:
                    filled_status_summary[period]["No Terms Pages"].add(
                        url
                    )  # Site exists but no robots.txt file for the period
                else:
                    time_key = time_key.strftime("%Y-%m-%d")
                    if time_key not in date_to_verdict:
                        print(f"Found bug: {date_to_verdict}")
                    filled_status_summary[period][date_to_verdict[time_key]].add(url)

    return filled_status_summary


def tos_temporal_to_df(filled_status_summary, url_set, url_to_counts={}):
    """
    Args:
        filled_status_summary: {Period --> Status --> set(URLs)}
        url_set: Limit to this list of URLs (e.g. head vs random)
        url_to_counts: {url -> num tokens}. If available, will sum the tokens for each URL in counts

    Returns:
        Dataframe: [Period, Status, Count, Tokens]
    """

    # Convert results to a DataFrame for easy viewing and manipulation
    summary_df_list = []
    for period, status_urls in filled_status_summary.items():
        for status, urls in status_urls.items():
            included_urls = [url for url in urls if url in url_set]
            if len(included_urls) < len(urls):
                print(f"{len(included_urls)} | {len(urls)}")
            summary_df_list.append(
                {
                    "period": period,
                    "status": status,
                    "count": len(included_urls),
                }
            )
            if url_to_counts:
                summary_df_list[-1].update(
                    {"tokens": sum([url_to_counts[url] for url in included_urls])}
                )

    summary_df = pd.DataFrame(summary_df_list)
    return summary_df


############################################################
###### Other Helper Functions
############################################################


def bucket_urls_by_size(url_sizes, bucket_boundaries):
    bucket_keys = [
        f"{bucket_boundaries[i]}-{bucket_boundaries[i+1]}"
        for i in range(len(bucket_boundaries) - 1)
    ]

    # {"lower_bound - upper_bound" -> [list of URLs]}
    bucket_to_urls = defaultdict(list)
    for url, size in url_sizes.items():
        for i in range(len(bucket_boundaries) - 1):
            if bucket_boundaries[i] <= size < bucket_boundaries[i + 1]:
                bucket_to_urls[bucket_keys[i]].append(url)
                break

    # Print the results
    for i, bucket in enumerate(bucket_keys):
        print(f"Bucket {bucket}: {len(bucket_to_urls[bucket])}")

    return bucket_to_urls


# analyze_url_changes(robots_filled_status_summary, "*All Agents*")
def analyze_url_changes(data, agent):
    periods = sorted(data.keys())  # Ensure periods are in order
    previous_urls = {}
    results = {}

    for period in periods:
        current_urls = {}
        unchanged = 0
        changed = defaultdict(lambda: 0)
        # Aggregate all URLs for the current period
        for status, urls in data[period][agent].items():
            for url in urls:
                current_urls[url] = status

        if period == periods[0]:
            # Skip comparison for the first period
            previous_urls = current_urls
            results[period] = {"unchanged": unchanged, "changed": changed}
            continue

        # Compare with previous period
        for url, status in current_urls.items():
            if url in previous_urls:
                if previous_urls[url] == status:
                    unchanged += 1
                else:
                    changed[(previous_urls[url], status)] += 1

        results[period] = {"unchanged": unchanged, "changed": changed}
        previous_urls = current_urls

    return results


def plot_size_against_restrictions(
    url_robots_summary, size_bucket_to_urls, agent_group, setting=None
):
    agent_names = get_bots(agent_group, setting=setting)
    # {URL --> Date --> Agent --> Status} --> {URL —> status}
    current_url_status = get_latest_url_robot_statuses(url_robots_summary, agent_names)
    print(len(current_url_status))

    set(current_url_status.keys())

    cat_keys = ["all", "some", "none"]
    data_groups = defaultdict(lambda: [0, 0, 0])
    url_to_bucket_key = {
        url: sz for sz, urls in size_bucket_to_urls.items() for url in urls
    }
    # data: "bucket range": [full restrictions, some restrictions, no restrictions]
    for url, status in current_url_status.items():
        if not status:
            print(url)
            print(status)
        data_groups[url_to_bucket_key[url]][cat_keys.index(status)] += 1

    print(data_groups)

    return visualization_util.plot_stackedbars(
        data_groups,
        title=None,
        category_names=["Full Restrictions", "Some Restrictions", "No Restrictions"],
        custom_colors=["#e04c71", "#e0cd92", "#82b5cf"],
        group_order=sorted(
            size_bucket_to_urls.keys(), key=lambda x: int(x.split("-")[0])
        ),
        total_dsets=len(url_to_bucket_key),
        legend=True,
        savepath=f"paper_figures/altair/robots_restrictions_vs_token_count_{agent_group}.json",
    )


def tos_get_most_recent_verdict(
    tos_policies,
    tos_license_policies,
    tos_compete_policies,
):
    url_to_recent_policy = {}
    for url, time_to_subpage_to_verdicts in tos_policies.items():
        recent_key = max(time_to_subpage_to_verdicts.keys())

        url_to_recent_policy[url] = determine_tos_verdicts(
            recent_key,
            time_to_subpage_to_verdicts,
            tos_license_policies[url],
            tos_compete_policies[url],
        )

    return url_to_recent_policy


def prepare_recent_robots_tos_info(
    tos_policies_dict,
    tos_license_policies,
    tos_compete_policies,
    url_robots_summary,
    companies,
):
    agent_names = [agent for company in companies for agent in get_bots(company)]
    # {URL --> Date --> Agent --> Status} --> {URL —> status}
    url_robots_status = get_latest_url_robot_statuses(url_robots_summary, agent_names)
    # print(len(url_robots_status))
    url_tos_verdicts = tos_get_most_recent_verdict(
        tos_policies_dict, tos_license_policies, tos_compete_policies
    )
    return url_robots_status, url_tos_verdicts


def encode_latest_tos_robots_into_df(
    url_results_df,
    tos_policies,
    tos_license_policies,
    tos_compete_policies,
    url_robots_summary,
    companies,
    robots_detailed=False,
):

    recent_url_robots, recent_url_tos_verdicts = prepare_recent_robots_tos_info(
        tos_policies,
        tos_license_policies,
        tos_compete_policies,
        url_robots_summary,
        companies,
    )
    url_results_df["robots"] = url_results_df["URL"].map(recent_url_robots)
    url_results_df["robots"].fillna("none", inplace=True)
    if robots_detailed:
        url_results_df["Restrictive Robots.txt"] = url_results_df["robots"].map(
            {
                "no_robots": False,
                "none": False,
                "none_sitemap": False,
                "none_crawl_delay": False,
                "some_pattern_restrictions": False,
                "some_disallow_important_dir": False,
                "some_other": False,
                "all": True,
            }
        )
    else:
        url_results_df["Restrictive Robots.txt"] = url_results_df["robots"].map(
            {"all": True, "some": False, "none": False}
        )

    url_results_df["ToS"] = url_results_df["URL"].map(recent_url_tos_verdicts)
    # print(url_results_df[url_results_df["URL"]])
    # print("start")
    # print(len(recent_url_tos_verdicts))
    # print(len(url_results_df))
    # print(len(url_results_df[url_results_df["sample"] == "random"]))
    # print(len(url_results_df[url_results_df["sample"] == "random"][url_results_df["ToS"].isna()]))
    url_results_df["ToS"].fillna("No Terms Pages", inplace=True)
    # print(len(url_results_df[url_results_df["sample"] == "random"][url_results_df["ToS"].isin(["Unrestricted Use","No Terms Pages"])]))
    # print(Counter(url_results_df["ToS"].tolist()))

    url_results_df["Restrictive Terms"] = url_results_df["ToS"].apply(
        lambda x: x not in ["Unrestricted Use", "No Terms Pages"]
    )

    url_to_tos_map = dict(zip(url_results_df["URL"], url_results_df["ToS"]))

    return url_results_df  # , url_to_tos_map


def plot_robots_time_map_original(df, agent_type, val_key, frequency="M"):

    filtered_df = df[df["agent"] == agent_type]

    grouped_df = (
        filtered_df.groupby(["period", "status"])[val_key].sum().unstack(fill_value=0)
    )
    ordered_statuses = [
        "N/A",
        "none",
        "some",
        "all",
    ]
    grouped_df = grouped_df[ordered_statuses]
    total_counts = grouped_df.sum(axis=1)

    percent_df = grouped_df.div(total_counts, axis=0) * 100

    colors = ["gray", "blue", "orange", "red"]

    percent_df.columns = [
        "No Website/Robots",
        "No Restrictions",
        "Some Restrictions",
        "Full Restrictions",
    ]
    # gray (n/a), blue (some), red (none), orange (all)
    # Plotting the stacked area chart
    # percent_df.plot(kind='area', stacked=True, figsize=(10, 6))#, color=colors)
    percent_df.plot(kind="area", stacked=True, figsize=(10, 6), color=colors)

    plt.title(
        f"Restriction Status for {agent_type} over 10k Random Sample [{frequency}]"
    )
    plt.xlabel("Period")
    plt.ylabel("Percentage")
    plt.legend(title="Status")
    plt.show()
    plt.clf()


def prepare_tos_robots_confusion_matrix(
    tos_policies,
    tos_license_policies,
    tos_compete_policies,
    url_robots_summary,
    companies,
    url_token_lookup,
    use_token_counts=True,
    corpora_choice="c4",
    font_size=20,
    font_style="sans-serif",
    width=400,
    height=400,
):
    recent_url_robots, recent_tos_verdicts = prepare_recent_robots_tos_info(
        tos_policies,
        tos_license_policies,
        tos_compete_policies,
        url_robots_summary,
        companies,
    )

    ROBOTS_LABELS = {
        "none": "None",
        "some": "Partial",
        "all": "Restricted",
    }
    yaxis_order = ["Restricted", "Partial", "None"]
    xaxis_order = [
        "No Restrictions",
        "Conditional Restrictions",
        "Prohibits AI",
        "Prohibits Scraping",
        "Prohibits Scraping & AI",
    ]

    # Create a defaultdict to store counts
    counts = defaultdict(lambda: defaultdict(int))
    token_counts = defaultdict(lambda: defaultdict(int))

    # Count the occurrences of each (status, policy) pair
    total_instances, total_tokens = 0, 0
    url_token_counts = url_token_lookup.get_url_to_token_map(corpora_choice)
    for url in set(recent_url_robots.keys()).intersection(
        set(recent_tos_verdicts.keys())
    ):
        # for url in url_to_status.keys():
        status = ROBOTS_LABELS[recent_url_robots.get(url, "none")]
        policy = recent_tos_verdicts.get(url, "No Restrictions")
        counts[status][policy] += 1
        total_instances += 1
        token_counts[status][policy] += url_token_counts[url]
        total_tokens += url_token_counts[url]

    # Create a list of tuples (status, policy, count)
    data = [
        {
            "Robots Restrictions": status,
            "Terms of Service Policies": policy,
            "Count": count,
            "Token Counts": token_counts[status][policy],
            "Percent": round(100 * count / total_instances, 2),
            "Percent Tokens": round(
                100 * token_counts[status][policy] / total_tokens, 2
            ),
        }
        for status in yaxis_order
        for policy in xaxis_order
        if (count := counts[status][policy]) > 0
    ]

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data)
    df["Formatted Percent"] = df["Percent"].apply(lambda x: f"{x:.1f} %")
    df["Formatted Percent Tokens"] = df["Percent Tokens"].apply(lambda x: f"{x:.1f} %")

    if use_token_counts:
        color_axis, text_axis = "Percent Tokens", "Formatted Percent Tokens"
    else:
        color_axis, text_axis = "Percent", "Formatted Percent"

    # print(df)
    return visualization_util.plot_confusion_matrix(
        df,
        yaxis_order=yaxis_order,
        xaxis_order=xaxis_order,
        text_axis=text_axis,
        color_axis=color_axis,
        yaxis_title="Robots Restrictions",
        xaxis_title="Terms of Service Policies",
        font_size=20,
        font_style="sans-serif",
        width=400,
        height=400,
    )


def plot_robots_time_map_altair(
    df: pd.DataFrame,
    agent_type: str,
    period_col: str,
    status_col: str,
    val_col: str,
    title: str = "",
    ordered_statuses: typing.List[str] = None,
    status_colors: typing.Dict[str, str] = None,
    datetime_swap: bool = False,
    legend_cols: int = 1,
    vertical_line_dates: typing.List[str] = [],
    label_fontsize: int = 14,
    title_fontsize: int = 16,
    width: int = 1000,
    height: int = 200,
    forecast_startdate: str = None,
    legend_title: str = None,
    configure=False,
) -> alt.Chart:
    # Filter the DataFrame for the relevant agent
    filtered_df = df[df["agent"] == agent_type]
    return plot_temporal_area_map_altair(
        filtered_df,
        period_col=period_col,
        status_col=status_col,
        val_col=val_col,
        title=title,
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
        datetime_swap=datetime_swap,
        legend_cols=legend_cols,
        vertical_line_dates=vertical_line_dates,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
        width=width,
        height=height,
        forecast_startdate=forecast_startdate,
        legend_title=legend_title,
        configure=configure,
    )


def plot_robots_time_map_altair_detailed(
    df: pd.DataFrame,
    agent_type: str,
    period_col: str,
    status_col: str,
    val_col: str,
    title: str = "",
    ordered_statuses: typing.List[str] = None,
    status_colors: typing.Dict[str, str] = None,
    datetime_swap: bool = False,
    legend_cols: int = 1,
    vertical_line_dates: typing.List[str] = [],
    label_fontsize: int = 14,
    title_fontsize: int = 16,
    width: int = 1000,
    height: int = 200,
    forecast_startdate: str = None,
    configure: bool = True,
    legend_title: str = None,
) -> alt.Chart:
    # Filter the DataFrame for the relevant agent
    filtered_df = df[df["agent"] == agent_type]

    # Group by 'period' and 'status', and sum up the 'count'
    grouped_df = (
        filtered_df.groupby([period_col, status_col])[val_col]
        .sum()
        .unstack(fill_value=0)
    )

    # Ensure all required statuses are present in the DataFrame
    required_statuses = [
        "no_robots",
        "none",
        "none_sitemap",
        "none_crawl_delay",
        "some_other",
        "some_disallow_important_dir",
        "some_disallow_file_types",
        "some_pattern_restrictions",
        "all",
    ]
    missing_statuses = set(required_statuses) - set(grouped_df.columns)
    for status in missing_statuses:
        grouped_df[status] = 0

    # Reorder the columns as desired
    if ordered_statuses is None:
        ordered_statuses = required_statuses
    grouped_df = grouped_df[ordered_statuses]

    # Calculate the total counts for each period
    total_counts = grouped_df.sum(axis=1)

    # Calculate the percentage of each status per period
    percent_df = grouped_df.div(total_counts, axis=0).reset_index()

    # Handle datetime_swap parameter
    if datetime_swap:
        percent_df[period_col] = pd.to_datetime(percent_df[period_col])
    else:
        percent_df[period_col] = percent_df[period_col].dt.to_timestamp()

    # Convert to long format for Altair
    percent_long_df = percent_df.melt(
        id_vars=period_col, var_name=status_col, value_name="percentage"
    )

    # Create the chart using the general plotting function
    chart = visualization_util.create_stacked_area_chart(
        df=percent_long_df,
        period_col=period_col,
        status_col=status_col,
        percentage_col="percentage",
        title=title,
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
        legend_cols=legend_cols,
        vertical_line_dates=vertical_line_dates,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
        width=width,
        height=height,
        forecast_startdate=forecast_startdate,
        configure=configure,
        legend_title=legend_title,
    )

    return chart


def plot_temporal_area_map_altair(
    df: pd.DataFrame,
    period_col: str,
    status_col: str,
    val_col: str,
    title: str = "",
    ordered_statuses: typing.List[str] = None,
    status_colors: typing.Dict[str, str] = None,
    datetime_swap: bool = False,
    legend_cols: int = 1,
    vertical_line_dates: typing.List[str] = [],
    label_fontsize: int = 14,
    title_fontsize: int = 16,
    width: int = 1000,
    height: int = 200,
    forecast_startdate: str = None,
    **plot_kwargs,
):
    # Group by 'period' and 'status', and sum up the 'count'
    grouped_df = (
        df.groupby([period_col, status_col])[val_col].sum().unstack(fill_value=0)
    )

    # Reorder the columns as desired
    if ordered_statuses is None:
        ordered_statuses = grouped_df.columns.tolist()

    grouped_df = grouped_df[ordered_statuses]

    # Calculate the total counts for each period
    total_counts = grouped_df.sum(axis=1)

    # Calculate the percentage of each status per period
    percent_df = grouped_df.div(total_counts, axis=0).reset_index()

    if datetime_swap:
        percent_df[period_col] = pd.to_datetime(percent_df[period_col])
    else:
        percent_df[period_col] = percent_df[period_col].dt.to_timestamp()

    # Convert to long format for Altair
    percent_long_df = percent_df.melt(
        id_vars=period_col, var_name=status_col, value_name="percentage"
    )

    # Create the chart using the general plotting function
    chart = visualization_util.create_stacked_area_chart(
        df=percent_long_df,
        period_col=period_col,
        status_col=status_col,
        percentage_col="percentage",
        title=title,
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
        legend_cols=legend_cols,
        vertical_line_dates=vertical_line_dates,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
        width=width,
        height=height,
        forecast_startdate=forecast_startdate,
        **plot_kwargs,
    )

    return chart


############################################################
###### Plotly Code
############################################################


def plot_robots_heat_map_plotly(
    filled_status_summary, agent_groups_to_track, val_key="count"
):
    rows = []
    for period, agent_dict in filled_status_summary.items():
        for agent, status_dict in agent_dict.items():
            if agent in agent_groups_to_track:
                for status, url_set in status_dict.items():
                    rows.append(
                        {
                            "period": period,
                            "agent": agent,
                            "status": status,
                            val_key: len(url_set),
                        }
                    )
    df = pd.DataFrame(rows)

    filtered_df = df[df["agent"].isin(agent_groups_to_track)]

    grouped_df = (
        filtered_df.groupby(["period", "status", "agent"])[val_key]
        .sum()
        .unstack([1, 2], fill_value=0)
    )

    ordered_statuses = ["no_robots", "none", "some", "all"]
    ordered_agents = sorted(grouped_df.columns.levels[1])
    ordered_columns = list(itertools.product(ordered_statuses, ordered_agents))
    grouped_df = grouped_df[ordered_columns]
    total_counts = grouped_df.sum(axis=1)
    percent_df = grouped_df.div(total_counts, axis=0) * 100
    percent_df.index = percent_df.index.astype(str)

    traces = []
    for status, agent in ordered_columns:
        trace = go.Scatter(
            x=percent_df.index,
            y=percent_df[status, agent],
            mode="lines",
            name=f"{agent} - {status}",
            stackgroup="one",
            groupnorm="percent",
            line=dict(width=0),
            fillcolor=None,
            hovertemplate="%{y:.2f}%<extra></extra>",
        )
        traces.append(trace)

    fig = go.Figure(
        data=go.Heatmap(
            z=percent_df.values,
            x=percent_df.columns.get_level_values(1),
            y=percent_df.index,
            colorscale="Viridis",
            colorbar=dict(title="Percentage", ticksuffix="%"),
            hovertemplate="Agent: %{x}<br>Period: %{y}<br>Percentage: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Restriction Status across Agents", font=dict(size=24)),
        xaxis=dict(title="Agent", tickfont=dict(size=14), tickangle=45),
        yaxis=dict(title="Period", tickfont=dict(size=14), autorange="reversed"),
        plot_bgcolor="white",
        height=600,
        width=800,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    # Create the figure and display the plot
    # fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_robots_time_map_subplot_plotly(df, agent_types, val_key, frequency="M"):
    num_agents = len(agent_types)
    cols = 2  # Number of columns
    rows = (num_agents + 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=[f"{agent}" for agent in agent_types],
    )

    colors = ["#505050", "#0D47A1", "#FF6F00", "#B71C1C"]

    for index, agent_type in enumerate(agent_types):
        row = (index // cols) + 1
        col = (index % cols) + 1

        filtered_df = df[df["agent"] == agent_type]
        grouped_df = (
            filtered_df.groupby(["period", "status"])[val_key]
            .sum()
            .unstack(fill_value=0)
        )
        ordered_statuses = ["no_robots", "none", "some", "all"]
        grouped_df = grouped_df[ordered_statuses]
        total_counts = grouped_df.sum(axis=1)
        percent_df = grouped_df.div(total_counts, axis=0) * 100
        percent_df.index = percent_df.index.astype(str)

        for j, status in enumerate(ordered_statuses):
            trace = go.Scatter(
                x=percent_df.index,
                y=percent_df[status],
                mode="lines",
                name=status,
                stackgroup="one",
                groupnorm="percent",
                line=dict(width=0),
                fillcolor=colors[j],
                hovertemplate="%{y:.2f}%<extra></extra>",
                showlegend=(index == 0),
            )
            fig.add_trace(trace, row=row, col=col)

        fig.update_xaxes(title_text="Period", row=row, col=col, tickfont=dict(size=10))

    if frequency == "M":
        freq = "Monthly"
    elif frequency == "Y":
        freq = "Annual"

    fig.update_layout(
        title=f"Restriction Status for Different Agents over Dolma Head Sample ({freq})",
        height=1200,
        width=1000,
        plot_bgcolor="white",
        hovermode="x unified",
        hoverlabel=dict(font=dict(size=10)),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    fig.update_yaxes(title_text="Percentage", tickfont=dict(size=10))

    fig.show()


def plot_robots_time_map_matplotlib(df, agent_type, val_key):
    filtered_df = df[df["agent"] == agent_type]

    grouped_df = (
        filtered_df.groupby(["period", "status"])[val_key].sum().unstack(fill_value=0)
    )

    ordered_statuses = ["no_robots", "none", "some", "all"]
    grouped_df = grouped_df[ordered_statuses]

    total_counts = grouped_df.sum(axis=1)

    percent_df = grouped_df.div(total_counts, axis=0) * 100

    colors = ["#505050", "#0D47A1", "#FF6F00", "#B71C1C"]

    percent_df.columns = [
        "No Robots",
        "No Restrictions",
        "Some Restrictions",
        "Full Restrictions",
    ]

    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(12, 8))

    percent_df.plot(kind="area", stacked=True, color=colors, ax=ax)

    ax.set_title(
        f"Restriction Status for {agent_type} over Dolma Head Sample",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xlabel("Period", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    ax.legend(
        title="Status",
        fontsize=12,
        title_fontsize=14,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    plt.show()
    plt.clf()


def plot_robots_time_map_facet_heatmap(
    filled_status_summary, agent_groups_to_track, val_key="count"
):
    rows = []
    for period, agent_dict in filled_status_summary.items():
        for agent, status_dict in agent_dict.items():
            if agent in agent_groups_to_track:
                for status, url_set in status_dict.items():
                    rows.append(
                        {
                            "period": str(period),
                            "agent": agent,
                            "status": status,
                            val_key: len(url_set),
                        }
                    )
    df = pd.DataFrame(rows)

    filtered_df = df[df["agent"].isin(agent_groups_to_track)]

    filtered_df["period"] = pd.to_datetime(filtered_df["period"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["period"])
    filtered_df = filtered_df[filtered_df["period"] <= pd.to_datetime("2024-04-30")]
    filtered_df["period"] = filtered_df["period"].dt.strftime("%Y-%m")

    filtered_df = filtered_df.sort_values(by="period")

    filtered_df["year"] = filtered_df["period"].str[:4]

    months_per_year = filtered_df.groupby("year")["period"].nunique().reset_index()
    months_per_year.columns = ["year", "months"]

    normalized_df = filtered_df.merge(months_per_year, on="year")

    normalized_df["normalized_count"] = normalized_df[val_key] / normalized_df["months"]

    pivot_df = normalized_df.pivot_table(
        index=["period", "status"],
        columns="agent",
        values="normalized_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot_df = pd.melt(
        pivot_df, id_vars=["period", "status"], var_name="agent", value_name="count"
    )

    pivot_df["period"] = pivot_df["period"].astype(str)
    pivot_df["status"] = pivot_df["status"].astype(str)
    pivot_df["agent"] = pivot_df["agent"].astype(str)

    fig = px.density_heatmap(
        pivot_df,
        x="period",
        y="agent",
        z="count",
        facet_col="status",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        title="Restriction Status across Agents over Time",
        xaxis_title="Period",
        yaxis_title="Agent",
        height=800,
        width=1200,
        margin=dict(l=60, r=60, t=80, b=60),
        coloraxis_colorbar=dict(title="Counts"),
    )

    fig.show()


def plot_robots_time_map_3d_surface_matplotlib(
    filled_status_summary, agent_groups_to_track, val_key="count"
):
    rows = []
    for period, agent_dict in filled_status_summary.items():
        for agent, status_dict in agent_dict.items():
            if agent in agent_groups_to_track:
                for status, url_set in status_dict.items():
                    rows.append(
                        {
                            "period": str(period),
                            "agent": agent,
                            "status": status,
                            val_key: len(url_set),
                        }
                    )
    df = pd.DataFrame(rows)

    filtered_df = df[df["agent"].isin(agent_groups_to_track)]
    filtered_df["period"] = pd.to_datetime(filtered_df["period"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["period"])
    filtered_df = filtered_df[filtered_df["period"] <= pd.to_datetime("2024-04-30")]
    filtered_df["year"] = filtered_df["period"].dt.year.astype(str)

    months_per_year = filtered_df.groupby("year")["period"].nunique().reset_index()
    months_per_year.columns = ["year", "months"]
    normalized_df = filtered_df.merge(months_per_year, on="year")
    normalized_df["normalized_count"] = normalized_df[val_key] / normalized_df["months"]

    pivot_df = normalized_df.pivot_table(
        index=["agent", "year"],
        columns="status",
        values="normalized_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    pivot_df["agent_code"] = pivot_df["agent"].astype("category").cat.codes
    pivot_df["year_code"] = pivot_df["year"].astype("category").cat.codes

    years = pivot_df["year_code"].unique()
    agents = pivot_df["agent_code"].unique()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    status_labels = pivot_df.columns[2:]
    for status in status_labels:
        z_values = pivot_df.pivot(
            index="agent_code", columns="year_code", values=status
        ).values

        X, Y = np.meshgrid(years, agents)
        Z = z_values

        surf = ax.plot_surface(
            X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.7, label=status
        )

    ax.set_title("Restriction Status across Agents over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Agent")
    ax.set_zlabel("Normalized Count")
    ax.set_xticks(years)
    ax.set_xticklabels(pivot_df["year"].unique(), rotation=45, ha="right")
    ax.set_yticks(agents)
    ax.set_yticklabels(pivot_df["agent"].unique())

    ax.legend(status_labels)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()
    plt.clf()


def prepare_temporal_robots_for_corpus(
    url_robots_summary_detailed,
    head_urls,
    random_urls,
    service_to_urls,
    url_to_counts,
    agent_groups_to_track,
    robots_strictness_order,
    temporal_start_date,
    temporal_end_date,
    website_start_dates,
):
    urlsubset_to_robots_summary, urlsubsets = {}, {}
    for key, url_subset in service_to_urls.items():
        url_robots_summary_rand_detailed = {
            url: url_robots_summary_detailed[url]
            for url in random_urls
            if url in url_robots_summary_detailed
        }
        url_robots_summary_rand_subset = {
            url: details
            for url, details in url_robots_summary_rand_detailed.items()
            if url in url_subset
        }
        url_robots_summary_head_detailed = {
            url: url_robots_summary_detailed[url]
            for url in head_urls
            if url in url_robots_summary_detailed
        }
        url_robots_summary_head_subset = {
            url: details
            for url, details in url_robots_summary_head_detailed.items()
            if url in url_subset
        }
        # RANDOM
        # {Period --> Agent --> Status --> set(URLs)}
        robots_filled_status_rand_summary_detailed = (
            prepare_robots_temporal_summary_detailed(
                url_robots_summary=url_robots_summary_rand_subset,
                group_to_agents=agent_groups_to_track,
                start_time=temporal_start_date,
                end_time=temporal_end_date,
                time_frequency="M",
                website_start_dates=website_start_dates,
            )
        )
        # [Period, Agent, Status, Count, Tokens, URLs]
        robots_temporal_rand_summary_detailed = robots_temporal_to_df(
            robots_filled_status_rand_summary_detailed,
            strictness_order=robots_strictness_order,
            url_to_counts=url_to_counts,
        )
        # HEAD
        # {Period --> Agent --> Status --> set(URLs)}
        robots_filled_status_head_summary_detailed = (
            prepare_robots_temporal_summary_detailed(
                url_robots_summary=url_robots_summary_head_subset,
                group_to_agents=agent_groups_to_track,
                start_time=temporal_start_date,
                end_time=temporal_end_date,
                time_frequency="M",
                website_start_dates=website_start_dates,
            )
        )
        # [Period, Agent, Status, Count, Tokens, URLs]
        robots_temporal_head_summary_detailed = robots_temporal_to_df(
            robots_filled_status_head_summary_detailed,
            strictness_order=robots_strictness_order,
            url_to_counts=url_to_counts,
        )
        urlsubset_to_robots_summary[f"rand-{key}"] = (
            robots_temporal_rand_summary_detailed
        )
        urlsubset_to_robots_summary[f"head-{key}"] = (
            robots_temporal_head_summary_detailed
        )
        urlsubsets[f"rand-{key}"] = set(url_robots_summary_rand_subset.keys())
        urlsubsets[f"head-{key}"] = set(url_robots_summary_head_subset.keys())

        # if key == "all":
        # print(len(url_robots_summary_head_detailed))
        # print(len(url_robots_summary_head_subset))
        # target_head_summary_df = robots_temporal_head_summary_detailed
    return urlsubset_to_robots_summary, urlsubsets


def prepare_temporal_tos_for_corpus(
    tos_policies,
    tos_license_policies,
    tos_compete_policies,
    head_urls,
    random_urls,
    service_to_urls,
    url_to_counts,
    agent_groups_to_track,
    temporal_start_date,
    temporal_end_date,
    manual_annotated_urls,
    website_start_dates,
):
    urlsubset_to_tos_summary, urlsubsets = {}, {}
    url_to_time_to_tos_verdict = get_tos_url_time_verdicts(
        tos_policies,
        tos_license_policies,
        tos_compete_policies,
        manual_annotated_urls,
        website_start_dates,
    )
    for key, url_subset in service_to_urls.items():
        # if key != "all":
        #     continue

        url_tos_summary_rand_detailed = {
            url: url_to_time_to_tos_verdict[url]
            for url in random_urls
            if url in url_to_time_to_tos_verdict
        }
        url_tos_summary_rand_subset = {
            url: details
            for url, details in url_tos_summary_rand_detailed.items()
            if url in url_subset
        }
        url_tos_summary_head_detailed = {
            url: url_to_time_to_tos_verdict[url]
            for url in head_urls
            if url in url_to_time_to_tos_verdict
        }
        url_tos_summary_head_subset = {
            url: details
            for url, details in url_tos_summary_head_detailed.items()
            if url in url_subset
        }
        # print(len(url_tos_summary_head_subset))
        # print(len(url_tos_summary_rand_subset))

        # Period --> Status --> set(URLs)
        period_tos_verdict_urls_head = prepare_tos_temporal_summary(
            url_tos_summary_head_subset,
            start_time=temporal_start_date,
            end_time=temporal_end_date,
            time_frequency="M",
            website_start_dates=website_start_dates,
        )
        # cc_head_urls = set()
        # for k, vs in period_tos_verdict_urls_head.items():
        #     for urls in vs.values():
        #         cc_head_urls = cc_head_urls.union(urls)
        # print(len(cc_head_urls))

        # Period --> Status --> set(URLs)
        period_tos_verdict_urls_rand = prepare_tos_temporal_summary(
            url_tos_summary_rand_subset,
            start_time=temporal_start_date,
            end_time=temporal_end_date,
            time_frequency="M",
            website_start_dates=website_start_dates,
        )
        # cc_rand_urls = set()
        # for k, vs in period_tos_verdict_urls_rand.items():
        #     for urls in vs.values():
        #         cc_rand_urls = cc_rand_urls.union(urls)
        # print(len(cc_rand_urls))

        # Dataframe: [Period, Status, Count, Tokens]
        tos_summary_df_head = tos_temporal_to_df(
            period_tos_verdict_urls_head,
            url_set=url_subset,
            url_to_counts=url_to_counts,
        )
        tos_summary_df_rand = tos_temporal_to_df(
            period_tos_verdict_urls_rand,
            url_set=url_subset,
            url_to_counts=url_to_counts,
        )

        urlsubset_to_tos_summary[f"head-{key}"] = tos_summary_df_head
        urlsubset_to_tos_summary[f"rand-{key}"] = tos_summary_df_rand
        urlsubsets[f"head-{key}"] = set(url_tos_summary_head_subset.keys())
        urlsubsets[f"rand-{key}"] = set(url_tos_summary_rand_subset.keys())
    return urlsubset_to_tos_summary, urlsubsets


def compute_corpus_restriction_estimates(
    head_df,
    rand_df,
    head_token_frac,
    rand_token_frac,
    target_agent,
    restrictive_statuses,
    save_fpath,
    verbose=False,
):
    # Get the minimum and maximum date from the 'period' column
    # rand_df['period'] = rand_df['period'].dt.to_timestamp()
    start_period = rand_df["period"].min()
    end_period = rand_df["period"].max()
    all_periods = pd.period_range(start=start_period, end=end_period, freq="M")

    # print(rand_df[rand_df['agent'] == target_agent][rand_df["period"] == "2024-04"])
    rand_restricted = rand_df.loc[rand_df["agent"] == target_agent].loc[
        rand_df["status"].isin(restrictive_statuses)
    ][["period", "tokens"]]
    rand_restricted = rand_restricted.groupby("period").sum()["tokens"]
    rand_restricted = rand_restricted.reindex(all_periods, fill_value=0)

    rand_all = rand_df.loc[rand_df["agent"] == target_agent][["period", "tokens"]]
    rand_all = rand_all.groupby("period").sum()["tokens"]

    head_restricted = head_df.loc[head_df["agent"] == target_agent].loc[
        head_df["status"].isin(restrictive_statuses)
    ][["period", "tokens", "count"]]
    if verbose:
        print(head_restricted[head_restricted["period"] == "2024-04"])
    head_restricted = head_restricted.groupby("period").sum()["tokens"]
    head_restricted = head_restricted.reindex(all_periods, fill_value=0)

    head_all = head_df.loc[head_df["agent"] == target_agent][["period", "tokens"]]
    head_all = head_all.groupby("period").sum()["tokens"]

    assert (rand_restricted.index == head_restricted.index).all()
    assert (rand_all.index == head_all.index).all()
    assert (rand_restricted.index == head_all.index).all()
    assert rand_restricted.index.is_unique
    assert head_restricted.index.is_unique
    assert rand_all.index.is_unique
    assert head_all.index.is_unique

    # print(len(head_restricted[-1]))
    rand_frac = rand_restricted / rand_all
    head_frac = head_restricted / head_all
    # print(save_fpath)
    # print(f"rand token frac = {rand_token_frac}")
    # print(f"head token frac = {head_token_frac}")
    out = pd.concat(
        [
            rand_frac.rename("Random"),
            head_frac.rename("Head"),
            (rand_token_frac * rand_frac).rename("Rand Portion"),
            (head_token_frac * head_frac).rename("Head Portion"),
            ((rand_token_frac * rand_frac) + (head_token_frac * head_frac)).rename(
                "Full Corpus"
            ),
        ],
        axis=1,
    )
    if verbose:
        print(head_frac[-1])
        print(rand_frac[-1])
        print(head_token_frac * head_frac[-1])
        print(rand_token_frac * rand_frac[-1])
        print((rand_token_frac * rand_frac[-1]) + (head_token_frac * head_frac[-1]))

    # out.to_csv(f'src/analysis/{CHOSEN_CORPUS}_total_token_estimates.csv', index=True)
    out.to_csv(save_fpath, index=True)


def generate_corpus_restriction_estimates_per_url_split(
    urlsubset_to_robots_summary,
    url_subsets,
    target_corpus,
    url_token_lookup,
    target_agent,
    robot_statuses_to_include,
    save_dir="output_data_robots",
):

    total_tokens = url_token_lookup._TOTAL_TOKENS[target_corpus.lower()]
    top_corpus_urls = url_token_lookup.top_k_urls(target_corpus.lower(), 2000)
    corpus_urls_to_counts = url_token_lookup.get_url_to_token_map(target_corpus.lower())
    num_head_tokens = sum(
        [v for k, v in corpus_urls_to_counts.items() if k in top_corpus_urls]
    )
    non_head_tokens = total_tokens - num_head_tokens
    assert non_head_tokens >= 0

    num_tokens_splits = {}
    service_keys = set([k.split("-")[-1] for k in urlsubset_to_robots_summary.keys()])
    for key in service_keys:
        url_keys = list(urlsubset_to_robots_summary[f"head-{key}"].keys())
        num_head_service_tokens = sum(
            [
                v
                for k, v in corpus_urls_to_counts.items()
                if k in url_subsets[f"head-{key}"]
            ]
        )
        num_rand_service_tokens = sum(
            [
                v
                for k, v in corpus_urls_to_counts.items()
                if k in url_subsets[f"rand-{key}"]
            ]
        )
        num_tokens_splits[key] = {
            "head": num_head_service_tokens / num_head_tokens,
            "rand": num_rand_service_tokens / non_head_tokens,
        }
    # overwrite "all".
    num_tokens_splits["all"] = {
        "head": num_head_tokens / total_tokens,
        "rand": non_head_tokens / total_tokens,
    }

    for i, splitkey in enumerate(service_keys):
        # print(splitkey)
        save_fpath = os.path.join(
            save_dir, f"{target_corpus}_{splitkey}_token_estimates.csv"
        )
        head_df = urlsubset_to_robots_summary[f"head-{splitkey}"]
        rand_df = urlsubset_to_robots_summary[f"rand-{splitkey}"]
        # if splitkey == "all":
        #     print(rand_df[rand_df['period'] == "2024-04"])
        if "agent" not in head_df:
            head_df["agent"] = target_agent
            rand_df["agent"] = target_agent
        compute_corpus_restriction_estimates(
            head_df,
            rand_df,
            num_tokens_splits[splitkey]["head"],
            num_tokens_splits[splitkey]["rand"],
            target_agent,
            robot_statuses_to_include,
            save_fpath,
            # verbose=splitkey=="all",
        )


def save_robots_agent_statistics(
    agents_to_track: set,
    url_robots_summary: dict,  # Can be either dict or DataFrame
    save_path: str,
    legend_mapping: dict,
):
    """
    Save detailed statistics for each agent including domain-level information over time.
    Saves data in both JSON and JSONL formats for efficient processing.

    Args:
        agents_to_track: Set of individual agents to track
        url_robots_summary: Either a dictionary mapping URLs to their parsed robots.txt data
                          or a DataFrame with temporal robots data
        save_path: Base path for saving files (will append .json/.jsonl)
        legend_mapping: Dictionary mapping raw statuses to display names
        corpus_name: Name of the corpus (c4, rf, dolma)
    """
    json_path = f"{save_path}agent_robots_statistics.json"
    jsonl_path = f"{save_path}agent_robots_statistics.jsonl"

    json_records = []

    # Handle DataFrame input
    if isinstance(url_robots_summary, dict) and any(
        isinstance(v, pd.DataFrame) for v in url_robots_summary.values()
    ):
        # Process each split (head/rand) separately
        for split_name, df in url_robots_summary.items():
            # Group by period and agent to get status distribution
            grouped = (
                df.groupby(["period", "agent", "status"])
                .agg({"tokens": "sum", "count": "sum"})
                .reset_index()
            )

            for period in grouped["period"].unique():
                period_data = grouped[grouped["period"] == period]
                record = {
                    "split": split_name,
                    "period": (
                        period.strftime("%Y-%m-%d")
                        if isinstance(period, pd.Timestamp)
                        else str(period)
                    ),
                    "agents": {},
                }

                for agent in agents_to_track:
                    agent_data = period_data[period_data["agent"] == agent]
                    if not agent_data.empty:
                        record["agents"][agent] = {
                            "statuses": {
                                row["status"]: {
                                    "tokens": row["tokens"],
                                    "count": row["count"],
                                }
                                for _, row in agent_data.iterrows()
                            }
                        }

                json_records.append(record)

    # Handle original dictionary input
    else:
        for url, date_to_statuses in url_robots_summary.items():
            url_record = {"domain": url, "timestamps": {}}

            for date_str, agent_statuses in date_to_statuses.items():
                try:
                    date = pd.to_datetime(date_str)
                    date_str = date.strftime("%Y-%m-%d")
                    url_record["timestamps"][date_str] = {}

                    agent_statuses_lower = {
                        k.lower(): v for k, v in agent_statuses.items()
                    }
                    wildcard_status = agent_statuses_lower.get("*", "none")

                    for agent in agents_to_track:
                        raw_status = agent_statuses_lower.get(
                            agent.lower(), wildcard_status
                        )
                        status = legend_mapping.get(raw_status, raw_status)
                        url_record["timestamps"][date_str][agent] = status

                except Exception as e:
                    print(f"Error processing {date_str}: {str(e)}")
                    continue

            json_records.append(url_record)

    # Save JSON format (one record per line for easier processing)
    with open(jsonl_path, "w") as f:
        for record in json_records:
            f.write(json.dumps(record) + "\n")

    # Save full JSON
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)


def save_tos_agent_statistics(
    url_to_time_to_tos_verdict: dict,  # URL -> time -> ToS verdict mapping
    save_path: str,
):
    """Save ToS statistics to JSON and JSONL files in the same format as robots data.

    Args:
        url_to_time_to_tos_verdict: Dictionary mapping URLs to their temporal ToS verdicts
        save_path: Base path to save the JSON/JSONL files (without extension)
    """
    json_path = f"{save_path}agent_tos_statistics.json"
    jsonl_path = f"{save_path}agent_tos_statistics.jsonl"

    json_records = []

    # Process each URL's temporal data
    for url, time_to_verdict in url_to_time_to_tos_verdict.items():
        url_record = {"domain": url, "timestamps": {}}

        for date_str, verdict in time_to_verdict.items():
            try:
                date = pd.to_datetime(date_str)
                date_str = date.strftime("%Y-%m-%d")
                url_record["timestamps"][date_str] = verdict

            except Exception as e:
                print(f"Error processing {date_str}: {str(e)}")
                continue

        json_records.append(url_record)

    # Save JSON format (one record per line for easier processing)
    with open(jsonl_path, "w") as f:
        for record in json_records:
            f.write(json.dumps(record) + "\n")

    # Save full JSON
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)
