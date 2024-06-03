from collections import defaultdict, Counter
import re
import itertools

import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from urllib.parse import urlparse

from . import parse_robots
from analysis import visualization_util, analysis_constants

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
    "Cohere": {
        "train": ["cohere-ai"],
        "retrieval": ["cohere-ai"]
    },
    "Meta": {
        "train": ["FacebookBot"],
        "retrieval": ["FacebookBot"],
        # https://developers.facebook.com/docs/sharing/bot/
    },
    "Internet Archive": {
        "train": ["ia_archiver"], 
        "retrieval": ["ia_archiver"]
    },
    "Google Search": {
        "train": ["Googlebot"],
        "retrieval": ["Googlebot"],
    },
    # "Perplexity AI": {
    #     "train": ["PerplexityBot"],
    #     "retrieval": ["PerplexityBot"]
    #     # https://docs.perplexity.ai/docs/perplexitybot
    # },
    # "Microsoft": {
    #     "train": [],
    #     "retrieval": ["Bingbot"]
    # },
}

def get_bot_groups(groups=BOT_TRACKER.keys()):
    ret = {}
    for group in groups:
        ret[group] = list(set(BOT_TRACKER[group]["train"] + BOT_TRACKER[group]["retrieval"]))
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
        # index_map = {'c4': 0, 'rf': 1, 'dolma': 2}
        # dataset_index = index_map[dataset_name]
        # total = 0
        # for tokens in self.lookup_map.values():
        #     total += tokens[dataset_index]
        # return total

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
        top_urls = [url for url, tokens in sorted_urls[:k]]
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
        return [url for url in self.lookup_map if url not in top_urls]

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
        some_categories = [status for status in agent_statuses if status.startswith("some")]
        if "some_pattern_restrictions" in some_categories:
            return "some_pattern_restrictions"
        elif "some_disallow_file_types" in some_categories:
            return "some_disallow_file_types"
        elif "some_disallow_important_dir" in some_categories:
            return "some_disallow_important_dir"
        else:
            return "some_other"
    elif any(status.startswith("none") for status in agent_statuses):
        none_categories = [status for status in agent_statuses if status.startswith("none")]
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
    # Status summary to be returned (only for relevant agents)
    status_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    
    # Counter to track statuses for all agents
    agent_counter = Counter()
    status_counter = defaultdict(lambda: Counter({'all': 0, 'none': 0, 'some': 0}))

    for url, date_to_robots in data.items():
        if None in date_to_robots:
            print(url)
        _, parsed_result = parse_robots.analyze_robots(date_to_robots)
        
        for date_str, agent_to_status in parsed_result.items():
            date = pd.to_datetime(date_str)
            for agent in relevant_agents:
                status = agent_to_status.get(agent, agent_to_status.get("*", "none"))
                status_summary[url][date][agent] = status
            for agent, status in agent_to_status.items():
                # Update counters for all agents
                agent_counter[agent] += 1
                if status == "all":
                    status_counter[agent]['all'] += 1
                elif status == "none":
                    status_counter[agent]['none'] += 1
                else:
                    status_counter[agent]['some'] += 1

    # Create DataFrame from the status counters
    agent_counter_df = pd.DataFrame(
        [(agent, agent_counter[agent], counts['all'], counts['some'], counts['none'])
         for agent, counts in status_counter.items()],
        columns=['agent', 'observed', 'all', 'some', 'none']
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
        if None in date_to_robots:
            print(url)
        _, parsed_result = robots_stats, url_interpretations = (
            parse_robots.analyze_robots(date_to_robots)
        )
        for date_str, agent_to_status in parsed_result.items():
            date = pd.to_datetime(date_str)
            for agent in relevant_agents:
                status = agent_to_status.get(agent, agent_to_status.get("*", "none"))
                robots_txt = date_to_robots[date_str]

                if status == "some":
                    if re.search(r"Disallow:\s+/.*\?", robots_txt):
                        status = "some_pattern_restrictions"
                    elif re.search(r"Disallow:\s+\*\.(?:pdf|jpe?g|png|gif|bmp|ico|tiff?|svg)", robots_txt):
                        status = "some_disallow_file_types"
                    elif re.search(r"Disallow:\s*/(?:admin|private|confidential)", robots_txt):
                        status = "some_disallow_important_dir"
                    else:
                        status = "some_other"
                elif status == "none" or status == "*":
                    if re.search(r"Crawl-delay:", robots_txt):
                        status = "none_crawl_delay"
                    elif re.search(r"Sitemap:", robots_txt):
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
    with open(fpath, 'r') as file:
        start_dates = json.load(file)

    # Map the sanitized URLs back to the original URLs
    website_start_dates = {url: start_dates.get(sanitize_url(url), pd.to_datetime('1970-01-01')) for url in robots_urls}
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


def robots_temporal_to_df(filled_status_summary, url_to_counts={}):
    """
    Args:
        filled_status_summary: {Period --> Agent --> Status --> set(URLs)}
        url_to_counts: {url -> num tokens}. If available, will sum the tokens for each URL in counts

    Returns:
        Dataframe: [Period, Agent, Status, Count, Tokens]
    """

    # Convert results to a DataFrame for easy viewing and manipulation
    summary_df_list = []
    for period, agent_statuses in filled_status_summary.items():
        for agent, statuses in agent_statuses.items():
            for status, urls in statuses.items():
                summary_df_list.append(
                    {
                        "period": period,
                        "agent": agent,
                        "status": status,
                        "count": len(urls),
                    }
                )
                if url_to_counts:
                    summary_df_list[-1].update(
                        {"tokens": sum([url_to_counts[url] for url in urls])}
                    )

    summary_df = pd.DataFrame(summary_df_list)
    return summary_df


def get_latest_url_robot_statuses(url_robots_summary, agents):
    # {URL --> Date --> Agent --> Status}
    # URL —> status
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

def get_tos_url_time_verdicts(
    tos_policies
):
    """
    Input:
        URL --> time --> ToS subpage --> {verdict: <code>, evidence: <evidence>}

    Returns:
        URL --> time --> ToS verdict string.    
    """
    url_to_time_to_verdict = {}
    for url, time_to_subpage_to_verdicts in tos_policies.items():
        for tos_time, subpage_verdict in time_to_subpage_to_verdicts.items():
            verdict_codes = [vinfo["verdict"] for vinfo in time_to_subpage_to_verdicts[tos_time].values()]
            if url not in url_to_time_to_verdict:
                url_to_time_to_verdict[url] = {}
            url_to_time_to_verdict[url].update(
                {tos_time: analysis_constants.TOS_AI_SCRAPING_VERDICT_MAPPER[max(verdict_codes)]}
            )
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
                tos_time_keys = sorted([pd.to_datetime(date_str) for date_str in date_to_verdict.keys()])
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

def tos_temporal_to_df(
    filled_status_summary, 
    url_set, 
    url_to_counts={}
):
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
        category_names=['Full Restrictions', 'Some Restrictions', 'No Restrictions'],
        custom_colors=['#e04c71','#e0cd92','#82b5cf'],
        group_order=sorted(size_bucket_to_urls.keys(), key=lambda x: int(x.split('-')[0])), 
        total_dsets=len(url_to_bucket_key), 
        legend=True, 
        savepath=f"paper_figures/altair/robots_restrictions_vs_token_count_{agent_group}.json"
    )


def tos_get_most_recent_verdict(tos_policies):
    url_to_recent_policy = {}
    for url, time_to_subpage_to_verdicts in tos_policies.items():
        recent_key = max(time_to_subpage_to_verdicts.keys())
        verdict_codes = [vinfo["verdict"] for vinfo in time_to_subpage_to_verdicts[recent_key].values()]
        url_to_recent_policy[url] = analysis_constants.TOS_AI_SCRAPING_VERDICT_MAPPER[max(verdict_codes)]
    return url_to_recent_policy

def prepare_recent_robots_tos_info(
    tos_policies_dict,
    url_robots_summary,
    companies,
):
    agent_names = [agent for company in companies for agent in get_bots(company)]
    # {URL --> Date --> Agent --> Status} --> {URL —> status}
    url_robots_status = get_latest_url_robot_statuses(url_robots_summary, agent_names)
    print(len(url_robots_status))
    url_tos_verdicts = tos_get_most_recent_verdict(tos_policies_dict)
    return url_robots_status, url_tos_verdicts

def encode_latest_tos_robots_into_df(
    url_results_df,
    tos_policies,
    url_robots_summary,
    companies,
):

    recent_url_robots, recent_url_tos_verdicts = prepare_recent_robots_tos_info(
        tos_policies, url_robots_summary, companies
    )
    url_results_df["robots"] = url_results_df["URL"].map(recent_url_robots)
    url_results_df['robots'].fillna("none", inplace=True)
    url_results_df["Restrictive Robots.txt"] = url_results_df["robots"].map({"all": True, "some": False, "none": False})
    url_results_df["ToS"] = url_results_df["URL"].map(recent_url_tos_verdicts)
    url_results_df['ToS'].fillna('No Restrictions', inplace=True)
    tos_strictness = {v: k < 5  for k, v in analysis_constants.TOS_AI_SCRAPING_VERDICT_MAPPER.items()}
    url_results_df["Restrictive Terms"] = url_results_df["ToS"].map(tos_strictness)
    return url_results_df


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

    plt.title(f"Restriction Status for {agent_type} over 10k Random Sample [{frequency}]")
    plt.xlabel("Period")
    plt.ylabel("Percentage")
    plt.legend(title="Status")
    plt.show()
    plt.clf()


def prepare_tos_robots_confusion_matrix(
    tos_policies,
    url_robots_summary,
    companies,
    url_token_lookup,
    use_token_counts=True,
    corpora_choice="c4",
    font_size=20, 
    font_style='sans-serif',
    width=400,
    height=400,
):
    recent_url_robots, recent_tos_verdicts = prepare_recent_robots_tos_info(
        tos_policies, url_robots_summary, companies,
    )

    ROBOTS_LABELS = {
        "none": "None",
        "some": "Partial",
        "all": "Restricted",
    }
    yaxis_order = ["Restricted", "Partial", "None"]
    xaxis_order = ["No Restrictions", "Conditional Restrictions", "Prohibits AI", "Prohibits Scraping", "Prohibits Scraping & AI"]
    
    # Create a defaultdict to store counts
    counts = defaultdict(lambda: defaultdict(int))
    token_counts = defaultdict(lambda: defaultdict(int))
    
    # Count the occurrences of each (status, policy) pair
    total_instances, total_tokens = 0, 0
    url_token_counts = url_token_lookup.get_url_to_token_map(corpora_choice)
    for url in set(recent_url_robots.keys()).intersection(set(recent_tos_verdicts.keys())):
    # for url in url_to_status.keys():
        status = ROBOTS_LABELS[recent_url_robots.get(url, "none")]
        policy = recent_tos_verdicts.get(url, "No Restrictions")
        counts[status][policy] += 1
        total_instances += 1
        token_counts[status][policy] += url_token_counts[url]
        total_tokens += url_token_counts[url]
    
    # Create a list of tuples (status, policy, count)
    data = [{"Robots Restrictions": status, "Terms of Service Policies": policy, "Count": count, "Token Counts": token_counts[status][policy],
             "Percent": round(100 * count / total_instances, 2), 
             "Percent Tokens": round(100 * token_counts[status][policy] / total_tokens, 2),}
            for status in yaxis_order
            for policy in xaxis_order
            if (count := counts[status][policy]) > 0]
    
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data)
    df['Formatted Percent'] = df['Percent'].apply(lambda x: f"{x:.1f} %")
    df['Formatted Percent Tokens'] = df['Percent Tokens'].apply(lambda x: f"{x:.1f} %")
    
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
        font_style='sans-serif',
        width=400,
        height=400,
    )


def plot_robots_time_map_altair(
    df,  
    agent_type, 
    period_col, 
    status_col, 
    val_col, 
    title='', 
    ordered_statuses=None, 
    status_colors=None,
    datetime_swap=False,
):
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
    )
    
    
def plot_robots_time_map_altair_detailed(
    df,  
    agent_type, 
    period_col, 
    status_col, 
    val_col, 
    title='', 
    ordered_statuses=None, 
    status_colors=None,
    detailed=False,
):
    # Filter the DataFrame for the relevant agent
    filtered_df = df[df["agent"] == agent_type]
    
    # Group by 'period' and 'status', and sum up the 'count'
    grouped_df = filtered_df.groupby([period_col, status_col])[val_col].sum().unstack(fill_value=0)
    
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
        "all"
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
    percent_df[period_col] = percent_df[period_col].dt.to_timestamp()
    
    # Convert to long format for Altair
    percent_long_df = percent_df.melt(id_vars=period_col, var_name=status_col, value_name='percentage')
    
    # Create the chart using the general plotting function
    chart = visualization_util.create_stacked_area_chart(
        df=percent_long_df,
        period_col=period_col,
        status_col=status_col,
        percentage_col='percentage',
        title=title,
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
    )
    
    return chart
    

def plot_temporal_area_map_altair(
    df,
    period_col, 
    status_col, 
    val_col, 
    title='', 
    ordered_statuses=None, 
    status_colors=None,
    datetime_swap=False,
):
    # Group by 'period' and 'status', and sum up the 'count'
    grouped_df = df.groupby([period_col, status_col])[val_col].sum().unstack(fill_value=0)
    
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
    percent_long_df = percent_df.melt(id_vars=period_col, var_name=status_col, value_name='percentage')
    
    # Create the chart using the general plotting function
    chart = visualization_util.create_stacked_area_chart(
        df=percent_long_df,
        period_col=period_col,
        status_col=status_col,
        percentage_col='percentage',
        title=title,
        ordered_statuses=ordered_statuses,
        status_colors=status_colors
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


def plot_robots_time_map_plotly(df, agent_type, val_key):
    filtered_df = df[df["agent"] == agent_type]

    grouped_df = (
        filtered_df.groupby(["period", "status"])[val_key].sum().unstack(fill_value=0)
    )

    ordered_statuses = ["no_robots", "none", "some", "all"]
    grouped_df = grouped_df[ordered_statuses]
    total_counts = grouped_df.sum(axis=1)
    percent_df = grouped_df.div(total_counts, axis=0) * 100
    percent_df.index = percent_df.index.astype(str)

    traces = []
    for status in ordered_statuses:
        trace = go.Scatter(
            x=percent_df.index,
            y=percent_df[status],
            mode="lines",
            name=status,
            stackgroup="one",
            groupnorm="percent",
            line=dict(width=0),
            fillcolor=None,
            hovertemplate="%{y:.2f}%<extra></extra>",
        )
        traces.append(trace)

    colors = ["#505050", "#0D47A1", "#FF6F00", "#B71C1C"]
    for i, trace in enumerate(traces):
        trace.fillcolor = colors[i]

    layout = go.Layout(
        title=dict(
            text=f"Restriction Status for {agent_type} over Dolma Head Sample",
            x=0.5,
            font=dict(size=24),
        ),
        xaxis=dict(title="Period", tickfont=dict(size=14)),
        yaxis=dict(title="Percentage", tickfont=dict(size=14), ticksuffix="%"),
        legend=dict(
            title="Status",
            font=dict(size=14),
            x=1.02,
            y=1,
            borderwidth=1,
            bordercolor="#d3d3d3",
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        hoverlabel=dict(font=dict(size=14)),
        margin=dict(l=60, r=60, t=80, b=60),
        shapes=[
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=1,
                x1=1,
                y0=0,
                y1=1,
                line=dict(color="#d3d3d3", width=1),
            )
        ],
    )
    fig = go.Figure(data=traces, layout=layout)
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
    filtered_df["year"] = filtered_df["period"].dt.year.astype(
        str
    )

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

