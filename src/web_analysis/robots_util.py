import os
import sys
from datetime import datetime
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

from . import parse_robots
from analysis import analysis_util

############################################################
###### Robots.txt Bot Methods
############################################################

# Grouping bots by company and their usage
BOT_TRACKER = {
    "*All Agents*": {
        "train": ["*All Agents*"],
        "retrieval": ["*All Agents*"]
        # An aggregation of policies towards All Agents.
    },
    "*": {
        "train": ["*"],
        "retrieval": ["*"]
    },
    "OpenAI": {
        "train": ["GPTBot"],
        "retrieval": ["ChatGPT-User"]
        # https://platform.openai.com/docs/gptbot
        # https://platform.openai.com/docs/plugins/bot
    },
    "Google": {
        "train": ["Google-Extended"],
        "retrieval": ["Google-Extended"]
        # https://developers.google.com/search/docs/crawling-indexing/overview-google-crawlers
    },
    "Common Crawl": {
        "train": ["CCBot"],
        "retrieval": ["CCBot"]
        # https://commoncrawl.org/ccbot
    },
    "Amazon": {
        "train": ["Amazonbot"],
        "retrieval": ["Amazonbot"]
        # https://developer.amazon.com/amazonbot
    },
    "Anthropic False": {
        "train": ["anthropic-ai"],
        "retrieval": ["Claude-Web"]
        # https://support.anthropic.com/en/articles/8896518-does-anthropic-crawl-data-from-the-web-and-how-can-site-owners-block-the-crawler
    },
    "Anthropic": {
        "train": ["ClaudeBot", "CCBot"],
        "retrieval": ["ClaudeBot", "CCBot"]
        # https://support.anthropic.com/en/articles/8896518-does-anthropic-crawl-data-from-the-web-and-how-can-site-owners-block-the-crawler
    },
    "Cohere": {
        "train": ["cohere-ai"],
        "retrieval": ["cohere-ai"]
    },
    "Facebook": {
        "train": ["FacebookBot"],
        "retrieval": ["FacebookBot"]
        # https://developers.facebook.com/docs/sharing/bot/
    },
    "Internet Archive": {
        "train": ["ia_archiver"],
        "retrieval": ["ia_archiver"]
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

def get_bot_groups():
    ret = {}
    for group in BOT_TRACKER.keys():
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
        return set([bot for role in BOT_TRACKER.get(company, {}).values() for bot in role])
    if setting:
        # Return all bots for a specific setting across all companies
        return set([bot for company in BOT_TRACKER.values() for bot in company.get(setting, [])])
    # Return all bots
    return set([bot for company in BOT_TRACKER.values() for role in company.values() for bot in role])


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
        self.index_map = {'c4': 0, 'rf': 1, 'dolma': 2}

        self._TOTAL_TOKENS = {
            "c4": 170005451386,
            "rf": 431169198316,
            "dolma": 1974278779400,
        } 
        self._TOTAL_URLS = {
            "c4": 15928138,
            "rf": 33210738,
            "dolma": 45246789,
        } # Total URLs in common = 10136147
    
    def _create_lookup_map(self):
        """
        Create a lookup map from the CSV file.
        
        Returns:
        dict: A dictionary with URLs as keys and tuples of token counts as values.
        """
        df = pd.read_csv(self.file_path)
        # x = df.set_index('url')[['c4_tokens', 'rf_tokens', 'dolma_tokens']]
        # return {row['url']: (row['c4_tokens'], row['rf_tokens'], row['dolma_tokens']) for index, row in df.iterrows()}

        df.set_index('url', inplace=True)

        # Step 3: Convert the DataFrame to a dictionary with tuples as values
        lookup_map = df.to_dict('index')  # Converts to a dictionary of dictionaries

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
        sorted_urls = sorted(self.lookup_map.items(), key=lambda item: item[1][dataset_index], reverse=True)
        
        # Extract the top K URLs
        top_urls = [url for url, tokens in sorted_urls[:k]]
        num_tokens = sum([tokens[self.index_map[dataset_name]] for url, tokens in sorted_urls[:k]])
        if verbose:
            print(f"Number of tokens in {k} URLs: {num_tokens} | {round(100*num_tokens / self._TOTAL_TOKENS[dataset_name], 2)}% of {dataset_name}")
        return top_urls

    def get_10k_random_sample(self):
        top_urls = self.top_k_urls("c4", 2000, False) + self.top_k_urls("rf", 2000, False) + self.top_k_urls("dolma", 2000, False)
        return [url for url in self.lookup_map if url not in top_urls]


    def get_url_to_token_map(self, dataset_name):
        dataset_index = self.index_map[dataset_name]
        dataset_lookup = {k: v[dataset_index] for k, v in self.lookup_map.items()}
        return dataset_lookup




def agent_and_operation(agent_statuses):
    """Given a list of agent blocks: all, some, none, and N/A,
    return the strictest designation."""
    if "all" in agent_statuses:
        return "all"
    elif "some" in agent_statuses:
        return "some"
    elif "none" in agent_statuses:
        return "none"
    return agent_statuses[0]

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
        
        if (key_date <= target_date if direction == "backward" else key_date >= target_date):
            if closest_key is None or compare(key_date, closest_key.to_pydatetime().date()):
                closest_key = key
    return closest_key

def compute_url_date_agent_status(data, relevant_agents):
    """
    Args: 
        data: {URL --> Date --> robots.txt raw text}
        relevant_agents: List of agent names to track

    Returns: {URL --> Date --> Agent --> Status}
    """
    # Convert date strings to pandas datetime for easier manipulation
    # {URL --> Date --> Agent --> Status}
    # all_statuses = []
    status_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    for url, date_to_robots in data.items():
        if None in date_to_robots:
            print(url)
        _, parsed_result = robots_stats, url_interpretations = parse_robots.analyze_robots(date_to_robots)
        for date_str, agent_to_status in parsed_result.items():
            date = pd.to_datetime(date_str)
            for agent in relevant_agents:
                status = agent_to_status.get(agent, agent_to_status.get("*", "none"))
                status_summary[url][date][agent] = status
                # all_statuses.append(status)
    # print(set(all_statuses))
    return status_summary


def prepare_robots_temporal_summary(
    url_robots_summary, 
    group_to_agents, 
    start_time, 
    end_time, 
    time_frequency="M",
):
    """
    Fill in the missing weeks for each URL.

    Args:
        url_robots_summary: {URL --> Date --> Agent --> Status}
        group_to_agents: {group_name --> [agents]}
        start_time: YYYY-MM-DD
        end_time: YYYY-MM-DD
        time_frequency: "M" = Monthly, "W" = Weekly.

    Intermediate Prepares: 
        {Period --> Agent --> Status --> set(URLs)}

    If url_to_counts then Returns:
        {Period --> Agent --> Status --> total_tokens(URLs)}
    Else Returns:
        {Period --> Agent --> Status -->  count(URLs)}
    """
    relevant_agents = [v for vs in group_to_agents.values() for v in vs]
    start_date = pd.to_datetime(start_time)
    end_date = pd.to_datetime(end_time)
    date_range = pd.period_range(start_date, end_date, freq=time_frequency)
    # {Period --> Agent --> Status --> set(URLs)}
    filled_status_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for wi, period in enumerate(date_range):
        if wi % 10 == 0:
            print(period)
        # for agent in relevant_agents:
        for url, date_agent_status in url_robots_summary.items():
            robots_time_keys = sorted(list(date_agent_status.keys()))
            time_key = find_closest_time_key(robots_time_keys, period, direction="backward")
            for group, agents in group_to_agents.items():
                statuses = ["N/A" if time_key is None else date_agent_status[time_key][agent] for agent in agents]
                group_status = agent_and_operation(statuses)
                filled_status_summary[period][group][group_status].add(url)

    return filled_status_summary

def robots_temporal_to_df(
    filled_status_summary, 
    url_to_counts={}
):
    """
    Args: 
        filled_status_summary: {Period --> Agent --> Status --> set(URLs)}
        url_to_counts: {url -> num tokens}. If available, will sum the tokens for each URL in counts

    If url_to_counts then Returns:
        {Period --> Agent --> Status --> total_tokens(URLs)}
    Returns:
        Dataframe: [Period, Agent, Status, Count, Tokens]
    """

    # Convert results to a DataFrame for easy viewing and manipulation
    summary_df_list = []
    for period, agent_statuses in filled_status_summary.items():
        for agent, statuses in agent_statuses.items():
            for status, urls in statuses.items():
                summary_df_list.append({'period': period, 'agent': agent, 'status': status, 'count': len(urls)})
                if url_to_counts:
                    summary_df_list[-1].update({'tokens': sum([url_to_counts[url] for url in urls])})
    
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

def bucket_urls_by_size(url_sizes, bucket_boundaries):
    bucket_keys = [f"{bucket_boundaries[i]}-{bucket_boundaries[i+1]}" for i in range(len(bucket_boundaries) - 1)]

    # {"lower_bound - upper_bound" -> [list of URLs]}
    bucket_to_urls = defaultdict(list)
    for url, size in url_sizes.items():
        for i in range(len(bucket_boundaries) - 1):
            if bucket_boundaries[i] <= size < bucket_boundaries[i+1]:
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
            results[period] = {'unchanged': unchanged, 'changed': changed}
            continue

        # Compare with previous period
        for url, status in current_urls.items():
            if url in previous_urls:
                if previous_urls[url] == status:
                    unchanged += 1
                else:
                    changed[(previous_urls[url], status)] += 1

        results[period] = {'unchanged': unchanged, 'changed': changed}
        previous_urls = current_urls

    return results


def plot_size_against_restrictions(
    url_robots_summary,
    size_bucket_to_urls,
    agent_group,
    setting=None
):
    agent_names = get_bots(agent_group, setting=setting)
    # {URL --> Date --> Agent --> Status} --> {URL —> status}
    current_url_status = get_latest_url_robot_statuses(url_robots_summary, agent_names)
    print(len(current_url_status))

    set(current_url_status.keys())

    cat_keys = ["all", "some", "none"]
    data_groups = defaultdict(lambda: [0, 0, 0])
    url_to_bucket_key = {url: sz for sz, urls in size_bucket_to_urls.items() for url in urls}
    # data: "bucket range": [full restrictions, some restrictions, no restrictions]
    for url, status in current_url_status.items():
        if not status:
            print(url)
            print(status)
        data_groups[url_to_bucket_key[url]][cat_keys.index(status)] += 1
    
    print(data_groups)

    return analysis_util.plot_stackedbars(
        data_groups, 
        title=None, 
        category_names=['Full Restrictions', 'Some Restrictions', 'No Restrictions'],
        custom_colors=['#e04c71','#e0cd92','#82b5cf'],
        group_order=sorted(size_bucket_to_urls.keys(), key=lambda x: int(x.split('-')[0])), 
        total_dsets=len(url_to_bucket_key), 
        legend=True, 
        savepath=f"paper_figures/altair/robots_restrictions_vs_token_count_{agent_group}.json"
    )


def plot_robots_time_map(df, agent_type, val_key):

    # Filter the DataFrame for the relevant agent
    filtered_df = df[df['agent'] == agent_type]
    
    # Group by 'period' and 'status', and sum up the 'count'
    grouped_df = filtered_df.groupby(['period', 'status'])[val_key].sum().unstack(fill_value=0)
    
    # Optional: Reorder the columns as desired (replace 'status1', 'status2', etc., with your actual status names)
    ordered_statuses = ['N/A', 'none', 'some', 'all']  # Example: reorder as per your preference
    grouped_df = grouped_df[ordered_statuses]
    
    # Calculate the total counts for each period
    total_counts = grouped_df.sum(axis=1)
    
    # Calculate the percentage of each status per period
    percent_df = grouped_df.div(total_counts, axis=0) * 100
    
    # Specify colors for each stack (ensure this matches the order of statuses in 'ordered_statuses')
    colors = ['gray', 'blue', 'orange', 'red']  # Assign colors to each status
    
    # Optional: Rename columns for custom labels in the legend
    percent_df.columns = ['No Website/Robots', 'No Restrictions', 'Some Restrictions', 'Full Restrictions']  # Example labels
    # gray (n/a), blue (some), red (none), orange (all)
    # Plotting the stacked area chart
    # percent_df.plot(kind='area', stacked=True, figsize=(10, 6))#, color=colors)
    percent_df.plot(kind='area', stacked=True, figsize=(10, 6), color=colors)
    
    # plt.title(f"Restriction Status for {agent_type} over C4 Top 800")
    plt.title(f"Restriction Status for {agent_type} over 10k Random Sample")
    plt.xlabel('Period')
    plt.ylabel('Percentage')
    plt.legend(title='Status')
    plt.show()
    plt.clf()