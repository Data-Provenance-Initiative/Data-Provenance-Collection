import argparse
import io
import gzip
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_robots_txt(robots_txt):
    """Parses the robots.txt to create a map of agents, sitemaps, and parsing oddities."""
    rules = {"ERRORS": []}
    current_agents = []
    
    for line in robots_txt.splitlines():
        line = line.strip()
        if line.startswith('User-agent:'):
            agent = line.split(':', 1)[1].strip()
            if agent not in rules:
                rules[agent] = defaultdict(list)
            current_agents = [agent]
        elif line.startswith(('Allow:', 'Disallow:', 'Crawl-delay:')) and current_agents:
            for agent in current_agents:
                rules[agent][line.split(":")[0]].append(":".join(line.split(":")[1:]).strip())
        elif line.lower().startswith('sitemap:'):
            rules.setdefault('Sitemaps', []).append(line.split(':', 1)[1].strip())
        elif line == "" or line.startswith('#'):
            current_agents = []
        else:
            rules["ERRORS"].append(f"Unmatched line: {line}")
    
    return rules

def summarize_rules(rules):
    summary = defaultdict(lambda: {'total': 0, 'disallow_all': 0})
    for agent, rule in rules.items():
        if agent not in ["ERRORS", "Sitemaps"]:
            summary[agent]['total'] += 1
            if any('/' == x.strip() for x in rule['Disallow:']):
                summary[agent]['disallow_all'] += 1
    return summary

def read_robots_file(file_path):
    with gzip.open(file_path, 'rt') as file:
        return json.load(file)

def main(args):
    data = read_robots_file(args.file_path)
    print(f"Total URLs: {len(data)}")
    print(f"URLs with robots.txt: {sum(1 for robots_txt in data.values() if robots_txt)}")

    url_to_rules = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(parse_robots_txt, txt): url for url, txt in data.items() if txt}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                url_to_rules[url] = future.result()
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    summary = defaultdict(lambda: {'total': 0, 'disallow_all': 0})
    for result in url_to_rules.values():
        for agent, details in summarize_rules(result).items():
            summary[agent]['total'] += details['total']
            summary[agent]['disallow_all'] += details['disallow_all']

    # Create a sorted list of agents based on the percentage of total URLs
    sorted_summary = sorted(summary.items(), key=lambda x: x[1]['total'] / len(data) if len(data) > 0 else 0, reverse=True)

    # Print the results in a tabular format
    print(f"{'Agent':<30} {'Total (%)':>15} {'Disallow All (%)':>20}")
    for i, (agent, counts) in enumerate(sorted_summary):
        total_percent = counts['total'] / len(data) * 100 if len(data) > 0 else 0
        disallow_all_percent = counts['disallow_all'] / len(data) * 100 if len(data) > 0 else 0
        print(f"{agent:<20} {counts['total']:>5} ({total_percent:6.2f}%) {'':>5} {counts['disallow_all']:>5} ({disallow_all_percent:6.2f}%)")
        if i > 15:
            print(f"........")
            break



if __name__ == "__main__":
    """
    Example commands:

    python src/web_analysis/parse_robots.py <in-path> <out-path>
    
    python src/web_analysis/parse_robots.py data/robots-test.json.gz data/robots-analysis.csv 

    Process:
        1. Reads the txt/csv of URLs from your input path
        2. Pulls the robots.txt for any URLs that are not already in the <output-path> if it exists
        3. Saves a new mapping from base-url to robots.txt text at the <output-path>
    """
    parser = argparse.ArgumentParser(description="Parse and analyze robots.txt.")
    parser.add_argument("file_path", type=str, help="Path to the JSON.GZ file mapping urls to robots.txt text.")
    parser.add_argument("output_path", type=str, help="Path where analysis will be saved.")
    args = parser.parse_args()
    main(args)
