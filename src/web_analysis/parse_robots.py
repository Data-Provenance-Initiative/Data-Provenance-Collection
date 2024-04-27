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

def interpret_agent(rules):

    agent_disallow = [x for x in rules.get("Disallow", []) if "?" not in x]
    agent_allow = [x for x in rules.get("Allow", []) if "?" not in x]

    if len(agent_disallow) == 0 or agent_disallow == [""] or (agent_allow == agent_disallow):
        disallow_type = "none"
    elif any('/' == x.strip() for x in agent_disallow) and len(agent_allow) == 0:
        disallow_type = "all"
    else:
        disallow_type = "some"

    return disallow_type


def interpret_robots(agent_rules, all_agents):
    """Given the robots.txt agent rules, and a list of relevant agents 
    (a superset of the robots.txt), determine whether they are:

    (1) blocked from scraping all parts of the website
    (2) blocked from scraping part of the website
    (3) NOT blocked from scraping at all
    """
    # agent --> "all", "some", or "none" blocked.
    agent_to_judgement = {}

    star_judgement = interpret_agent(agent_rules.get("*", {}))
    agent_to_judgement["*"] = star_judgement

    for agent in all_agents:
        rule = agent_rules.get(agent)
        judgement = interpret_agent(rule) if rule is not None else agent_to_judgement["*"]
        agent_to_judgement[agent] = judgement

    return agent_to_judgement

        
def aggregate_robots(url_to_rules, all_agents):
    """Across all robots.txt, determine basic stats:
    (1) For each agent, how often it is explicitly mentioned
    (2) For each agent, how often it is blocked from scraping all parts of the website
    (3) For each agent, how often it is blocked from scraping part of the website
    (4) For each agent, how often it is NOT blocked from scraping at all
    """
    robots_stats = defaultdict(lambda: {'counter': 0, 'all': 0, 'some': 0, 'none': 0})
    url_decisions = {}

    for url, robots in url_to_rules.items():
        agent_to_judgement = interpret_robots(robots, all_agents)
        url_decisions[url] = agent_to_judgement
        # Trace individual agents and wildcard agent
        for agent in all_agents + ["*"]:
            judgement = agent_to_judgement[agent]
            robots_stats[agent][judgement] += 1
            if agent in robots: # Missing robots.txt means nothing blocked
                robots_stats[agent]["counter"] += 1

        # See if *All Agents* are blocked on all content, 
        # or at least some agents can access some or more content, or 
        # there are no blocks on any agents at all.
        if all(v == "all" for v in agent_to_judgement.values()):
            agg_judgement = "all"
        elif any(v in ["some", "all"] for v in agent_to_judgement.values()):
            agg_judgement = "some"
        else:
            agg_judgement = "none"
        robots_stats["*All Agents*"]["counter"] += 1
        robots_stats["*All Agents*"][agg_judgement] += 1
        url_decisions[url]["*All Agents*"] = agg_judgement
        
    return robots_stats, url_decisions


def read_robots_file(file_path):
    with gzip.open(file_path, 'rt') as file:
        return json.load(file)

def main(args):
    data = read_robots_file(args.file_path)
    print(f"Total URLs: {len(data)}")
    print(f"URLs with robots.txt: {sum(1 for robots_txt in data.values() if robots_txt)}")

    # populate the empty rules (missing robots.txt)
    url_to_rules = {url: {} for url, txt in data.items() if not txt}
    # interpret the robots.txt
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(parse_robots_txt, txt): url for url, txt in data.items() if txt}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                url_to_rules[url] = future.result()
            except Exception as e:
                print(f"Error processing {url}: {e}")

    # Collect all agents
    all_agents = list(set([agent for url, rules in url_to_rules.items() 
                      for agent, _ in rules.items() if agent not in ["ERRORS", "Sitemaps", "*"]]))

    robots_stats, url_decisions = aggregate_robots(url_to_rules, all_agents)

    # URL --> agents in its robots.txt and their decisions (all/some/none).
    url_interpretations = {k: {agent: v for agent, v in vs.items() if agent not in ["ERRORS", "Sitemaps"] and agent in list(url_to_rules[k]) + ["*All Agents*"]} for k, vs in url_decisions.items()}

    # PRINT OUT INFO ON INDIVIDUAL URLS:
    # for url, interp in url_interpretations.items():
    #     if interp.get("*") == "all":
    #         print(url)
    # import pdb; pdb.set_trace()
    # print(url_interpretations["http://www.machinenoveltranslation.com/robots.txt"])

    sorted_robots_stats = sorted(robots_stats.items(), key=lambda x: x[1]['counter'] / len(data) if len(data) > 0 else 0, reverse=True)

    print(f"{'Agent':<30} {'Observed': <10} {'Blocked All': >15} {'Blocked Some': >15} {'Blocked None': >15}")
    for i, (agent, stats) in enumerate(sorted_robots_stats):
        counter_pct = stats['counter'] / len(data) * 100 if len(data) > 0 else 0
        all_pct = stats['all'] / len(data) * 100 if len(data) > 0 else 0
        some_pct = stats['some'] / len(data) * 100 if len(data) > 0 else 0
        none_pct = stats['none'] / len(data) * 100 if len(data) > 0 else 0
        print_outs = [
            f"{agent:<20}",
            f"{stats['counter']:>5} ({counter_pct:5.1f}%) {'':>5} ",
            f"{stats['all']:>5} ({all_pct:5.1f}%) {'':>5} ",
            f"{stats['some']:>5} ({some_pct:5.1f}%) {'':>5} ",
            f"{stats['none']:>5} ({none_pct:5.1f}%)",
        ]
        print(" ".join(print_outs))
        if i > 15:
            print(f"........")
            break


if __name__ == "__main__":
    """
    Example commands:

    python src/web_analysis/parse_robots.py <in-path> <out-path>
    
    python src/web_analysis/parse_robots.py data/robots-test.json.gz data/robots-analysis.csv 

    Process:
        1. Reads the json.gz mapping base-url to robots.txt
        2. Parse all the robots.txt so they can be analyzed on aggregate
    """
    parser = argparse.ArgumentParser(description="Parse and analyze robots.txt.")
    parser.add_argument("file_path", type=str, help="Path to the JSON.GZ file mapping urls to robots.txt text.")
    parser.add_argument("output_path", type=str, help="Path where analysis will be saved.")
    args = parser.parse_args()
    main(args)
