import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from web_analysis import robots_util


def load_json_file(filepath: str) -> Optional[List[Dict]]:
    """Load a JSON file if it exists, otherwise return None."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load {filepath}")
        return None


def get_domain_purposes(url: str, url_to_domain_df: pd.DataFrame) -> List[str]:
    """Get list of purposes for a given URL."""
    try:
        return url_to_domain_df[url_to_domain_df["URL"] == url]["Domains"].tolist()
    except:
        return []


def get_modalities(url: str, pretraining_df: pd.DataFrame) -> Dict[str, Any]:
    """Get modalities and access info for a URL from pretraining annotations.

    Returns dict with:
    - modalities: list of content types present (text, images, video, audio)
    - paywall: No/Some/All/None if unknown
    - advertisements: True/False/None if unknown
    """
    try:
        # Look for Domain column
        if "Domain" not in pretraining_df.columns:
            print(
                f"Warning: No Domain column found in pretraining annotations for {url}"
            )
            return {"modalities": [], "paywall": None, "advertisements": None}

        row = pretraining_df[pretraining_df["Domain"] == url]
        if row.empty:
            return {"modalities": [], "paywall": None, "advertisements": None}

        # Get modalities from "Content Modalities: X" columns
        modalities = []
        content_types = {
            "Content Modalities: Text": "text",
            "Content Modalities: Images": "images",
            "Content Modalities: Video": "video",
            "Content Modalities: Audio": "audio",
        }
        for col, modality in content_types.items():
            if col in row.columns and pd.notna(row[col].iloc[0]):
                modalities.append(modality)

        # Get paywall info
        paywall = None
        if "Paywall" in row.columns:
            val = row["Paywall"].iloc[0]
            if pd.notna(val):
                paywall = val

        # Get advertisements info
        ads = None
        if "Advertisements" in row.columns:
            val = row["Advertisements"].iloc[0]
            if pd.notna(val):
                # Handle both string and boolean types
                if isinstance(val, str):
                    ads = val.upper() == "TRUE"
                else:
                    ads = bool(val)

        return {"modalities": modalities, "paywall": paywall, "advertisements": ads}
    except Exception as e:
        print(f"Error processing modalities for {url}: {str(e)}")
        return {"modalities": [], "paywall": None, "advertisements": None}


def get_corpus_ranks(
    url: str, url_token_lookup: robots_util.URLTokenLookup
) -> List[Dict[str, Any]]:
    """Get corpus ranks for a URL using top_k_urls method."""
    corpora = []
    for corpus in ["c4", "rf", "dolma"]:
        try:
            # Get all URLs sorted by token count by setting k to a large number
            # verbose=False to avoid printing token counts
            all_ranked_urls = url_token_lookup.top_k_urls(
                corpus, k=100000, verbose=False
            )
            # Find rank of our URL (1-based indexing)
            for rank, ranked_url in enumerate(all_ranked_urls, 1):
                if ranked_url == url:
                    corpora.append({"corpus": corpus.upper(), "rank": rank})
                    break
        except:
            continue
    return corpora


def aggregate_all_data() -> None:
    """
    Collect all data and save to JSON in the following format:
    {
        "<website_url>": {
            "corpora": [
                {
                    "corpus": "Dolma",
                    "rank": 111
                },
                {
                    "corpus": "c4",
                    "rank": 92
                }
            ],
            "purpose": [
                "news",
                "..."
            ],
            "modalities": [
                "text",
                "images",
                "..."
            ],
            "paywall": "No",
            "advertisements": True,
            "robots.txt restrictions": {
                "GPTBot": {
                    "2016": "No robots",
                    "2017": "Partial restricted",
                    "...": "<status>"
                },
                "CCBot": {
                    "...": "<status>"
                },
                // ... other agents
            },
            "Terms of Service": {
                "2016": "No AI",
                "...": "<restrictions>"
            }
        }
    }
    """
    # Construct absolute paths using current_dir
    data_dir = os.path.join(current_dir, "data")

    # Load input files
    url_to_domain = pd.read_csv(os.path.join(data_dir, "url_domain_mappings.csv"))
    pretraining_annotations = pd.read_excel(
        os.path.join(data_dir, "all_pretraining_annos.xlsx")
    )
    robots_data = load_json_file(os.path.join(data_dir, "agent_robots_statistics.json"))
    tos_data = load_json_file(os.path.join(data_dir, "agent_tos_statistics.json"))
    # Initialize URL token lookup
    url_token_lookup = robots_util.URLTokenLookup(
        os.path.join(data_dir, "pretrain_data/relevant_url_token_counts.csv")
    )
    # Create lookup dictionaries for robots and ToS data
    robots_lookup = {
        record["domain"]: {
            date: {agent: status for agent, status in agent_statuses.items()}
            for date, agent_statuses in record["timestamps"].items()
        }
        for record in (robots_data or [])
    }
    tos_lookup = {record["domain"]: record["timestamps"] for record in (tos_data or [])}

    # Get all unique URLs
    all_urls = set(url_to_domain["URL"].unique())
    all_urls.update(url for record in (robots_data or []) for url in [record["domain"]])
    all_urls.update(url for record in (tos_data or []) for url in [record["domain"]])

    # Build final aggregated data
    aggregated_data = {}
    for url in all_urls:
        modality_info = get_modalities(url, pretraining_annotations)
        aggregated_data[url] = {
            "corpora": get_corpus_ranks(url, url_token_lookup),
            "purpose": get_domain_purposes(url, url_to_domain),
            "modalities": modality_info["modalities"],
            "paywall": modality_info["paywall"],
            "advertisements": modality_info["advertisements"],
            "robots.txt restrictions": robots_lookup.get(url, {}),
            "Terms of Service": tos_lookup.get(url, {}),
        }
    output_path = os.path.join(data_dir, "aggregated_data.json")
    with open(output_path, "w") as f:
        json.dump(aggregated_data, f, indent=2)


if __name__ == "__main__":
    aggregate_all_data()
