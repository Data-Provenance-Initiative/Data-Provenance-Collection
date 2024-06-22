import csv
import os
import json
import logging
import multiprocessing
import re
import sys
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm
from urllib.parse import urlparse
from pympler import asizeof

import chardet
from bs4 import BeautifulSoup

from ..extract_robots import normalize_url


def extract_urls(csv_file_path: Path, site_type: str) -> list[str]:
    """
    Extracts URLs from a single CSV file.
    """
    assert (
        csv_file_path.suffix == ".csv"
    ), f"Invalid input path, must be a csv file: {csv_file_path}"
    urls = set()
    with csv_file_path.open(mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if site_type == "tos":
                for i in range(1, 6):
                    column_name = f"Terms of Use Link {i}"
                    url = row.get(column_name)
                    if url:
                        if not url.startswith("https://") and not url.startswith(
                            "http://"
                        ):
                            url = "https://" + url
                        urls.add(url)
            else:
                url = row.get("URL")
                if url:
                    if site_type == "robots":
                        url = normalize_url(url)
                    elif site_type == "main":
                        if not url.startswith("https://") and not url.startswith(
                            "http://"
                        ):
                            url = "https://" + url
                    urls.add(url)
    if not urls:
        logging.error(f"No URLs found in the CSV file: {csv_file_path}")
    return list(urls)


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


def process_directory(
    directory: Path, url: str, site_type: str, num_workers: int
) -> dict[str, dict[str, str]]:
    """
    Processes a single directory of HTML files and returns a dictionary of formatted text.
    """
    data = {}

    def process_file(file_path: Path) -> Optional[dict[str, str]]:
        if file_path.suffix == ".html":
            formatted_text = extract_and_format_text(file_path)
            if formatted_text:
                date_string = file_path.stem[:14]  # extract the date from the filename
                date = datetime.strptime(date_string, "%Y%m%d%H%M%S")
                formatted_date = date.strftime("%Y-%m-%d")
                return {formatted_date: formatted_text}
        return None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for file_path in directory.glob("*.html"):
            futures.append(executor.submit(process_file, file_path))

        for future in futures:
            result = future.result()
            if result:
                data.update(result)

    data = dict(sorted(data.items()))

    if site_type == "tos":
        return {url: dict(data)} if data else {}
    else:
        return data if data else {}


def parse_html_directories(
    root_directory: Union[str, Path],
    csv_file_path: Union[str, Path],
    site_type: str,
    num_workers: int,
    num_processes: int = None,
    max_chunk_size: Optional[int] = 5000,
    output_json_path: Union[str, Path] = "temporal_data",
) -> list[Path]:
    """
    Processes directories of HTML files and writes the formatted text to JSON files.
    Returns a list of output file paths.
    """
    root_directory = Path(root_directory)
    csv_file_path = Path(csv_file_path)
    # either output_json_path or output_json_path.json if given name does not end with json
    output_json_path = Path(output_json_path.with_suffix(".json"))

    if not num_processes:
        num_processes = multiprocessing.cpu_count()

    current_chunk = {}
    current_chunk_size = 0
    chunk_number = 1
    output_files = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        worker_func = partial(
            process_row,
            root_directory=root_directory,
            site_type=site_type,
            num_workers=num_workers,
        )
        with csv_file_path.open("r") as file:
            csv_reader = csv.DictReader(file)
            for result in tqdm(
                pool.imap(worker_func, csv_reader),
                desc="Process HTML to JSON",
                unit="row",
                total=sum(1 for _ in csv_file_path.open("r")),
            ):
                for key, value in result.items():
                    item_size = asizeof.asizeof({key: value})
                    if (
                        max_chunk_size
                        and current_chunk_size + item_size > max_chunk_size
                    ):
                        chunk_file_path = (
                            output_json_path.parent
                            / f"{output_json_path.stem}_chunk_{chunk_number}.json"
                        )
                        with chunk_file_path.open(
                            "w", encoding="utf-8", errors="surrogateescape"
                        ) as chunk_file:
                            json.dump(
                                current_chunk, chunk_file, ensure_ascii=True, indent=4
                            )
                        output_files.append(chunk_file_path)
                        current_chunk = {}
                        current_chunk_size = 0
                        chunk_number += 1

                    current_chunk.update(result)
                    current_chunk_size += item_size

    if current_chunk:
        if len(output_files) == 0:
            with output_json_path.open(
                "w", encoding="utf-8", errors="surrogateescape"
            ) as file:
                json.dump(current_chunk, file, ensure_ascii=True, indent=4)
            output_files.append(output_json_path)
        else:
            chunk_file_path = (
                output_json_path.parent
                / f"{output_json_path.stem}_chunk_{chunk_number}.json"
            )
            with chunk_file_path.open(
                "w", encoding="utf-8", errors="surrogateescape"
            ) as chunk_file:
                json.dump(current_chunk, chunk_file, ensure_ascii=True, indent=4)
            output_files.append(chunk_file_path)

    return output_files


def process_row(
    row: dict, root_directory: Path, site_type: str, num_workers: int
) -> dict[str, dict[str, dict[str, str]]]:
    data = {}
    if site_type == "robots" or site_type == "main":
        url = row["URL"]

        if site_type == "robots":
            normalized_url = normalize_url(url)
            sanitized_url = sanitize_url(normalized_url)
        else:
            sanitized_url = sanitize_url(url)

        # Check if the sanitized URL exists as a directory
        if sanitized_url in os.listdir(root_directory):
            directory_path = root_directory / sanitized_url
            processed_data = process_directory(
                directory_path, url, site_type, num_workers
            )
            if processed_data:
                data[url] = processed_data

        # Check for directories with leading underscores
        elif "_" + sanitized_url in os.listdir(root_directory):
            directory_path = root_directory / ("_" + sanitized_url)
            processed_data = process_directory(
                directory_path, url, site_type, num_workers
            )
            if processed_data:
                data[url] = processed_data

        # Check for URLs starting with "www."
        elif url.startswith("www."):
            url_copy = url.replace("www.", "")
            sanitized_url_copy = sanitize_url(url_copy)

            if sanitized_url_copy in os.listdir(root_directory):
                directory_path = root_directory / sanitized_url_copy
                processed_data = process_directory(
                    directory_path, url, site_type, num_workers
                )
                if processed_data:
                    data[url] = processed_data

            elif "_" + sanitized_url_copy in os.listdir(root_directory):
                directory_path = root_directory / ("_" + sanitized_url_copy)
                processed_data = process_directory(
                    directory_path, url, site_type, num_workers
                )
                if processed_data:
                    data[url] = processed_data

            elif "_www_" + sanitized_url in os.listdir(root_directory):
                directory_path = root_directory / ("_www_" + sanitized_url)
                processed_data = process_directory(
                    directory_path, url, site_type, num_workers
                )
                if processed_data:
                    data[url] = processed_data

        # Handle leading/trailing underscores for "main" site_type
        if site_type == "main":
            if sanitized_url.startswith("_"):
                sanitized_url = sanitized_url[1:]
            if not sanitized_url.endswith("_"):
                sanitized_url += "_"
            directory_path = root_directory / sanitized_url
            if directory_path.is_dir():
                processed_data = process_directory(
                    directory_path, url, site_type, num_workers
                )
                if processed_data:
                    data[url] = processed_data

    elif site_type == "tos":
        url_domain = row["Domain"]
        count = 0
        for i in range(1, 6):
            column_name = f"Terms of Use Link {i}"
            url = row.get(column_name)
            if url:
                sanitized_url = sanitize_url(url)
                directory_path = root_directory / sanitized_url
                if directory_path.is_dir():
                    processed_data = process_directory(
                        directory_path, url, site_type, num_workers
                    )
                    if count == 0:
                        data[url_domain] = {}
                    count += 1
                    if processed_data:
                        data[url_domain].update(processed_data)

    return data


def extract_and_format_text(file_path: Path) -> str:
    """
    Extracts text from an HTML file, enhancing formatting for readability and structure.
    """
    try:
        with file_path.open("rb") as file:
            raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)["encoding"] or "utf-8"

        with file_path.open("r", encoding=detected_encoding, errors="replace") as file:
            soup = BeautifulSoup(file, "html.parser")

            for element in soup(["script", "style"]):
                element.decompose()

            replacements = {
                "h1": "\n\n# {content} #\n\n",
                "h2": "\n\n## {content} ##\n\n",
                "h3": "\n\n### {content} ###\n\n",
                "p": "\n\n{content}\n\n",
                "br": "\n",
            }
            for tag, template in replacements.items():
                for elem in soup.find_all(tag):
                    elem.replace_with(template.format(content=elem.get_text().strip()))

            text = "\n".join(
                [line.strip() for line in soup.get_text().splitlines() if line.strip()]
            )
            return text
    except Exception as e:
        logging.error(
            f"Error occurred while processing file: {file_path}", exc_info=True
        )
        return ""


def get_website_start_dates(snapshot_dir: str) -> dict[str, pd.Timestamp]:
    start_dates = {}
    for sanitized_url in os.listdir(snapshot_dir):
        url_dir = os.path.join(snapshot_dir, sanitized_url)
        if os.path.isdir(url_dir):
            snapshots = [f for f in os.listdir(url_dir) if f.endswith(".html")]
            if snapshots:
                earliest_snapshot = min(snapshots)
                start_date_str = earliest_snapshot.split(".")[0]
                start_date = pd.to_datetime(start_date_str, format="%Y%m%d%H%M%S")
                start_dates[sanitized_url] = start_date
    return start_dates


def find_farthest_dates(dates: list[str]) -> list[str]:
    """
    Find the two dates that are as far apart as possible.
    """
    if len(dates) <= 2:
        return dates
    date_objs = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    max_diff = -1
    farthest_pair = (date_objs[0], date_objs[-1])
    for date1, date2 in combinations(date_objs, 2):
        diff = abs((date2 - date1).days)
        if diff > max_diff:
            max_diff = diff
            farthest_pair = (date1, date2)
    return [date.strftime("%Y-%m-%d") for date in farthest_pair]


def create_json(json_directory: str, output_file: str) -> None:
    """
    Create a JSON file with biannual snapshots for each year.
    """
    sampled_data = {}
    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(json_directory, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            for domain, tos_links in data.items():
                sampled_domain_data = {}

                for tos_link, snapshots in tos_links.items():
                    date_snapshot_pairs = [
                        (date, snapshots[date]) for date in snapshots.keys()
                    ]

                    date_snapshot_pairs.sort()

                    sampled_snapshots = {}

                    for date, snapshot in date_snapshot_pairs:
                        year = date[:4]

                        if year not in sampled_snapshots:
                            sampled_snapshots[year] = []
                        sampled_snapshots[year].append((date, snapshot))

                    # two dates that are as far apart as possible for each year
                    final_snapshots = {}
                    for year, year_snapshots in sampled_snapshots.items():
                        if len(year_snapshots) > 1:
                            dates = [pair[0] for pair in year_snapshots]
                            farthest_dates = find_farthest_dates(dates)
                            selected_snapshots = [
                                pair
                                for pair in year_snapshots
                                if pair[0] in farthest_dates
                            ]
                        else:
                            selected_snapshots = year_snapshots
                        final_snapshots.update(
                            {date: snapshot for date, snapshot in selected_snapshots}
                        )

                    sampled_domain_data[tos_link] = final_snapshots

                sampled_data[domain] = sampled_domain_data

        with open(output_file, "w") as file:
            json.dump(sampled_data, file, indent=4)

        print("Sampling completed. Sampled data saved to", output_file)


def count_and_delete_zero_count_jsons(directory: str, delete=False) -> int:
    zero_count_jsons = 0

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                if all(value == 0 for value in data["change_counts"].values()):
                    zero_count_jsons += 1
                    if delete:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")

    return zero_count_jsons


def consolidate_tos_links(
    directory: str, output_file: str = "consolidated_terms_of_use.csv"
) -> None:
    consolidated_data = defaultdict(set)

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                domain = row["Domain"]
                tos_links = [
                    row[f"Terms of Use Link {i}"]
                    for i in range(1, 6)
                    if pd.notnull(row[f"Terms of Use Link {i}"])
                ]
                consolidated_data[domain].update(tos_links)

    data = []
    for domain, tos_links in consolidated_data.items():
        if tos_links:  # Only include domains with at least one TOS link
            tos_links_list = list(tos_links) + [None] * (
                5 - len(tos_links)
            )  # Pad the list to ensure 5 columns
            data.append([domain] + tos_links_list[:5])

    consolidated_df = pd.DataFrame(
        data,
        columns=[
            "Domain",
            "Terms of Use Link 1",
            "Terms of Use Link 2",
            "Terms of Use Link 3",
            "Terms of Use Link 4",
            "Terms of Use Link 5",
        ],
    )

    consolidated_df.to_csv(output_file, index=False)
    print(f"Consolidation complete. The consolidated file is saved as '{output_file}'.")
