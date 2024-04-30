import csv
import json
import logging
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import chardet
from bs4 import BeautifulSoup


def sanitize_url(url: str) -> str:
    """Sanitizes URL to be used as a folder name."""
    parsed_url = urllib.parse.urlparse(url)
    sanitized_netloc = parsed_url.netloc.replace(".", "_")
    sanitized_path = "_".join(
        filter(None, re.split(r"\/+", parsed_url.path.strip("/")))
    )
    sanitized_url = f"{sanitized_netloc}_{sanitized_path}"
    return sanitized_url.replace(".", "_")  # bug?


def extract_urls(csv_directory: Path) -> list[str]:
    """Extracts URLs from a directory of CSV files."""
    assert (
        csv_directory.is_dir()
    ), f"Invalid input path, must be directory: {csv_directory}"
    urls = set()
    for file_path in csv_directory.glob("*.csv"):
        with file_path.open(mode="r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                for i in range(1, 6):
                    column_name = f"Terms of Use Link {i}"
                    url = row.get(column_name)
                    if url:
                        urls.add(url)
    if not urls:
        logging.error(f"No URLs and/or CSV files found in: {csv_directory}")
    return list(urls)


def save_data_to_json(data: dict, output_file: str) -> None:
    """Saves dict to JSON file."""
    if not output_file.endswith(".json"):
        output_file = f"{output_file}.json"
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"Saved data to {output_file}")
    except IOError as e:
        logging.error(f"Failed to save data to {output_file}", exc_info=True)


def process_directory(
    directory: str, url: str, num_workers: int
) -> dict[str, dict[str, str]]:
    """Processes a single directory of HTML files and returns a dictionary of formatted text."""
    data = []

    def process_file(filename):
        if filename.endswith(".html"):
            file_path = os.path.join(directory, filename)
            formatted_text = extract_and_format_text(file_path)
            if formatted_text:
                date_string = filename[:14]  # extract the date from the filename
                date = datetime.strptime(date_string, "%Y%m%d%H%M%S")
                formatted_date = date.strftime("%m-%d-%Y")
                return (formatted_date, formatted_text)
        return None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for filename in os.listdir(directory):
            futures.append(executor.submit(process_file, filename))

        for future in as_completed(futures):
            result = future.result()
            if result:
                data.append(result)

    data.sort(key=lambda x: datetime.strptime(x[0], "%m-%d-%Y"))

    return {url: dict(data)}


def parse_html_directories(
    root_directory: str,
    csv_directory: str,
    num_workers: int,
) -> dict[str, dict[str, str]]:
    """Processes directories of HTML files and returns a dictionary of formatted text."""
    data = {}
    urls = extract_urls(csv_directory)

    for directory_name in os.listdir(root_directory):
        directory_path = os.path.join(root_directory, directory_name)
        if os.path.isdir(directory_path):
            sanitized_urls = [sanitize_url(url) for url in urls]
            url = next(
                (
                    url
                    for url, sanitized_url in zip(urls, sanitized_urls)
                    if sanitized_url == directory_name
                ),
                None,
            )
            if url:
                data.update(process_directory(directory_path, url, num_workers))

    return data


def extract_and_format_text(file_path: str) -> str:
    """Extracts text from an HTML file, enhancing formatting for readability and structure."""
    try:
        with open(file_path, "rb") as file:
            raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)["encoding"] or "utf-8"

        with open(file_path, "r", encoding=detected_encoding) as file:
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
