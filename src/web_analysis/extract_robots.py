import argparse
import os
import json
import gzip
import csv
import requests
import concurrent
import logging

from urllib.parse import urlparse, urlunparse
from tenacity import retry, stop_after_attempt, wait_fixed
from concurrent.futures import ThreadPoolExecutor


# Setting up logging to output errors to the console
logging.basicConfig(level=logging.ERROR)


def read_urls_from_file(file_path):
    """Reads the URLs from a txt file or csv file."""
    urls = []
    try:
        with open(file_path, 'r') as file:
            if file_path.endswith('.csv'):
                reader = csv.reader(file)
                urls = [row[0] for row in reader]
            else:
                urls = file.read().splitlines()
    except Exception as e:
        print(f"Error reading URLs from file: {e}")
    return urls


def normalize_url(url):
    """Normalizes URL by ensuring it has 'https://' and ends with '/robots.txt'."""
    # Parse the URL to components
    if not url.startswith("http://"):
        url = "http://" + url

    if not url.endswith("/robots.txt"):
        url = url.rstrip('/') + '/robots.txt'

    return url


# This function includes retry logic using tenacity
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_robots_txt(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad HTTP status codes
        if response.status_code == 200:
            return response.text
        else:
            logging.error(f"Failed to fetch {url}: Status code {response.status_code}")
            return None
    except requests.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        return None

# Function to process multiple URLs in parallel
def parse_robots_txt(urls, max_workers=10):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(get_robots_txt, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results[url] = result
                # if result is None:
                    # logging.error(f"No data returned for {url}")
            except Exception as e:
                logging.error(f"Exception for {url}: {e}")
                results[url] = None

    print("finished")
    return results

def read_existing_results(file_path):
    """Reads existing gzipped JSON file and returns the results as a dictionary."""
    try:
        with gzip.open(file_path, 'rt', encoding='UTF-8') as zipfile:
            return json.load(zipfile)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error reading existing file: {e}")
        return {}

def save_results(results, file_path):
    """Saves results in a gzipped JSON file."""
    try:
        with gzip.open(file_path, 'wt', encoding='UTF-8') as zipfile:
            json.dump(results, zipfile)
    except Exception as e:
        print(f"Error writing results to file: {e}")

def main(args):
    # Read URLs and prepare them
    base_urls = read_urls_from_file(args.file_path)
    print(f"Total URLs passed from input file: {len(base_urls)}")

    # Check if output file exists and read existing results
    existing_results = {}
    if os.path.exists(args.output_path):
        existing_results = read_existing_results(args.output_path)
        print(f"Existing URLs in output file: {len(existing_results)}")

    # Normalize and filter out URLs that have already been fetched
    normalized_urls = [normalize_url(url) for url in base_urls]
    urls_to_fetch = [url for url in normalized_urls if url not in existing_results]

    # Fetch robots.txt content
    new_results = parse_robots_txt(urls_to_fetch)
    successful_fetches = sum(1 for result in new_results.values() if result is not None)
    failed_fetches = len(new_results) - successful_fetches
    print(f"Successfully fetched: {successful_fetches}")
    print(f"Failed to fetch: {failed_fetches}")

    # Update existing results with new fetches
    existing_results.update(new_results)

    # Save updated results
    save_results(existing_results, args.output_path)
    print(f"Total URLs saved in output file: {len(existing_results)}")

if __name__ == "__main__":
    """
    Example commands:

    python src/web_analysis/extract_robots.py <in-path> <out-path>

    python src/web_analysis/extract_robots.py src/web_analysis/urls-test.txt data/robots-test.json.gz

    Process:
        1. Reads the txt/csv of URLs from your input path
        2. Pulls the robots.txt for any URLs that are not already in the <output-path> if it exists
        3. Saves a new mapping from base-url to robots.txt text at the <output-path>
    """
    parser = argparse.ArgumentParser(description="Fetch and update robots.txt files for a list of URLs.")
    parser.add_argument("file_path", type=str, help="Path to the file containing URLs.")
    parser.add_argument("output_path", type=str, help="Path where the gzipped JSON file will be saved.")
    args = parser.parse_args()
    main(args)

