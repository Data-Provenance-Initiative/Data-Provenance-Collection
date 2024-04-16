import argparse
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
import urllib.parse

import requests
from dateutil.relativedelta import relativedelta
from ratelimit import limits, sleep_and_retry

BASE_URL = "https://web.archive.org"

FREQUENCY_MAP = {
    "daily": ("timestamp:8", "%Y-%m-%d", relativedelta(days=1)),
    "weekly": ("timestamp:6", "%Y-%W", relativedelta(weeks=1)),
    "monthly": ("timestamp:6", "%Y-%m", relativedelta(months=1)),
    "annual": ("timestamp:4", "%Y", relativedelta(years=1)),
}

DEFAULT_FREQUENCY = "monthly"


class WaybackMachineClient:
    def __init__(self, num_workers, snapshots_folder, stats_folder):
        self.num_workers = num_workers
        self.snapshots_folder = snapshots_folder
        self.stats_folder = stats_folder
        self.session = requests.Session()

    @sleep_and_retry
    @limits(calls=3, period=1) # 3 calls per second
    def get_pages(
        self,
        url: str,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> tuple[list[tuple[datetime, str, str]], dict]:
        """
        Retrieves snapshots (web pages) for a URL by using the Wayback Machine CDX API.
        CDX Documentation: https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server

        Filters
        -------
        - '&from={start_date}&to={end_date}': Specific date range.
        - '&filter=!statuscode:404': Exclude snapshots with status code 404.
        - '&filter=!mimetype:warc/revisit': Exclude snapshots with MIME type "warc/revisit" (revisit record without content).
        - '&filter=mimetype:text/html': Only include snapshots with a MIME type of "text/html".
        - '&collapse={collapse_filter}': Collapse feature to group snapshots based on desired frequency.
        - '&fl=timestamp,original,mimetype,statuscode,digest': Field list to include in the API response.

        Returns
        -------
            tuple[list[tuple[datetime, str, str]], dict]: A tuple containing two elements:
                - A list of tuples, where each tuple represents a snapshot that contains:
                    - Snapshot datetime object.
                    - Snapshot URL.
                    - Snapshot content (HTML).
        """
        results = []
        unique_digests = defaultdict(set)  # if no change in digest, we do not save the snapshot
        stats = {"url": url}

        # use the CDX API collapse feature to get unique snapshots based on frequency
        collapse_filter, date_format, delta = FREQUENCY_MAP[frequency]

        api_url = f"{BASE_URL}/cdx/search/cdx?url={url}&output=json&from={start_date}&to={end_date}&filter=!statuscode:404&filter=!mimetype:warc/revisit&filter=mimetype:text/html&collapse={collapse_filter}&fl=timestamp,original,mimetype,statuscode,digest"

        while True:
            response = self.session.get(api_url)
            if response.status_code != 200:
                break

            data = response.json()
            header = data[0]  # get header for dictionary mapping so we don't have to hardcode indices
            field_indices = {field: index for index, field in enumerate(header)}

            for snapshot in data[1:]:
                snapshot_date = datetime.strptime(
                    snapshot[field_indices["timestamp"]], "%Y%m%d%H%M%S"
                )
                snapshot_url = f"{BASE_URL}/web/{snapshot[field_indices['timestamp']]}/{snapshot[field_indices['original']]}"
                snapshot_digest = snapshot[field_indices["digest"]]

                interval_key = snapshot_date.strftime(date_format)

                # process unique digests per frequency interval
                if snapshot_digest not in unique_digests[interval_key]:
                    unique_digests[interval_key].add(snapshot_digest)
                    snapshot_content = self.get_snapshot_content(snapshot_url)
                    if snapshot_content:
                        results.append((snapshot_date, snapshot_url, snapshot_content))

            # use built-in pagination feature of CDX API
            if "next_page_url" not in data:
                break
            api_url = data["next_page_url"]

        return results, stats

    @sleep_and_retry
    @limits(calls=3, period=1)
    def count_site_changes(self, url: str, start_date: str, end_date: str) -> int:
        """
        Counts the number of unique changes of a site within a given date range.
        To do this we use the collapse=digest feature to count unique snapshots only.
        ! NOTE: For the CDX API, only adjacent digest are collapsed, duplicates elsewhere in the cdx set are not affected.
        """
        api_url = f"{BASE_URL}/cdx/search/cdx?url={url}&from={start_date}&to={end_date}&output=json&filter=mimetype:text/html&collapse=digest"

        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            data = response.json()
            return len(data) - 1 if data else 0  # don't include the header
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to retrieve changes for {url}. Error: {str(e)}")
            return 0

    @sleep_and_retry
    @limits(calls=3, period=1)
    def get_snapshot_content(self, snapshot_url: str) -> Optional[str]:
        try:
            response = self.session.get(snapshot_url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(
                f"Failed to retrieve content for {snapshot_url}. Error: {str(e)}"
            )
            return None

    def process_url(
        self, url: str, start_date: str, end_date: str, frequency: str
    ) -> None:
        results, stats = self.get_pages(url, start_date, end_date, frequency)
        if results:
            for snapshot_date, snapshot_url, snapshot_content in results:
                logging.debug(f"Snapshot Date: {snapshot_date}")
                logging.debug(f"Snapshot URL: {snapshot_url}")
                self.save_snapshot(url, snapshot_date, snapshot_content)

        start_datetime = datetime.strptime(start_date, "%Y%m%d")
        end_datetime = datetime.strptime(end_date, "%Y%m%d")
        current_date = start_datetime

        _, date_format, delta = FREQUENCY_MAP[frequency]

        stats["change_counts"] = {}

        while current_date <= end_datetime:
            frequency_start = current_date.strftime("%Y%m%d")
            frequency_end = (current_date + delta - relativedelta(days=1)).strftime("%Y%m%d")

            frequency_change_count = self.count_site_changes(
                url, frequency_start, frequency_end
            )
            stats["change_counts"][current_date.strftime(date_format)] = frequency_change_count

            current_date = current_date + delta

        self.save_stats(url, stats)
        start_date_formatted = start_datetime.strftime("%m-%d-%Y")
        end_date_formatted = end_datetime.strftime("%m-%d-%Y")

        if not results:
            logging.info(
                f"No snapshots available for {url} between {start_date_formatted} and {end_date_formatted}"
            )
        else:
            logging.info(
                f"Processed snapshots for {url} between {start_date_formatted} and {end_date_formatted}"
            )

    def process_urls(
        self, urls: list[str], start_date: str, end_date: str, frequency: str
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.process_url, url, start_date, end_date, frequency)
                for url in urls
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing URL: {e}")

    def save_snapshot(
        self, url: str, snapshot_date: datetime, snapshot_content: str
    ) -> None:
        sanitized_url = self.sanitize_url(url)
        url_folder = os.path.join(self.snapshots_folder, sanitized_url)
        os.makedirs(url_folder, exist_ok=True)
        snapshot_filename = f"{snapshot_date.strftime('%Y%m%d%H%M%S')}.html"
        snapshot_path = os.path.join(url_folder, snapshot_filename)
        with open(snapshot_path, "w", encoding="utf-8") as file:
            file.write(snapshot_content)
        logging.debug(f"Snapshot saved as {snapshot_path}")

    def save_stats(self, url: str, stats: dict) -> None:
        os.makedirs(self.stats_folder, exist_ok=True)
        sanitized_url = self.sanitize_url(url)
        stats_filename = f"{sanitized_url}.json"
        stats_path = os.path.join(self.stats_folder, stats_filename)
        with open(stats_path, "w") as file:
            json.dump(stats, file, indent=4)
        logging.debug(f"Stats saved as {stats_path}")

    @staticmethod
    def sanitize_url(url: str) -> str:
        parsed_url = urllib.parse.urlparse(url)
        sanitized_netloc = parsed_url.netloc.replace(".", "_")
        sanitized_path = "_".join(
            filter(None, re.split(r"\/+", parsed_url.path.strip("/")))
        )
        sanitized_url = f"{sanitized_netloc}_{sanitized_path}"
        return sanitized_url.replace(".", "_")  # bug?


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve and save snapshots from the Wayback Machine for temporal analysis."
    )
    parser.add_argument(
        "--url-file",
        required=True,
        help="Path to urls text file.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--frequency",
        choices=list(FREQUENCY_MAP.keys()),
        default=DEFAULT_FREQUENCY,
        help="Frequency of collecting snapshots. Default is monthly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--snapshots-folder",
        default="snapshots",
        help="Path to the folder where snapshots will be saved.",
    )
    parser.add_argument(
        "--stats-folder",
        default="stats",
        help="Path to the folder where stats will be saved.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Extra logging statements used to debug.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    with open(args.url_file, "r") as file:
        urls = [line.strip() for line in file if line.strip()]

    client = WaybackMachineClient(
        args.num_workers, args.snapshots_folder, args.stats_folder
    )
    client.process_urls(urls, args.start_date, args.end_date, args.frequency)
