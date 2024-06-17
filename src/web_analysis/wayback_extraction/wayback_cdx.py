import json
import logging
import os
import threading
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import requests
from dateutil.relativedelta import relativedelta
from ratelimit import limits, sleep_and_retry

from .file_utils import sanitize_url

BASE_URL = "https://web.archive.org"

FREQUENCY_MAP = {
    "daily": ("timestamp:8", "%Y-%m-%d", relativedelta(days=1)),
    "monthly": ("timestamp:6", "%Y-%m", relativedelta(months=1)),
    "annually": ("timestamp:4", "%Y", relativedelta(years=1)),
}

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="wayback_client.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class WaybackMachineClient:
    def __init__(
        self, num_workers: int, snapshots_path: str, stats_path: str, site_type: str
    ):
        self.num_workers = num_workers
        self.snapshots_path = snapshots_path
        self.stats_path = stats_path
        self.failed_urls = set()
        self.session = requests.Session()
        if site_type == "robots":
            self.session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                }
            )
        self.lock = threading.Lock()

    @sleep_and_retry
    @limits(calls=2, period=1)  # 2 calls per second
    def get_pages(
        self, url: str, start_date: str, end_date: str, frequency: str
    ) -> tuple[list[tuple[datetime, str, str]], dict]:
        """
        Retrieves snapshots (web pages) for a URL by using the Wayback Machine CDX API.
        CDX Documentation: https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server

        Filters
        -------
        - '&from={start_date}&to={end_date}': Specific date range.
        - '&filter=!statuscode:404': Exclude snapshots with status code 404.
        - '&filter=!mimetype:warc/revisit': Exclude snapshots with MIME type "warc/revisit" (revisit record without content).
        - '&collapse={collapse_filter}': Collapse feature to group snapshots based on desired frequency.
        - '&fl=timestamp,original,mimetype,statuscode,digest': Field list to include in the API response.

        Parameters
        ----------
        url : str
            The URL of the site to analyze.
        start_date : str
            The start date for the analysis in 'YYYYMMDD' format.
        end_date : str
            The end date for the analysis in 'YYYYMMDD' format.
        frequency : str
            The frequency of the snapshots to retrieve (daily, monthly, or annually).

        Returns
        -------
        tuple[list[tuple[datetime, str, str]], dict]: A tuple containing two elements:
            - A list of tuples, where each tuple represents a snapshot that contains:
                - Snapshot datetime object.
                - Snapshot URL.
                - Snapshot content (HTML).
        """
        results = []
        unique_digests = defaultdict(set)

        collapse_filter, date_format, delta = FREQUENCY_MAP[frequency]
        api_url = f"{BASE_URL}/cdx/search/cdx?url={url}&output=json&from={start_date}&to={end_date}&filter=!statuscode:404&filter=!mimetype:warc/revisit&collapse={collapse_filter}&fl=timestamp,original,mimetype,statuscode,digest"

        while True:
            try:
                response = self.session.get(api_url, allow_redirects=True)
                response.raise_for_status()
                data = response.json()

                if len(data) <= 1:
                    break

                header = data[0]
                field_indices = {field: index for index, field in enumerate(header)}

                for snapshot in data[1:]:
                    snapshot_date = datetime.strptime(
                        snapshot[field_indices["timestamp"]], "%Y%m%d%H%M%S"
                    )
                    snapshot_url = f"{BASE_URL}/web/{snapshot[field_indices['timestamp']]}/{snapshot[field_indices['original']]}"
                    snapshot_digest = snapshot[field_indices["digest"]]

                    interval_key = snapshot_date.strftime(date_format)

                    if snapshot_digest not in unique_digests[interval_key]:
                        unique_digests[interval_key].add(snapshot_digest)
                        snapshot_content = self.get_snapshot_content(snapshot_url)
                        if snapshot_content:
                            results.append(
                                (snapshot_date, snapshot_url, snapshot_content)
                            )

                if "next_page_url" not in data:
                    break
                api_url = data["next_page_url"]

            except requests.exceptions.RequestException as e:
                self.handle_error(url, e)
                break
            except IndexError as e:
                self.handle_error(url, f"IndexError: {e}")
                break

        return results

    @sleep_and_retry
    @limits(calls=2, period=1)
    def count_site_changes(
        self, url: str, start_date: str, end_date: str, frequency: str = "daily"
    ) -> int:
        """
        Counts the number of unique changes of a site within a given date range (rate of change).
        To do this we use the collapse=digest feature to count unique snapshots only.
        This requires daily frequency to collapse correctly - use sparingly.

        Parameters
        ----------
        url : str
            The URL of the site to analyze.
        start_date : str
            The start date for the analysis in 'YYYYMMDD' format.
        end_date : str
            The end date for the analysis in 'YYYYMMDD' format.

        Returns
        --------
        int : The number of unique changes.
        """
        collapse_filter, _, _ = FREQUENCY_MAP[frequency]
        api_url = f"{BASE_URL}/cdx/search/cdx?url={url}&from={start_date}&to={end_date}&output=json&filter=mimetype:text/html&collapse={collapse_filter}"

        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            data = response.json()
            return len(data) - 1 if data else 0  # don't include the header
        except requests.exceptions.RequestException as e:
            self.handle_error(url, e)
            return 0

    @sleep_and_retry
    @limits(calls=2, period=1)
    def get_snapshot_content(self, snapshot_url: str) -> Optional[str]:
        """
        Retrieves the content of a snapshot from a given snapshot URL.

        Parameters
        ----------
        snapshot_url : str
            The URL of the snapshot to retrieve.

        Returns
        -------
        Optional[str] : The content of the snapshot, or None if an error occurs.
        """
        try:
            response = self.session.get(snapshot_url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.handle_error(snapshot_url, e)
            return None

    def process_url(
        self,
        url: str,
        start_date: str,
        end_date: str,
        frequency: str,
        count_changes: bool = False,
    ) -> None:
        """
        Processes a single URL to retrieve and save snapshots, and optionally count site changes.

        Parameters
        ----------
        url : str
            The URL to process.
        start_date : str
            The start date for snapshot retrieval in 'YYYYMMDD' format.
        end_date : str
            The end date for snapshot retrieval in 'YYYYMMDD' format.
        frequency : str
            The frequency of snapshots to retrieve.
        count_changes : bool = False
            Whether to track rate of change.
        """
        sanitized_url = sanitize_url(url)
        url_folder = os.path.join(self.snapshots_path, sanitized_url)

        snapshots_exist = os.path.exists(url_folder) and len(os.listdir(url_folder)) > 0

        results = []
        if not snapshots_exist:
            results = self.get_pages(url, start_date, end_date, frequency)
            if results:
                for snapshot_date, snapshot_url, snapshot_content in results:
                    self.save_snapshot(url, snapshot_date, snapshot_content)

        start_datetime = datetime.strptime(start_date, "%Y%m%d")
        end_datetime = datetime.strptime(end_date, "%Y%m%d")
        current_date = start_datetime

        _, date_format, delta = FREQUENCY_MAP[frequency]

        stats_filename = f"{sanitized_url}.json"
        stats_path = os.path.join(self.stats_path, stats_filename)

        stats_exist = os.path.exists(stats_path)

        if count_changes and not stats_exist:
            stats = {"url": url, "change_counts": {}}
            while current_date <= end_datetime:
                frequency_start = current_date.strftime("%Y%m%d")
                frequency_end = (current_date + delta - relativedelta(days=1)).strftime(
                    "%Y%m%d"
                )

                frequency_change_count = self.count_site_changes(
                    url, frequency_start, frequency_end, "daily"
                )
                stats["change_counts"][
                    current_date.strftime(date_format)
                ] = frequency_change_count

                current_date = current_date + delta

            self.save_stats(url, stats)

        start_date_formatted = start_datetime.strftime("%m-%d-%Y")
        end_date_formatted = end_datetime.strftime("%m-%d-%Y")

        if snapshots_exist:
            logging.info(
                f"Skipping saving snapshots for {url} between {start_date_formatted} and {end_date_formatted} as they already exist."
            )
        elif not results:
            logging.info(
                f"No snapshots available for {url} between {start_date_formatted} and {end_date_formatted}"
            )
        else:
            logging.info(
                f"Processed snapshots for {url} between {start_date_formatted} and {end_date_formatted}"
            )

    def process_urls(
        self,
        urls: list[str],
        start_date: str,
        end_date: str,
        frequency: str,
        count_changes: bool = False,
    ) -> None:
        """
        Processes a list of URLs to retrieve and save snapshots, and optionally count site changes.

        Parameters
        ----------
        urls : list[str]
            The list of URLs to process.
        start_date : str
            The start date for snapshot retrieval in 'YYYYMMDD' format.
        end_date : str
            The end date for snapshot retrieval in 'YYYYMMDD' format.
        frequency : str
            The frequency of snapshots to retrieve.
        count_changes : bool = False
            Whether to count site changes.
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self.process_url,
                    url,
                    start_date,
                    end_date,
                    frequency,
                    count_changes,
                ): url
                for url in urls
            }

            with tqdm(total=len(urls), desc="Processing URLs") as pbar:
                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.handle_error(url, e)
                    pbar.update(1)

    def save_snapshot(
        self, url: str, snapshot_date: datetime, snapshot_content: str
    ) -> None:
        """
        Saves a snapshot's content to a file.

        Parameters
        ----------
        url : str
            The URL of the snapshot.
        snapshot_date : datetime
            The date of the snapshot.
        snapshot_content : str
            The content of the snapshot.
        """
        sanitized_url = sanitize_url(url)
        url_folder = os.path.join(self.snapshots_path, sanitized_url)
        os.makedirs(url_folder, exist_ok=True)
        snapshot_filename = f"{snapshot_date.strftime('%Y%m%d%H%M%S')}.html"
        snapshot_path = os.path.join(url_folder, snapshot_filename)
        with open(snapshot_path, "w", encoding="utf-8") as file:
            file.write(snapshot_content)
        logging.info(f"Snapshot saved as {snapshot_path}")

    def save_stats(self, url: str, stats: dict) -> None:
        """
        Saves the stats for a URL to a file.

        Parameters
        ----------
        url : str
            The URL for which stats were collected.
        stats : dict
            The stats data to save.
        """
        os.makedirs(self.stats_path, exist_ok=True)
        sanitized_url = sanitize_url(url)
        stats_filename = f"{sanitized_url}.json"
        stats_path = os.path.join(self.stats_path, stats_filename)
        with open(stats_path, "w") as file:
            json.dump(stats, file, indent=4)
        logging.info(f"Stats saved as {stats_path}")

    def handle_error(self, url: str, error: Exception) -> None:
        """
        Handles errors that occur during processing by logging them and saving the failed URLs.

        Parameters
        ----------
        url : str
            The URL that caused the error.
        error : Exception
            The error that occurred.
        """
        with self.lock:
            self.failed_urls.add((url, str(error)))
        logging.error(f"Error processing {url}: {error}")
        self.save_failed_urls_async()

    def save_failed_urls_async(self) -> None:
        """
        Starts a separate thread to save the failed URLs to a file.
        """
        thread = threading.Thread(target=self.save_failed_urls)
        thread.start()

    def save_failed_urls(self, filename: str = "failed_urls.txt") -> None:
        """
        Saves the failed URLs to a file.
        """
        with self.lock:
            if self.failed_urls:
                with open(filename, "a") as f:
                    for url, error in self.failed_urls:
                        f.write(f"{url} --> error: {error}\n")
                logging.info(f"Failed URLs saved to {filename}")
                self.failed_urls.clear()
