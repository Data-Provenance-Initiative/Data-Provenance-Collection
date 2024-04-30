import argparse
import logging
from pathlib import Path

from file_utils import extract_urls, parse_html_directories, save_data_to_json
from wayback_cdx import WaybackMachineClient


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve and save snapshots from the Wayback Machine for temporal analysis."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to a directory of CSV files containing URLs (assumes DPI annotations format).",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default="./temporal_data.json",
        help="Path to save the output JSON file with extracted text for all URLs.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="20160101",
        help="Start date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="20240419",
        help="End date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="monthly",
        choices=["daily", "weekly", "monthly", "annually"],
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
        type=Path,
        default="snapshots",
        help="Path to the folder where snapshots will be saved.",
    )
    parser.add_argument(
        "--stats-folder",
        type=Path,
        default="stats",
        help="Path to the folder where stats will be saved.",
    )
    parser.add_argument(
        "--count-changes",
        action="store_true",
        help="Count the number of unique changes for each site in the date range. WARNING: This requires many API calls, use for short date ranges or increase rate limit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Extra logging statements used to debug.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    urls = extract_urls(args.input_path)

    client = WaybackMachineClient(
        args.num_workers, args.snapshots_folder, args.stats_folder
    )
    client.process_urls(
        urls,
        args.start_date,
        args.end_date,
        args.frequency,
        args.count_changes,
    )
    client.save_failed_urls("failed_urls.txt")

    parsed_tos = parse_html_directories(args.snapshots_folder, args.input_path, args.num_workers)
    save_data_to_json(parsed_tos, args.output_json_path)
