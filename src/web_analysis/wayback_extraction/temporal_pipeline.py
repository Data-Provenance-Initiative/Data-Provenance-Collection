import argparse
from pathlib import Path
from colorama import init, Fore, Style

from .file_utils import extract_urls, parse_html_directories
from .wayback_cdx import WaybackMachineClient

init(autoreset=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve and save snapshots from the Wayback Machine for temporal analysis and/or collect rate of change data."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to a directory of CSV files containing URLs (assumes DPI annotations format).",
    )
    parser.add_argument(
        "--output-json-path",
        type=Path,
        default=Path("./temporal_tos.json"),
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
        default="annually",
        choices=["daily", "monthly", "annually"],
        help="Frequency of collecting snapshots. Default is monthly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--snapshots-path",
        type=Path,
        default=Path("snapshots"),
        help="Path to the folder where snapshots will be saved.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("stats"),
        help="Path to the folder where rate of change stats will be saved.",
    )
    parser.add_argument(
        "--count-changes",
        action="store_true",
        help="Track rate of change by counting the number of unique changes for each site in the date range.",
    )
    parser.add_argument(
        "--process-to-json",
        action="store_true",
        help="Process the extracted snapshots and save them to a JSON file.",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Whether to save and process snapshots from the Wayback Machine.",
    )
    parser.add_argument(
        "--site-type",
        type=str,
        default="main",
        choices=["tos", "robots", "main"],
        help="Type of site to process (terms of service or robots.txt). If type is main, we will process the main page/domain of the site.",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1000,
        help="Chunk size (MB) for saving data to JSON file. Default is 1000 MB.",
    )
    return parser.parse_args()


def print_colored(message: str, color: str) -> None:
    # wayback_machine logger saves all import errors to wayback_client.log
    print(color + message + Style.RESET_ALL)


if __name__ == "__main__":
    args = parse_arguments()

    print_colored("\nTemporal Pipeline", Fore.CYAN)
    print("\nArgs\n----")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    urls = extract_urls(args.input_path, args.site_type)

    if args.save_snapshots:
        client = WaybackMachineClient(
            args.num_workers,
            args.snapshots_path,
            args.stats_path,
            args.site_type,
        )

        print_colored(
            f"\nStarting WaybackMachineClient processing with {len(urls)} URLs...",
            Fore.GREEN,
        )

        client.process_urls(
            urls,
            args.start_date,
            args.end_date,
            args.frequency,
            args.count_changes,
        )

        client.save_failed_urls("failed_urls.txt")
        print_colored("Failed URLs saved to failed_urls.txt", Fore.YELLOW)

    if args.process_to_json:
        print(f"Parsing HTML directories...")
        parse_html_directories(
            args.snapshots_path,
            "/Users/ariel-raive/Desktop/Data-Provenance-Collection/src/web_analysis/wayback_extraction/robots/robots_head.csv",
            args.site_type,
            args.num_workers,
            max_chunk_size=(
                args.max_chunk_size * 1024 * 1024 if args.max_chunk_size else None
            ),  # MB
            output_file=args.output_json_path,
        )
        # TODO: fix chunking file names
        print_colored(f"Parsed data saved to {args.output_json_path}", Fore.GREEN)
