import csv
import re
import urllib.parse
from pathlib import Path


def sanitize_url(url: str) -> str:
    """Sanitizes URL to be used as a folder name."""
    parsed_url = urllib.parse.urlparse(url)
    sanitized_netloc = parsed_url.netloc.replace(".", "_")
    sanitized_path = "_".join(
        filter(None, re.split(r"\/+", parsed_url.path.strip("/")))
    )
    sanitized_url = f"{sanitized_netloc}_{sanitized_path}"
    return sanitized_url.replace(".", "_")  # bug?


def extract_urls_from_csv(csv_path: Path) -> list[str]:
    """Extracts URLs from a single CSV file."""
    with csv_path.open(mode="r") as file:
        csv_reader = csv.DictReader(file)
        return list(
            {
                row.get("url")
                for row in csv_reader
                if row.get("url") and row.get("url").startswith("http")
            }
        )


def extract_urls_from_txt(txt_path: Path) -> list[str]:
    """Extracts URLs from a single text file."""
    with txt_path.open(mode="r") as file:
        return [line.strip() for line in file if line.strip()]


def extract_urls_from_directory(directory_path: Path) -> list[str]:
    """Extracts URLs from all CSV and text files within a directory."""
    urls = set()
    for file_path in directory_path.glob("*"):
        if file_path.suffix == ".csv":
            urls.update(extract_urls_from_csv(file_path))
        elif file_path.suffix == ".txt":
            urls.update(extract_urls_from_txt(file_path))
    return list(urls)


def extract_urls(input_path: Path) -> list[str]:
    """Determines how to extract URLs based on the nature of the path (directory, CSV file, or text file)."""
    if input_path.is_dir():
        return extract_urls_from_directory(input_path)
    elif input_path.is_file():
        if input_path.suffix == ".txt":
            return extract_urls_from_txt(input_path)
        elif input_path.suffix == ".csv":
            return extract_urls_from_csv(input_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}")