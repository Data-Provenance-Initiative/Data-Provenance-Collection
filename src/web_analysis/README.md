# Web Analysis

This portion of the repository has scripts for analyzing web scrapes.

### Website Geolocation

This script identifies the IP address and country of origin for a web domain.

```
<run command>
```

### Website Temporal Extraction

This script collects historical versions of webpages and extracts their raw text/HTML content for analysis. It uses the Wayback Machine and [CDX API](https://github.com/internetarchive/wayback/blob/master/wayback-cdx-server/README.md) to retrieve snapshots of websites at different time intervals (daily, weekly, monthly, or yearly). Additionally, it tracks the number of times a given webpage has changed its content within the specified time range. To install the required dependencies, use the `requirements_wayback.txt` file.
```
python src/web_analysis/wayback_cdx.py --url-file <in-path> --start-date <YYYYMMDD> --end-date <YYYYMMDD>
```
**Arguments**

- `--url-file` (required): Path to a text file containing a list of URLs, one per line.
- `--start-date` (required): Start date in the YYYYMMDD format.
- `--end-date` (required): End date in the YYYYMMDD format.
- `--frequency` (optional): Frequency of data collection. Options are "daily", "weekly", "monthly", or "annual" (default is "monthly").
- `--num-workers` (optional): Number of worker threads (default is 10).
- `--snapshots-folder` (optional): Directory to save the website snapshots (default is "snapshots").
- `--stats-folder` (optional): Directory to save statistical data (default is "stats").
- `--debug` (optional): Enable detailed logging and/or debugging.

**Example**

Here's a concrete example using the `urls_example.txt` file in this folder: 
```
python src/web_analysis/wayback_cdx.py --url-file urls_example.txt --start-date 20230101 --end-date 20230107 --frequency daily
```
This command creates two directories: `snapshots` and `stats`.

1. The `snapshots` folder contains subfolders for each website, with sanitized URLs as folder names (e.g., `www_bloomberg_com_tos`). Inside each subfolder, there are daily snapshots of the website in HTML format (e.g., `20230102204130.html`).

2. The `stats` folder contains JSON files with statistical data about the website changes. For example, for the site `https://www.bloomberg.com/tos/`, the corresponding JSON file `www_bloomberg_com_tos_stats.json` has following information:

```json
{
  "url": "https://www.bloomberg.com/tos/",
  "change_counts": {
    "2023-01-01": 0,
    "2023-01-02": 1,
    "2023-01-03": 1,
    "2023-01-04": 0,
    "2023-01-05": 1,
    "2023-01-06": 0,
    "2023-01-07": 1
  }
}
```
This indicates that the content of the website changed 4 times within the specified date range (01-01-2023 to 01-07-2023).

**Rate Limiting**

To avoid overwhelming sites and respect rate limits, this script uses the `ratelimit` library to limit the number of requests to 3 requests per second. 

If you need to adjust the rate limit, modify the `@limits` decorator in the `get_pages`, `get_snapshot_content`, and `count_site_changes` methods.


### Robots.txt Extraction & Parsing

These scripts (a) extract the robots.txt from a list of websites, and (b) interpret the robots.txt, producing aggregate statistics.

```
python src/web_analysis/extract_robots.py <in-path> <out-path>
```

```
python src/web_analysis/parse_robots.py <in-path> <out-path>
```

### Pretrain Corpora Analysis

These scripts process a dump of pretraining text data (C4, RefinedWeb, and Dolma). We map the high-level base domains (e.g. www.en.wikpiedia.org) to the number of scraped paths (e.g. www.en.wikipedia.org/wiki/The_Shawshank_Redemption) and the total number of text tokens across those paths.

```
<C4 run command>
```


```
<RefinedWeb run command>
```


```
<Dolma run command>
```
