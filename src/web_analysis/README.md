# Web Analysis

This portion of the repository has scripts for analyzing web scrapes.

### Website Geolocation

This script identifies the IP address and country of origin for a list of web domains. Given a list of domains, it converts each URL to an IP address and matches that address to its host country. `website_geolocation.py` uses the IP2Location LITE database for [IP geolocation](https://lite.ip2location.com). To install the required dependencies, use the `requirements_website_geolocation.txt` file.

Example usage - 

Running the default URL list and writing the output to a file: 

```
python website_geolocation.py --write_file
```

Using a CSV file containing custom domains (note - URLs must be in the first column with no header):

```
python website_geolocation.py --url_csv "path/to/your/url_file.csv"
```

Manually specifying a custom list of URLs: 

```
python website_geolocation.py --custom_url_list "http://example.com" "http://example.org"
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

NOTE: The Wayback CDX API does not allow for filtering by week, so if you choose "weekly" the script will filtr using daily granularity and then aggregate the results into weeks manually in the post-processing step.

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

**Basic Plotting (WIP)**

Plot of basic output statistics after running historical extraction.
```
python src/web_analysis/plot.py <data-in-path> <png-out-path>
```


### Robots.txt Extraction & Parsing

These scripts (a) extract the robots.txt from a list of websites, and (b) interpret the robots.txt, producing aggregate statistics.

```
python src/web_analysis/extract_robots.py <in-path> <out-path>
```

```
python src/web_analysis/parse_robots.py <in-path> <out-path>
```

### Pretrain Corpora Analysis

See the `src/web_analysis/downloading_web/` folder for the scripts that process pretraining text data (C4, RefinedWeb, and Dolma).
These scripts map the high-level base domains (e.g. www.en.wikpiedia.org) to the number of scraped paths (e.g. www.en.wikipedia.org/wiki/The_Shawshank_Redemption) and the total number of text tokens across those paths.


### ChatGPT for Terms & Policies Analysis

This script facilitates the analysis of Terms of Service (ToS) and other usage policy documents using `gpt-4-turbo`. It is meant to determine the presence/type of policies related to scraping, AI usage, restrictions on competing services, illicit content, and licensing types. It handles data loading, optional data sampling, processing texts through the GPT model, and exporting results to CSV format for review. To install the required dependencies, use the `requirements_gpt_tos_analysis.txt` file. Note: to run this code you will need to add your [OpenAI API](https://platform.openai.com/docs/quickstart) key to a `.env` file located in the `\data` directory. 

Example usage - 

Using the default dataset, run the `AI-policy` prompt on a sample size of 100. Save sample as a `.plk` file for future reference:

```
python gpt_tos_analysis.py --save_sample True --sample_size 100 --prompt_key AI-policy
```

Run the `type-of-license` prompt on a custom sample:

```
python gpt_tos_analysis.py --custom_sample True --sample_file_path "\path-to-sample-data.pkl" --prompt_key type-of-license
```

Note - 

Running this script requires the user to specify a `--prompt_key`. Key options are: `"scraping-policy", "AI-policy", "competing-services", "illicit-content", "type-of-license"`