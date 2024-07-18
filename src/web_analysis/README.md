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

This script collects historical versions of webpages and extracts their raw text/HTML content for analysis and optionally saves the data to a json. It uses the Wayback Machine and [CDX API](https://github.com/internetarchive/wayback/blob/master/wayback-cdx-server/README.md) to retrieve snapshots of websites at different time intervals (daily, monthly, or annually). Additionally, it can track the number of times a given webpage has changed its content within the specified time range. To install the required dependencies, use the `requirements_wayback.txt` file.

**Usage**

```
python -m web_analysis.wayback_extraction.temporal_pipeline --input-path <in-path> --snapshots-path snapshots --stats-path stats --output-json-path ./temporal_data.json --start-date 20160101 --end-date 20240419 --max-chunk-size 5000 --frequency monthly --num-workers 6 --site-type tos --save-snapshots --count-changes --process-to-json
```

**Arguments**

- `--input-path` (Path, required): Path to CSV file containing URLs (assumes DPI annotations format).
- `--output-json-path` (Path, default: `./temporal_data.json`): Path to save the output JSON file with extracted text for all URLs.
- `--start-date` (str, default: `"20160101"`): Start date in YYYYMMDD format.
- `--end-date` (str, default: `"20240419"`): End date in YYYYMMDD format.
- `--frequency` (str, default: `"monthly"`, choices: `["daily", "monthly", "annually"]`): Frequency of collecting snapshots.
- `--num-workers` (int, default: `6`): Number of worker threads.
- `--snapshots-path` (Path, default: `Path("snapshots")`): Path to the folder where snapshots will be saved.
- `--stats-path` (Path, default: `Path("stats")`): Path to the folder where rate of change stats will be saved.
- `--count-changes` (flag, default: `False`): Track rate of change by counting the number of unique changes for each site in the date range.
- `--process-to-json` (flag, default: `False`): Process the extracted snapshots and save them to a JSON file.
- `--save-snapshots` (flag, default: `False`): Whether to save and process snapshots from the Wayback Machine.
- `--site-type` (str, default: `"robots"`, choices: `["tos", "robots", "main"]`): Type of site to process (terms of service, robots.txt, or main page/domain).
- `--max-chunk-size` (int, default: `5000`): Chunk size (MB) for saving data to JSON file.

The only required argument is the input path to a CSV file with URLs.

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

See the `src/web_analysis/downloading_web/` folder for the scripts that process pretraining text data (C4, RefinedWeb, and Dolma).
These scripts map the high-level base domains (e.g. www.en.wikpiedia.org) to the number of scraped paths (e.g. www.en.wikipedia.org/wiki/The_Shawshank_Redemption) and the total number of text tokens across those paths.

### ChatGPT for Terms & Policies Analysis

This script facilitates the analysis of Terms of Service (ToS) and other usage policy documents using `gpt-4-turbo`. It is meant to determine the presence/type of policies related to scraping, AI usage, restrictions on competing services, illicit content, and licensing types. It handles data loading, optional data sampling, processing texts through the GPT model, and exporting results to CSV format for review. To install the required dependencies, use the `requirements_gpt_tos_analysis.txt` file. Note: to run this code you will need to add your [OpenAI API](https://platform.openai.com/docs/quickstart) key to a `.env` file located in the `\data` directory.

Example usage -

Using a full JSON dataset, run the `AI-policy` prompt and save output:

```
python gpt_tos_analysis.py --input_file_path "\path-to-data.json" --prompt_key "AI-policy" --output_file_path "\path-for-output-data.json"
```

Run the `type-of-license` prompt on a custom sample and save output to csv file:

```
python gpt_tos_analysis.py --input_sample_file_path "\path-to-sample-data.pkl" --prompt_key "type-of-license" --save_verdicts_to_csv True
```

Run the `scraping-policy` prompt on a custom sample, filter text without keywords:

```
python gpt_tos_analysis.py --input_sample_file_path "\path-to-sample-data.pkl" --prompt_key "scraping-policy" --filter_keywords True
```

Note -

Running this script requires the user to specify a `--prompt_key`. Key options are: `"scraping-policy", "AI-policy", "competing-services", "illicit-content", "type-of-license"`
