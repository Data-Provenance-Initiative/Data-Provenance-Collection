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

This script collects historical versions of webpages and extracts their raw text for analysis, using the Wayback Machine.

```
<run command>
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

### ChatGPT for Terms & Policies Analysis

This script facilitates the analysis of Terms of Service (ToS) and other usage policy documents using `gpt-4-turbo`. It is meant to determine the presence/type of policies related to scraping, AI usage, restrictions on competing services, illicit content, and licensing types. It handles data loading, optional data sampling, processing texts through the GPT model, and exporting results to CSV format for review. To install the required dependencies, use the `requirements_gpt_tos_analysis.txt` file.

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