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