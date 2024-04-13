## Web Analysis

This portion of the repository has scripts for analyzing web scrapes.

### Website Geolocation

This script identifies the IP address and country of origin for a web domain.

```
<run command>
```

### Website Temporal Extraction

This script collects historical versions of webpages and extracts their raw text for analysis, using the Wayback Machine.

```
<run command>
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