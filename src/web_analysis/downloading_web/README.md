# Download Web-Scale Datasets

There are a variety of web-scale composition datasets that can be downloaded, from the original common crawl through to cleaned versions (like C4) or meta-collections (such as Dolma), which further clean and bundle training data. 

Here, we show examples of downloading from the cleaned datasets (no pre-trainer should use the raw common crawl data). The scripts range from the simplest, C4, which uses huggingface streaming in a single loop, to RefinedWeb, which has a simple and easy restart mechanism, and to Dolma, which uses multithreading for speed. 

The examples shown here use simple string tokenizing for speed, but with sufficient compute you can use more sophisticated tokenizers. 
