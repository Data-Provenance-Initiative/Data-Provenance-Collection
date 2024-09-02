# Analysis Scripts / Notebooks

This folder contains code relevant to conducting analysis for Data Provenance Initiative projects.

## Data & Annotations

- `pretrain_data` --- Raw data on C4, RefinedWeb, and Dolma, their URLS and their token counts.
- `agents_counter` --- Counts for various agents.
- `multimodal_terms_data` --- Annotations for the terms of use attached to multimodal data collections.
- `speech_supporting_data` --- Extra metadata for speech datasets.

## Analysis Notebooks

- `text_ft_plots.ipynb` --- Notebook to produce plots for [Data Provenance Initiative first paper](https://arxiv.org/pdf/2310.16787) on text finetuning dataset analysis.
- `robots_analysis.ipynb` --- Notebook to analyze the robots.txt and Terms of Service trends over time.
- `robots_analysis-tables-confusion-matrices-will.ipynb` --- Notebook to generate the [Consent in Crisis](https://www.dataprovenance.org/Consent_in_Crisis.pdf) papers' table comparing website features between the head and random distributions of the web, as well as robots.txt vs terms of service confusion matrices.
- `market_analysis.ipynb` --- Notebook to create plots for WildChat vs website purpose plots, used in the Consent in Crisis paper.
- `multimodal_analysis.ipynb` --- Notebook to compare text, video, and speech datasets.
- `paywall_domain_analysis.ipynb` --- Notebook to analyze the types of domains found behind paywalls.
- `prompt_analysis.ipynb` --- Notebook for plotting WildChat vs content domain type.
- `corpus_robots_trends.ipynb` --- Plotting robots and TOS domain specific trends.

## How to Run Notebooks

Download the folders listed below from the following GDrive: [Organized Raw Data](https://drive.google.com/drive/folders/1jfDAb0qKWZxMhGbCd4o1OIAscE3nc7tv?usp=share_link) and add them your local DPI repo in `src/analysis/data`.

- `domain_estimates` --- Domain estimate sheets for plotting in `corpus_robots_trends.ipynb`
- `GPT_analysis_results` --- GPT responses and grades for TOS policies.
- `raw_annotations` --- Raw annotation files required for plotting results in Consent in Crisis.
- `robots` --- Historical robots policies for the head and 10k random samples.

NOTE: `robots_analysis_p2.ipynb` --- Exploratory data analysis notebook for robots restrictions. None of the results / figures in this notebook were used in the final paper since the data was outdated. Thus, this notebook cannot be fully run but is included for completeness. Run `robots_analysis.ipynb` to reproduce results from the Consent in Crisis paper.
