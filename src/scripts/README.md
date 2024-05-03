# Infer metadata
This script queries HuggingFace, Semantic Ccholar, Paperswithcode and GitHub to find data provenance metadata (licenses) and other metadata (citation counts, number of downloads...).
### Usage (within repo root):
`python src/scripts/infer_metadata.py --collections ExpertQA --github_token YOURTOKEN`
This will add or update a `Inferred Metadata` to the `data_summaries` entry.
### Token info
#### Github token (required to fetch github information)
https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28
You can also omit the github token but it will not fill github fields.
#### Semantic scholar token (not required, but recommended when there are many datasets, N>100)
Use the --sch_api_key argument
You can request an API key here https://www.semanticscholar.org/product/api#api-key-form
### Requirements:
`pip install datasets ratelimit humanfriendly jsonlines funcy semanticscholar tenacity -q`
### Colab:
https://colab.research.google.com/drive/1btjynODfCIbuq0c1Wx4FveiM4lWTkWZ_?usp=sharing
