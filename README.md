# The Data Provenance Initiative

**Paper (coming soon)** | [**Data Provenance Explorer**](https://www.dataprovenance.org/)

[**Setup**](#setup) | [**Run**](#run) | [**Collected Information**](#collected-information) | [**Add a Collection**](#add-a-collection) | [**Dataset Format**](#dataset-format) | [**Legal Notice & Limitations**](#legal-notice-and-limitations) | [**Contact and Citation**](#contact-and-citation)


<p align="center">
  <img src="dpi.png" width="40%" height="40%" alt="Data Provenance Initiative">
</p>

The Data Provenance Initiative is a multi-disciplinary volunteer effort to improve transparency, documentation, and responsible use of training datasets for AI.
Through a large scale audit of 44 data collections, spanning 1800+ finetuning text-to-text datasets, referred to as the Data Provenance Collection, this initiative's first release thoroughly documents their web and machine sources, licenses, creators, and other metadata.
The scripts in this repository allow developers to filter for the finetuning datasets that best fit their requirements, from self-reported license constraints to other data characteristics.
Check out the [Data Provenance Explorer](https://www.dataprovenance.org/) to view the data composition, and affect of different filters you can apply in this repository.
Generating a subset of data from this repository will create a `Data Provenance Card`---symbolic attribution for all constituent datasets, that can be used as structured documentation.
Full details are available in our paper (coming soon).

**This repository does NOT constitute legal advice.**
[**See Legal Notice and Limitations**](#legal-notice-and-limitations).

This is the first step of the initiative, and we plan to expand the resources and tools, as well as the academic analysis. If you would like to contribute, reach out to us at [data.provenance.init@gmail.com](mailto:data.provenance.init@gmail.com).


## Setup
```
pip install -r requirements.txt
```

## Run

This script will allow you to download all datasets, filtered for any criteria in [Collected Information](#collected-information).
Either pass in arguments via argparse, or your own yaml config, similar to `src/configs/default.yaml`.
All datasets will be normalized to the same format.
By default, we use this [format](#dataset-format) as it generalizes to multi-turn dialog and response rankings, but you can also normalize the datasets for supervised finetuning.

```
python src/download_and_filter.py -c src/configs/default.yaml
```

## Collected Information

#### Identifier Information

* **Unique Dataset Identifier** --- A unique identifier for the dataset, formatted as <collection>-<dataset>. 
* **Dataset Name** --- The common name for the dataset.
* **Paper Title** --- The title of the associated paper, if available.
* **Dataset URL** --- The URL to the dataset's original github repository, website, or best available resource.
* **GitHub URL** --- The URL to the dataset's original github repository, if available.
* **Hugging Face URL** --- The URL to the dataset's Hugging Face page.
* **Papers with Code URL** --- The URL to the dataset's 'Papers with Code' page.
* **ArXiv URL** --- The URL to the ArXiv abstract page if the dataset is accompanied by a paper.
* **Semantic Scholar Corpus ID** --- The Corpus ID associated with the dataset's paper on semantic scholar.
* **Collection** --- The collection from which this version of the dataset is a part.
* **Collection URL** --- The URL to the collection's github repository or website.

#### Dataset Characteristics

* **Languages** --- A list of the natural or coding languages represented in the text.
* **Task Categories** --- A list of the tasks represented in the dataset.
* **Text Sources** --- A list of web/offline sources from which the text was derived.
* **Text Topics** --- A list of the topics discussed in the text.
* **Text Metrics**  --- Statistics on the minimum, mean, and maximum characters for each example's input and output, and the number of dialog turns if applicable.
* **Format** --- Options are 'Zero-shot Prompt', 'Few-shot Prompt', 'Chain-of-Thought Prompt', 'Multi-turn Dialog', 'Response Rankings', or some other combination, if applicable.
* **Time of Collection** --- Hugging Face, Semantic Scholar, and GitHub initial upload times, for when the dataset was created.
* **Citation Count** --- If on Semantic Scholar, the recorded citation count as of September 2023.
* **Download Count** --- If on Hugging Face, the recorded download count for the month of September 2023.
* **Dataset Filter IDs** -- A list of the dataset filter IDs associated with the Hugging Face dataset, it is applicable when the same collection has multiple sources/languages/tasks and they can be separated.

#### Dataset Provenance

* **Model Generated** -- The model used to generate this dataset if applicable.
* **Text Sources** --- A list of web/offline sources from which the text was scraped or derived. 
* **Text Domains** --- From the text sources, you can infer the `Text Domain` (e.g. biomedical, legal, encyclopedia, book, etc) using `constants/domain_groups.json`.
* **Derived from Datasets** --- A list of other machine learning datasets from which this dataset is derived.
* **Human Annotation** --- Whether additional human annotation was part of the dataset curation process, after the data was taken from its text sources.
* **Creators** --- A list of the universities, corporations and other organizations that compiled (or created) this dataset.
* **Licenses** --- A list of the licenses associated with this dataset by the creators/compilers of the NLP dataset, including their URLs.
* **License Conditions** --- Using the above, and `constants/license_classes.json` we categorize every licenses use restrictions, attribution and sharealike requirements.
* **License Notes** --- Any notes from the human annotator regarding the license information they retrieved.


## Add a Collection

Instructions:

1. For a new collection add a json file in `data_summaries/`. Try to enumerate all datasets or separable splits of the dataset (by language, task, license, source, etc) as separate entities in the json file, each with their own `Unique Dataset Identifier`. 
Please refer to the `data_summaries/_template.json` for reference and info on required fields.
2. Write a downloader function for your collection in `src/downloaders.py`.
The downloader should download the collection, and filter out any subsets not represented by the `accepted_filter_ids` argument. 
This argument corresponds to the `"Dataset Filter IDs"` for each unique dataset, that can be used to filter a Hugging Face dataset object based on some column, e.g. `source`, that it might have.
This downloader returns a list, of any format, to be parsed in the next step.
The `"Dataset Filter ID"` should also be a field on each row, so we can map examples back to their origin later.


```
def download_<your_collection>(accepted_filter_ids):
    """Downloads your datasets and filters to the subset in `accepted_filter_ids`.

    accepted_filter_ids: A list of `"Dataset Filter IDs"` from the dataset summary files
        whose `"Unique Dataset Identifier"` that passed the filters applied at 
        runtime on license/language/task/etc. Use these to partition the downloaded 
        dataset into just the relevant data points.

    Returns a list of rows (any format), representing the dataset. 

    For instance, the Flan Collection has 100s of datasets with different licenses.
    `accepted_filter_ids` will be the list of remaining datasets after our license
    filtering, and they should corresond to example tags, in the huggingface download.
    """
    # Download your dataset, from Hugging Face using this helper function, e.g.:
    dset = huggingface_download('nomic-ai/gpt4all-j-prompt-generations', split='train')
    # Or download your dataset directly from a URL, using this helper function, e.g.:
    dset = direct_data_request("https://raw.githubusercontent.com/<your-repo>/data.json")
    # You can return the dset now, or you can further filter the downloaded dataset (see below).
    return dset

    # Next, we filter to the relevant data subsets, represented by `accepted_filter_ids`.
    # If there are multiple datasets in `dset`, then the ones 
    # that passed the filters will have their `dataset_filter_key` passed into `accepted_filter_ids`.
    # You can use `pool_filter` to filter the dataset in parallel (see documentation).
    # 'source' is an example of `task_key` as defined in `pool_filter` in downloader.py
    return pool_filter(dset, "source", accepted_filter_ids)
```

3. Write a data preparer function for your collection in `src/preparers.py`.
The preparer function will take in the output of your custom downloader function, and format the data like this (also explained [here](#dataset-format)):

Here is an example of how you could write your dataset preparer:
```
def prepare_<your_collection>(row):
    """ 
        The preparer function could use a function      `convert_inputs_taergets_to_messages` that takes the output from your   custom downloader function, and format the data like this:

        A list of messages in a dialog: [{"from": <>, "text": <>, "parent": <>, "score": <>}, ...]

        'from' is either 'user' or 'assistant'
        'text' is the content of a single message
        'parent' is the 0-indexed ID of the preceding message in the dialog list
        'score' (optional) is the score of that response, if applicable

        You can also customize your own preparer processing function as long as they have the same output format.
    """

    return convert_inputs_targets_to_messages(row['prompt'], row['response'],'<dset_name>')


```

4. Add your collection to the `COLLECTION_FN_MAPPER` in `src/collection_mapper.py` in the following format:
```
"<Your Dataset File Name (without extension)>":{
    "download_function": downloaders.download_<your_collection>,
    "prepare_function": "preparers.prepare_<your_collection>",
}

```


5. Run the following tests to confirm your code works (or help debug it):

```
python src/test_new_collection.py --collection "<your collection>"
```

If you want to run everything and print all errors collectively, use `--no_halt` after the previous command:
```
python src/test_new_collection.py --collection "<your collection>" --no_halt
``` 
## Dataset Format

Our dataset is structured as a list of dictionaries, with each dictionary representing an individual message in a dialog. Every message dictionary contains the following fields:

* **"from"**: This denotes the sender and can have values "user" or "assistant".
* **"text"**: This provides the actual content of the message.
* **"parent"**: For a message that initiates the conversation, this field contains the ID of the dataset. For subsequent messages, it holds the 0-indexed position of the message to which it is replying within the list.
* **"score"** (optional): Some messages, especially responses, might come with a score. This helps in scenarios where there are multiple competing responses to a single message, showcasing potential conversation branches.

Messages are typically organized in pairs where an "assistant" responds to a "user" and vice versa, but there can be multiple replies to a single message, indicating diverging paths in the dialogue.

```
[
    {
        "from": "user",
        "text": "Hello, how are you?",
        "parent": "dataset1234"
    },
    {
        "from": "assistant",
        "text": "I'm good! How can I assist you?",
        "parent": 0,
        "score": 1
    },
    {
        "from": "assistant",
        "text": "Hello!",
        "parent": 0,
        "score": 0
    },
    ...
]
```


## Legal Notice and Limitations

The Data Provenance Initiative is a research effort to increase transparency in machine learning.
The information provided on this page and any output of the Data Provenance Initiative does not, and is not intended to, constitute legal advice; instead, all information, content, and materials are for general informational purposes only.
No reader, user, or browser of this project should act or refrain from acting on the basis of information from the Data Provenance Initiative without first seeking legal advice from counsel in the relevant jurisdiction.

The code in this repository is Apache 2.0 licensed.

## Contact and Citation

Contact [data.provenance.init@gmail.com](mailto:data.provenance.init@gmail.com) to update or contribute to this resource.

```
Citation coming soon.
```
