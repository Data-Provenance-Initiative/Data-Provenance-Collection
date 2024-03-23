import os
from collections import defaultdict
from tabulate import tabulate
from datetime import datetime

from helpers import io

def generate_datacard(
    data_summary, 
    selected_licenses,
    selected_languages,
    selected_task_categories,
    savedir,
):
    """
    HuggingFace Dataset Cards: https://huggingface.co/docs/hub/datasets-cards
        - Composition of languages, task categories, sources (num exs)

    Datasheets for Datasets: https://arxiv.org/pdf/1803.09010.pdf
        - Composition of languages, task categories, sources (num exs)
        - How was it subsetted?
        - What is the raw data?
        - Raw data use and intention? (Supervised, RLHF, etc)
        - Redundancies in the dataset?
        - Privacy, confidential, or licensing composition?
        - Sensitive characteristics / bias in the data?

    Model Cards: https://arxiv.org/pdf/1810.03993.pdf
        - Evaluation Data. Details on the dataset(s) used for the
        quantitative analyses in the card.
            - Datasets
            - Motivation
            - Preprocessing
        - Training Data. May not be possible to provide in practice.
        When possible, this section should mirror Evaluation Data.
        If such detail is not possible, minimal allowable information
        should be provided here, such as details of the distribution
        over various factors in the training datasets.
    """
    # High level summary
    num_datasets = len(data_summary)
    total_exs = sum([row.get("Num Dialogs", 0) if row else 0 for row in data_summary["Text Metrics"].tolist()])

    def summarize_column(df, col):
        col_to_exs = defaultdict(lambda: [0, 0])
        for i, row in data_summary.iterrows():
            if isinstance(col, list):
                items = [in_row[col[1]] for in_row in row[col[0]]]
            else:
                items = row[col]

            if isinstance(items, list):
                for item in items:
                    col_to_exs[item][0] += 1
                    if row.get("Text Metrics", {}):
                        col_to_exs[item][1] += int(row.get("Text Metrics", {}).get("Num Dialogs", 1) / len(items))
            else:
                col_to_exs[items][0] += 1
                if row.get("Text Metrics", {}):
                    col_to_exs[items][1] += row.get("Text Metrics", {}).get("Num Dialogs", 1)
                
        # dictionary: item --> (num datasets, num exs)
        return sorted(col_to_exs.items(), key=lambda x: x[1][0], reverse=True)

    collection_dist = summarize_column(data_summary, "Collection")
    lang_dist = summarize_column(data_summary, "Languages")
    taskcat_dist = summarize_column(data_summary, "Task Categories")
    license_dist = summarize_column(data_summary, ["Licenses", "License"])

    def tabulate_present(header, dist):
        out = f"### {header}\n\n"
        headers = [header[:-1], "Datasets", "Examples"]
        total_datasets = sum([row[1][0] for row in dist])
        total_exs = sum([row[1][1] for row in dist])
        table = []
        for item, (n_dsets, n_exs) in dist:
            table.append(
                [
                    item,
                    f"{n_dsets} ({round(100 * n_dsets / total_datasets, 2)} %)",
                    f"{n_exs} ({round(100 * n_exs / total_exs, 2) if n_exs else 0} %)",
                ]
            )
        # <Item>: <num_datasets> (%) | <num exs> (%)
        out += tabulate(table, headers, tablefmt="simple")
        return out

    # Begin writing dataset card
    data_card = ["## Data Card"]

    selection_txt = "\n\n".join([
        "The authors select their datasets according to the following criteria:",
        f"Languages: {str(selected_languages)}",
        f"Task Categories: {str(selected_task_categories)}",
        f"Licenses: {str(selected_licenses)}",
        f"The filtering and selection was conducted using tools at https://github.com/shayne-longpre/opal-dl on {datetime.today().strftime('%Y-%m-%d')}.",
    ])
    data_card.append(selection_txt)

    # Collection composition
    data_card.append(tabulate_present("Collections", collection_dist))
    # Language composition
    data_card.append(tabulate_present("Languages", lang_dist))
    # Task Category composition
    data_card.append(tabulate_present("Task Categories", taskcat_dist))
    # License composition
    data_card.append(tabulate_present("Licenses", license_dist))

    limitations_txt = """NB: Num examples and percentages are approximated."""
    data_card.append(limitations_txt)
    
    data_card = "\n\n\n\n".join(data_card)
    io.write_txt(os.path.join(savedir, "data_card.txt"), data_card)

    # Full CSV with links for attribution to sources and licenses
    # keep_cols = [
    #     "Unique Dataset Identifier", "Dataset Name", "Dataset URL", "HuggingFace URL", 
    #     "Collection", "Collection URL", "Languages", "Text Source",
    #     "Task Categories", "Licenses", "License Notes",
    #     "Num Instances",
    # ]
    keep_cols = list(data_summary.columns)
    data_summary[keep_cols].to_csv(os.path.join(savedir, "data_attribution.csv"), index=False)