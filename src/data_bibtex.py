import os
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Union
from collection_mapper import COLLECTION_FN_MAPPER
from helpers import io
from glob import glob


def generate_bibtex(
        collection: Union[str, pd.DataFrame],
        data_dir: str = 'data_summaries',
        save_to_file: bool = False,
        output_dir: str = None,
        overwrite: bool = False,
):
    """
    Generate BibTeX citations for all datasets in a collection.
    """
    bibtex_entries = ""
    all_successful = True
    if isinstance(collection, pd.DataFrame):
        # Collection is a DataFrame, process directly
        for _, row in collection.iterrows():
            if 'Bibtex' in row and pd.notnull(row['Bibtex']) and row['Bibtex'].strip():
                if save_to_file:
                    bibtex_entries += str(row['Bibtex']) + '\n\n'
    else:
        # Open data collection metadata file
        collection_filepath = os.path.join(data_dir, f"{collection}.json")
        assert os.path.exists(collection_filepath), f"There is no collection file at {collection_filepath}"
        collection_info = io.read_json(collection_filepath)

        for dataset_uid, dataset_info in collection_info.items():
            if overwrite:
                print(f"Overwriting BibTeX for {dataset_uid}")
                dataset_info.pop('Bibtex', None)
            else:
                if 'Bibtex' in dataset_info:
                    if save_to_file:
                        # Use existing BibTeX entry if available
                        bibtex_entries += dataset_info['Bibtex'] + '\n\n'
                        continue
                    else:
                        print(f"Skipping {dataset_uid} as it already has a BibTeX citation")
                        continue

            corpus_id = dataset_info.get('Semantic Scholar Corpus ID')
            print(f"Generating BibTeX for {dataset_uid} (Corpus ID: {corpus_id})" if corpus_id else f"Generating BibTeX for {dataset_uid}")

            if isinstance(corpus_id, str):
                print(f"No Semantic Scholar Corpus ID found for {dataset_uid}")
                dataset_info['Bibtex'] = ""
                continue

            try:
                bibtex = io.get_bibtex_from_paper("CorpusId:{}".format(corpus_id))
                if save_to_file:
                    # Accumulate BibTeX entries in a string
                    bibtex_entries += bibtex + '\n\n'
                else:
                    dataset_info['Bibtex'] = bibtex
            except Exception as e:
                all_successful = False
                print(f"Error generating BibTeX for {dataset_uid}: {e}")
                dataset_info['Bibtex'] = ""

        if not save_to_file:
            io.write_json(collection_info, collection_filepath)

    if save_to_file and bibtex_entries:
        # Adjust the write_bib function call to include the output directory
        output_path = os.path.join(output_dir if output_dir else '.', 'refs.bib')
        io.write_bib(bibtex_entries, append=False, save_dir=output_path)

    if isinstance(collection, pd.DataFrame) or all_successful:
        return f"Successfully generated BibTeX citations for {collection} collection."
    else:
        return f"BibTeX citations were not generated for all datasets."


if __name__ == '__main__':
    """
    Example run:

    python src/data_bibtex.py --collection "Alpaca"

    """
    parser = argparse.ArgumentParser(description='Generate bibtex for a collection')
    parser.add_argument(
        '--collection',
        type=str,
        required=False,
        default=None,
        help='Name of the collection to generate bibtex for')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=False,
        default='data_summaries',
        help='Directory containing the collection metadata files')
    parser.add_argument(
        '--save_to_file',
        action='store_true',
        help='Save the BibTeX entries to a file')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        default=None,
        help='Directory to save the BibTeX file')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing BibTeX entries')
    args = parser.parse_args()
    DEFAULT_DATA_DIR = ['data_summaries', 'data_summaries-speech', 'data_summaries-video']

    if args.collection:
        assert args.data_dir, "Please provide the data directory"
        all_collections = [os.path.basename(x).split('.')[0] for x in glob(f"{args.data_dir}/*.json")]
        assert args.collection in all_collections, \
            f"Invalid collection. Change current data directory '{args.data_dir}' or choose from {all_collections}"
        collections = [args.collection]
    else:
        assert args.data_dir in DEFAULT_DATA_DIR, f"Invalid data directory. Choose from {DEFAULT_DATA_DIR}"
        collections = [os.path.splitext(os.path.basename(x))[0] for x in glob(f"{args.data_dir}/*.json")]

    for collection in tqdm(list(collections)):
        print(f"Generating bibtex for {collection} collection")
        result = generate_bibtex(collection, args.data_dir)
        print(result)

