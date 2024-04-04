import os
import argparse
from tqdm import tqdm
from collection_mapper import COLLECTION_FN_MAPPER
from helpers import io


def generate_bibtex(collection_name: str, save_to_file: bool = False, output_dir: str = None):
    """
    Generate BibTeX citations for all datasets in a collection.
    """
    DATA_DIR = 'data_summaries'
    # Open data collection metadata file
    collection_filepath = os.path.join(DATA_DIR, f"{collection_name}.json")
    assert os.path.exists(collection_filepath), f"There is no collection file at {collection_filepath}"
    collection_info = io.read_json(collection_filepath)

    all_successful = True
    bibtex_entries = ""
    for dataset_uid, dataset_info in collection_info.items():
        if 'Bibtex' in dataset_info:
            if save_to_file:
                # Use existing BibTeX entry if available
                bibtex_entries += dataset_info['Bibtex'] + '\n\n'
                continue
            else:
                print(f"Skipping {dataset_uid} as it already has a BibTeX citation")
                continue

        corpus_id = dataset_info.get('Semantic Scholar Corpus ID')

        if isinstance(corpus_id, str):
            print(f"No Semantic Scholar Corpus ID found for {dataset_uid}")
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

    if save_to_file and bibtex_entries:
        # Adjust the write_bib function call to include the output directory
        output_path = os.path.join(output_dir if output_dir else '.', 'refs.bib')
        io.write_bib(bibtex_entries, append=True, save_dir=output_path)
    elif not save_to_file:
        io.write_json(collection_info, collection_filepath)

    if all_successful:
        return f"Successfully generated BibTeX citations for all datasets in the {collection_name} collection."
    else:
        return f"BibTeX citations were not generated for all datasets in the {collection_name} collection."


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
        choices=list(COLLECTION_FN_MAPPER.keys()) + [None],
        help='Name of the collection to generate bibtex for')
    args = parser.parse_args()
    collections = [args.collection]
    collections = COLLECTION_FN_MAPPER.keys() if args.collection is None else [args.collection]

    for collection in tqdm(list(collections)):
        print(f"Generating bibtex for {collection} collection")
        result = generate_bibtex(collection)
        print(result)

