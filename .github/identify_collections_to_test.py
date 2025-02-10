import os
import json
import fnmatch
import subprocess


def get_changed_files():
    return subprocess.check_output([
        'git', 'diff',
        '--name-only', 'origin/main...'
    ]).decode().splitlines()


def get_collections_from_file(file):
    with open(file, 'rt') as f:
        data = json.load(f)

    collections = []
    for k in data.keys():
        if 'Collection' in data[k].keys():
            collections += [data[k]['Collection']]

    # we're only running in response to changes in data_summaries/*.json files,
    # so if there's such a file without a collection field we don't like that
    # and should fail the test
    assert len(collections) > 0

    return collections


def main():
    changed_files = get_changed_files()

    collections_to_test = []
    for file in changed_files:
        if not fnmatch.fnmatch(file, 'data_summaries/*.json'):
            continue

        if not os.path.exists(file):  # i.e., file was deleted
            continue

        collections_to_test += get_collections_from_file(file)

    if collections_to_test:
        print("::set-output name=collections::" + '\n'.join(collections_to_test))
    else:
        print("::set-output name=collections::")


if __name__ == "__main__":
    main()
