# Config Summary

These config files specify the dataset inclusion/exclusion criteria when downloading collections using `src/download_and_filter.py`.
The config settings allow users to toggle for specific datasets/collections, license categories, terms, conditions, model-generated, languages, tasks, sources, a release time range, and normalized data format.

Note that these filters operate on the licenses and terms at the dataset-level. Much of the text in popular text datasets is crawled from web-sources, some of which may have their own terms or licenses. We leave the responsibility of verifying licenses to the dataset curators, or to the user: we do provide source labels in each data summary file.

Here is a summary of the provided configurations. They are in ascending order of restrictiveness:
* `default.yaml` includes all datasets. Available on [HuggingFace](https://huggingface.co/datasets/DataProvenanceInitiative/Everything).
* `commercial_or_unspecified_licenses.yaml` includes only datasets with commercial or unspecified licenses. Does not exclude datasets with non-commercial Terms of Use, e.g. from OpenAI.
* `commercial_or_unspecified_licenses_and_terms.yaml` includes only datasets with commercial or unspecified licenses, but excludes datasets that have non-commercial Terms of Use from OpenAI.
* `commercial_licenses.yaml` includes only datasets with commercial licenses. Does not exclude datasets with non-commercial Terms of Use from OpenAI.
* `commercial_licenses_and_terms.yaml` includes only datasets with commercial licenses, but excludes datasets that have non-commercial Terms of Use from OpenAI.
* `common_pile_ultra_permissive.yaml` includes only datasets from a select list that (a) contain English or code text, (b) are not model generated, (c) have a commercial license that is open source compliant or appears in the Gold, Silver or Bronze lists of the [Blue Oak Council](https://blueoakcouncil.org/list), and (d) where the original dataset sources are only from a select list that are also public domain or open source compliant.