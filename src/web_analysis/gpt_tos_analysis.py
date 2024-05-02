from gpt import GPT
import asyncio
import json
import random
import re
import pickle
from datetime import datetime
from tqdm import tqdm
import csv
import argparse


def load_data(file_path):
    """
    Load JSON data from a file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Data loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def sample_random_data(data, sample_size=20, start_date=None, end_date=None, save_data=False, file_name='data/sampled_data.pkl'):
    """
    Sample random entries from the data between specified dates.

    Parameters:
    - data (dict): The data to sample from.
    - sample_size (int): Number of samples to retrieve.
    - start_date (str, optional): Starting date in 'mm-dd-yyyy' format.
    - end_date (str, optional): Ending date in 'mm-dd-yyyy' format.
    - save_data (bool): Whether to save the sampled data.
    - file_name (str): File path to save the data if save_data is True.

    Returns:
    - list: A list of flattened sampled data.
    """
    flat_list = []

    if start_date:
        start_date = datetime.strptime(start_date, '%m-%d-%Y')
    if end_date:
        end_date = datetime.strptime(end_date, '%m-%d-%Y')
    
    for url, links in data.items():
        for link, dates in links.items():
            for date, text in dates.items():
                date_obj = datetime.strptime(date, '%m-%d-%Y')
                if (not start_date or date_obj >= start_date) and (not end_date or date_obj <= end_date):
                    flat_list.append((url, link, date, text))
    
    if len(flat_list) < sample_size:
        raise ValueError("Not enough data to sample from within the specified time frame.")
    
    sampled_data = random.sample(flat_list, sample_size)

    if save_data == True:
        save_to_pickle(sampled_data,file_name)
        
    return sampled_data

def sample_data_per_url(data, sample_size=20, save_data=False, file_name='data/sampled_data.pkl'):
    """
    Sample data entries for distinct URLs.

    Parameters:
    - data (dict): The data dictionary containing URLs and associated data.
    - sample_size (int): Number of URLs to sample.
    - save_data (bool): Whether to save the sampled data.
    - file_name (str): File path to save the data if save_data is True. Defaults to 'data/sampled_data.pkl'

    Returns:
    - dict: A dictionary containing sampled data each URL with all of the most recent associated TOS links.
    """
    non_empty_urls = [url for url, links in data.items() if links]

    if len(non_empty_urls) < sample_size:
        raise ValueError("Not enough parent URLs with data to sample from.")

    sampled_urls = random.sample(non_empty_urls, sample_size)

    output_data = {}
    for url in sampled_urls:
        tos_entries = []
        for tos_link, dates in data[url].items():
            # find the most recent date for each TOS link
            most_recent_date = max(dates, key=lambda date: datetime.strptime(date, '%m-%d-%Y'))
            most_recent_text = dates[most_recent_date]
            tos_entries.append((tos_link, most_recent_date, most_recent_text))
        
        if tos_entries:
            output_data[url] = tos_entries
        
    if save_data == True:
        save_to_pickle(output_data,file_name)

    return output_data

def save_to_pickle(data, file_name):
    """
    Save data to a pickle file.

    Parameters:
    - data (any): The sample data to pickle.
    - file_name (str): Path to the file where data will be saved.

    Returns:
    - None: Prints the status of the save operation.
    """
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to {file_name}")
    except (FileNotFoundError, IOError) as e:
        print(f"Error accessing the file {file_name}: {e}")
    except (pickle.PicklingError, Exception) as e:
        print(f"Error during pickling or an unexpected error: {e}")

def clean_text(text):
    """
    Clean text by removing unwanted characters and normalizing whitespace.

    Parameters:
    - text (str): The text to clean.

    Returns:
    - str: The cleaned text.
    """
    cleaned_text = re.sub(r'[\t\n\r]+', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def pre_process_tos_text(data):
    """
    Pre-process Terms of Service (TOS) text by cleaning each entry.

    Parameters:
    - data (dict): A dictionary where keys are URLs and values are lists of tuples containing TOS data.

    Returns:
    - dict: A dictionary with URLs as keys and cleaned TOS texts as values.
    """
    tos_texts_by_url = {}
    for url, entries in data.items():
        # collect and clean all TOS texts for the current URL
        all_texts = [clean_text(text) for (_, _, text) in entries]
        tos_texts_by_url[url] = all_texts
    return tos_texts_by_url

def pre_process_tos_text_keywords(data_dict, prompt_key):
    """
    Processes a dictionary containing domain-to-Terms of Service (ToS) mappings to extract and preprocess ToS text
    relevant to specified keywords. The function combines all ToS text per domain, cleans it, and then finds relevant
    sections of text based on the provided prompt_key that matches specific keywords.

    Parameters:
        data_dict (dict): A dictionary mapping parent domains to a list of tuples, each containing a link, date, 
                          and ToS text.
        prompt_key (str): The key that corresponds to specific keywords used to filter relevant text.
    Returns:
        list of tuples: A list where each tuple contains a domain URL and the post-processed text that contains
                        keywords relevant to the prompt_key.
    """
    combined_tos_text = []
    for parent_domain, tos_links in data_dict.items():
        all_tos_text = ""
        for link, date, tos_text in tos_links:
            all_tos_text += ' ' + clean_text(tos_text) + ' '
        all_tos_text = all_tos_text.strip()
        combined_tos_text.append((parent_domain,all_tos_text))
        
    results = []
    for url, tos_text in combined_tos_text:
        relevant_text = find_relevant_text(tos_text, prompt_key)
        # if(relevant_text):
        results.append((url,post_process_tos_text_keywords(relevant_text)))

    return results

def post_process_tos_text_keywords(data_dict):
    """
    Aggregates values from a dictionary where each value is a list of strings, concatenating them into a single string.
    This function is typically used to concatenate text fragments that have been identified as relevant based on specific
    keywords.

    Parameters:
        data_dict (dict): A dictionary where each key is a text category and the value is a list of strings containing
                          relevant extracted text.
    Returns:
        str: A single string composed of all concatenated values from the input dictionary.
    """
    results = []
    result = ""
    for values in data_dict.values():
        result += ' '.join(values) + ' '
    result = result.strip()

    return result

def find_relevant_text(tos_text, prompt_key):
    """
    Searches for sentences in the given text that contain keywords related to a specified prompt key. The function
    uses regular expressions to identify sentences that contain any of the keywords and returns these along with 
    the preceding and following sentences for context.

    Parameters:
        tos_text (str): The text from a Terms of Service document.
        prompt_key (str): The key for which to find relevant keywords. This key should correspond to a predefined
                          set of keywords.
    Returns:
        dict: A dictionary where each key is a keyword found in the text and the value is a list of strings, each 
              string being a "relevant text" snippet that includes the keyword and its surrounding context.
    """
    keywords = {'scraping-policy-keywords': ["Scrape", "harvest", "extract", "Web scraping", "Web Crawler", "spiders", "scripts", "Archival", "Machine-readable data", "Metadata", "Crawling", "Indexing", "Sitemaps", "Robots.txt files", "Descriptive metadata", "Keyword research", "Timeliness of content", "Last modified", "HTTP headers", "Automated web scraping", "CAPTCHAs", "Denial-of-service attack", "data gathering", "extraction methods", "public search engine", "non-amalgamated", "API"],
                 'AI-policy-keywords': ["machine learning", "ML", "artificial intelligence", "AI", "system", "training", "software", "program", "data sets", "generative", "models", "GPTBot user agent", "API"],
                 'competing-services-keywords': ["Commercial Use", "display", "distribute", "license", "publish", "reproduce", "duplicate", "copy", "create derivative works from", "modify", "sell", "resell", "exploit", "transfer", "commercial", "buying", "exchanging", "selling", "promotion"],
                 'illicit-content-keywords': ["unlawful", "threatening", "abusive", "harassing", "defamatory", "libelous", "deceptive", "fraudulent", "invasive of another's privacy", "tortious", "obscene", "vulgar", "pornographic", "offensive", "profane", "contains", "depicts nudity", "contains", "depicts sexual activity", "inappropriate", "criminal"],
                 'type-of-license-keywords': ["property", "respective owners", "copyright", "trademark", "subsidiaries", "affiliates", "assigns", "licensors", "without limitation", "creative", "transmit", "transfer", "sale", "sell", "derivative works", "amalgamated"]
                }
    # reg. ex. patterns
    sentence_pattern = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.IGNORECASE)
    keyword_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords[prompt_key])) + r')\b', re.IGNORECASE)
    
    sentences = sentence_pattern.findall(tos_text)
    
    relevant_texts = {}

    for i, sentence in enumerate(sentences):
        match = keyword_pattern.search(sentence)
        if match:
            keyword_found = match.group(0).lower()
            combined_text = ''
            if i > 0:
                combined_text += sentences[i - 1] + " "  # previous sentence
            combined_text += sentence  # keyword sentence
            if i < len(sentences) - 1:
                combined_text += " " + sentences[i + 1]  # next sentence

            # clean text
            combined_text = combined_text.replace('\n', ' ')
            combined_text = re.sub(r'\s+', ' ', combined_text)
            combined_text = combined_text.strip()

            if keyword_found not in relevant_texts:
                relevant_texts[keyword_found] = []
                
            relevant_texts[keyword_found].append(combined_text)

    return relevant_texts

def run_gpt(prompts, gpt_4_turbo, liscence_type):
    """
    Process Terms of Service documents using a GPT model.

    Parameters:
    - prompts (dict): A dictionary containing URLs as keys and lists of TOS texts as values.
    - gpt_4_turbo (GPT): An instance of a GPT model.
    - liscence_type (bool): Indicates if the license type is being checked.

    Returns:
    - dict: A dictionary containing the URLs as keys and tuples of verdicts and evidence as values.
    """
    verdicts = {}
    if 'keywords' in gpt_4_turbo.get_prompt_key():
        for url, text in tqdm(prompts, desc="Processing TOS documents"):
            verdict = False
            evidence = None
            if(not text):
                verdicts[url] = (verdict, evidence)
            else: 
                results = asyncio.run(gpt_4_turbo.process_prompts_in_batches_async([text]))
                for r in results:
                    if r['verdict'] == 'True':
                        verdict = True
                        evidence = r['evidence']  
                verdicts[url] = (verdict, evidence)
    else:
        for url, texts in tqdm(prompts.items(), desc="Processing TOS documents"):
            # does this need to be in an async function?
            results = asyncio.run(gpt_4_turbo.process_prompts_in_batches_async(texts))
            if(liscence_type == True):
                verdict = None
                evidence = None
                for r in results:
                    if r['verdict'] != 'N/A':
                        verdict = r['verdict']
                        evidence = r['evidence']     
                verdicts[url] = (verdict, evidence)
            else:
                verdict = False
                evidence = None
                for r in results:
                    if r['verdict'] == 'True':
                        verdict = True
                        evidence = r['evidence']     
                verdicts[url] = (verdict, evidence)
    return verdicts

def save_verdicts_to_csv(sampled_data, verdicts, prompt_key):
    """
    Save the verdicts obtained from the GPT model to a CSV file.

    Parameters:
    - sampled_data (dict): A dictionary of sampled data.
    - verdicts (dict): A dictionary containing verdicts and evidence.
    - prompt_key (str): The key for the specific prompt used in GPT processing.

    Returns:
    - None: Writes data directly to a CSV file.
    """
    results_to_write = []
    for url, entries in sampled_data.items():
        # extract only the TOS links from each tuple in the sampled data
        tos_links = [entry[0] for entry in entries][:5] 
        # pad if fewer than 5 links
        while len(tos_links) < 5:  
            tos_links.append(None) 
        
        answer = verdicts[url][0]
        evidence = verdicts[url][1]

        # create a new tuple with the URL, all (exactly 5) TOS links, answer, and evidence
        new_tuple = (url, *tos_links, answer, evidence)
        results_to_write.append(new_tuple)
        write_csv(results_to_write, prompt_key)

def write_csv(data, prompt_key):
    """
    Write data to a CSV file named based on a specified prompt key.

    Parameters:
    - data (list of tuples): The data to write to the CSV.
    - prompt_key (str): The prompt key that determines the questions for the CSV.

    Returns:
    - None: Writes data directly to a CSV file.
    """
    if('keywords' in prompt_key): 
        file_name = 'data/'+ prompt_key +'-gpt-keywords.csv'
    else: 
        file_name = 'data/'+ prompt_key +'-gpt.csv' # change this file path accordingly 
    questions = {'scraping-policy': 'Does the website restrict scraping? [True/False]', 'scraping-policy-keywords': 'Does the website restrict scraping? [True/False]',
                 'AI-policy': 'Does website have specific restrictions related to AI training? [True/False]', 'AI-policy-keywords': 'Does website have specific restrictions related to AI training? [True/False]',
                 'competing-services': 'Does website restrict use of content for competing services? [True/False]', 'competing-services-keywords': 'Does website restrict use of content for competing services? [True/False]',
                 'illicit-content': 'Does website restrict posting illicit content? [True/False]', 'illicit-content-keywords': 'Does website restrict posting illicit content? [True/False]',
                 'type-of-license': 'What may website content be used for?', 'type-of-license-keywords': 'What may website content be used for?'
                }
    question_text = questions[prompt_key]
    if (prompt_key != 'type-of-license'):
        reasoning_text = 'If True, provide evidence here:'
    else:
        reasoning_text = 'If license type found, provide evidence here:'
    headers = ['Domain', 'TOS link 1', 'TOS link 2', 'TOS link 3', 'TOS link 4', 'TOS link 5', question_text, reasoning_text]

    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for domain, tos1, tos2, tos3, tos4, tos5, verdict, evidence in data:
            try:
                writer.writerow([domain, tos1, tos2, tos3, tos4, tos5, verdict, evidence])
            except UnicodeEncodeError as e:
                print("Error writing data:", entry)
                raise e
            except Exception as e:
                print(e)

def open_sampled_data(sample_file_path):
    """
    Opens and loads data from a pickled file specified by the file path. This function is designed to handle 
    the file operations for reading binary data, specifically using the pickle module to deserialize objects 
    stored in files.

    Parameters:
        sample_file_path (str): The path to the file containing the pickled data.

    Returns:
        object: The data unpickled from the file if successful.
    """
    try:
        with open(sample_file_path, 'rb') as f:
            sampled_data = pickle.load(f)
            print("Sample data loaded successfully.")
            return sampled_data
    except FileNotFoundError:
        print(f"Error: The file {sample_file_path} does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

##################################################################################


def main(custom_sample, sample_file_path, sample_size, save_sample, prompt_key, save_verdicts):
    """
    Main function to orchestrate the data sampling and GPT processing based on command-line arguments.

    Parameters:
    args (Namespace): Command line arguments parsed by argparse.

    Returns:
    - dict or None: Returns the verdicts if save_verdicts is False. Otherwise, it saves the verdicts to a CSV and returns None.

    Raises:
    - ValueError: If no prompt_key is provided or if the sample_file_path is not provided when custom_sample is True.
    """

    # currently only supports sampled data - full dataset (allow for custom dataset/default) and keyword analysis coming soon
    # ADD NOTE ABOUT NEEDING .env FILE W OPEN_AI KEY

    if(prompt_key == None):
        raise ValueError("prompt_key must be provided. Options are: Options are: \"scraping-policy\", \"AI-policy\", \"competing-services\", \"illicit-content\", \"type-of-license\", \"scraping-policy-keywords\", \"AI-policy-keywords\", \"competing-services-keywords\", \"illicit-content-keywords\", \"type-of-license-keywords\"")
        return None
    else:
        # initialize gpt instance
        gpt_4_turbo = GPT(model='gpt-4-turbo', prompt=prompt_key)
        print(f"Using prompt: {gpt_4_turbo.get_guidelines_prompt()}")
        license_type = False
        if('type-of-license' in prompt_key): license_type = True

    if(custom_sample == True):
        if(not sample_file_path):
            raise ValueError("sample_file_path must be provided if custom_sample is set to True.")
        sampled_data = open_sampled_data(sample_file_path)
    else:
        # update the file path depending on where data is stored
        data = load_data('data/temporal_tos_data_1_reorg.json')
        sampled_data = sample_data_per_url(data, sample_size=sample_size, save_data=save_sample)

    if('keywords' in prompt_key):
        prompts = pre_process_tos_text_keywords(sampled_data, prompt_key)
        verdicts = run_gpt(prompts, gpt_4_turbo, license_type)
    else:
        prompts = pre_process_tos_text(sampled_data)
        verdicts = run_gpt(prompts, gpt_4_turbo, license_type)

    if(save_verdicts == True):
        save_verdicts_to_csv(sampled_data, verdicts, prompt_key)
    else:
        return verdicts
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--custom_sample', type=bool, default=False, help='Use a custom sample file. Must be a .pkl file in correct format.')
    parser.add_argument('--sample_file_path', type=str, default='', help='Path to the sample file')
    parser.add_argument('--sample_size', type=int, default=50, help='Size of the sample to generate')
    parser.add_argument('--save_sample', type=bool, default=False, help='Save the sampled data to .pkl file')
    parser.add_argument('--prompt_key', type=str, default=None, help='Prompt key for GPT model. Options are: \"scraping-policy\", \"AI-policy\", \"competing-services\", \"illicit-content\", \"type-of-license\", \"scraping-policy-keywords\", \"AI-policy-keywords\", \"competing-services-keywords\", \"illicit-content-keywords\", \"type-of-license-keywords\"')
    parser.add_argument('--save_verdicts', type=bool, default=True, help='Save the generated verdicts to .csv file')

    
    args = parser.parse_args()
    
    main(args.custom_sample, args.sample_file_path, args.sample_size, args.save_sample, args.prompt_key, args.save_verdicts)