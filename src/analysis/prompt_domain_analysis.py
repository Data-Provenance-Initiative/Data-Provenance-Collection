import openai
import os
import json
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

dataset_original = load_dataset('allenai/WildChat', split='train')

dataset = dataset_original.shuffle(seed=42)
dataset_sampled = dataset.select(range(1000))
prompts = [(entry['conversation'][0]['content'], entry['model'])
           for entry in dataset_sampled]


SYSTEM_PROMPT = """
You are a categorization assistant. I will provide you with a user prompt and a response. Your task is to classify the prompt into one the following Content Domain categories: 
Academic, Books, Biomedical/Health, Business/Finance, Education/Knowledge, E-Commerce, Entertainment, Exams, Government/Policy, Legal, News, Reviews, Social/Lifestyle, Technology/Code, Cultural/Artistic, Religion, General, Other.

Additionally, classify the prompt into one or more of the following Type of Service categories: 
Ecommerce, Periodicals, Social Media/Forums, Encyclopedia/Database, Academic, Government, Company website, Other.
Provide the classifications in a JSON format with keys 'Content Domain' and 'Type of Service'.
"""


def make_openai_request(final_prompt):
    """
    Makes a request to the OpenAI Chat API to generate completions for a given prompt.

    Parameters:
    - final_prompt (str): The final prompt to be sent to the OpenAI Chat API.

    Returns:
    - str: The response from the OpenAI Chat API containing the completion for the given prompt.
    """
    client = OpenAI(
        # this is also the default, it can be omitted
        api_key=os.environ['OPENAI_API_KEY'],
    )
    response = client.chat.completions.create(
        model='gpt-4o',
        temperature=0.2,           # lower temperature for more deterministic outputs
        top_p=0.1,                 # lower top_p to decrease randomness
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},

        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


def process_prompts(prompts):
    results = []
    for prompt, model in tqdm(prompts, desc="Processing Prompts"):
        formatted_prompt = f"{SYSTEM_PROMPT}\nUser Prompt: {prompt}"
        response = json.loads(str(make_openai_request(formatted_prompt)))
        result = {
            "WildChat Example Prompt": prompt,
            "WildChat Example Response": response,
            "WildChat Model": model,
            "Content Domain": response["Content Domain"],
            "Types of Service": response["Type of Service"]
        }
        results.append(result)
    return results


# Process the prompts
categorized_prompts = process_prompts(prompts)

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(categorized_prompts)
df.to_csv('wildchat_analysis_results.csv', index=False)
