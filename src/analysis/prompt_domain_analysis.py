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

dataset_original = load_dataset("allenai/WildChat", split="train")

dataset = dataset_original.shuffle(seed=42)
dataset_sampled = dataset.select(range(1000))
prompts = [
    (entry["conversation"][0]["content"], entry["model"]) for entry in dataset_sampled
]

SYSTEM_PROMPT_CONTENT_DOMAIN = """
You are a categorization assistant. I will provide you with a user prompt and a response. Your task is to classify the prompt into a list of the following Content Domain categories: 

Academic, Books, Biomedical/Health, Business/Finance, Education/Knowledge, E-Commerce, Entertainment, Exams, Government/Policy, Legal, News, Reviews, Social/Lifestyle, Technology/Code, Cultural/Artistic, Religion, General, Other.

Provide the classification in a JSON format with the key 'Content Domain' and the value being the list [] of content categories.
"""

SYSTEM_PROMPT_SERVICE = """
You are a categorization assistant. I will provide you with a user prompt and a response. Your task is to classify the prompt into one or more of the following 'Type of Service' categories.

Categories:
- General informational requests
- Creative composition
- Academic composition
- Coding composition
- Brainstorming, planning, or ideation
- Asking for an explanation, reasoning, or help solving a puzzle or math problem
- Translation
- Self-help, advice seeking, or self-harm
- Sexual or illegal content requests
- News or recent events informational requests
- E-commerce or information requests about products and purchasing
- Information requests specifically about organizations, companies, or persons
- Other (choose this only as a last resort)

Descriptions (do not include these in the labels):
- Creative composition: such as role-playing, fictional story writing, or continuation
- Academic composition: such as non-fiction essay writing, continuation, or editing
- Coding composition: fixing, debugging, or help
- Brainstorming, planning, or ideation
- Asking for an explanation, reasoning, or help solving a puzzle or math problem
- Self-help: advice seeking, or self-harm
- Sexual or illegal content requests: inappropriate or illicit content requests
- News or recent events informational requests
- E-commerce: information requests about products and purchasing
- Information requests: specifically about organizations, companies, or persons

Provide the classification in the following JSON format:
{
    "Type of Service": []
}
"""


SYSTEM_PROMPT_THIRD_CATEGORY = """
You are a categorization assistant. I will provide you with a user prompt and a response. Your task is to classify the prompt into one or more of the following Content Domain categories based on the EU AI Act:

Categories:
- Biometric Surveillance
- Critical Infrastructure
- Education
- Employment and Worker Management
- Access to Government Services
- Law Enforcement
- Migration and Asylum
- Administration of Justice
- Elections and Voting
- Other (choose this only as a last resort)

Key Categories (most likely to appear):
- Emotion Recognition (1c): Does the prompt ask for information about emotion recognition?
- Academic Dishonesty Detection (3d): Does the prompt ask for information about detecting academic dishonesty or the model to do something academically dishonest, like write a school/project essay?
- Truthfulness Assessment (6b): Does the prompt ask for an assessment of truthfulness?
- Legal Interpretation (8a): Does the prompt ask for an interpretation of the law?
- Voting Information (8b): Does the prompt ask for information about an election or voting behavior?

Descriptions (do not include these in the labels):
- Biometric Surveillance: includes biometric verification of a specific person, biometric categorization based on sensitive or protected attributes, and emotion recognition.
- Critical Infrastructure: includes management and operation of critical digital infrastructure, road traffic, and the supply of water, gas, heating, and electricity.
- Education: includes determining access or admission to an educational institution, evaluating learning outcomes, assessing the appropriate level of education an individual will receive, and monitoring/detecting cheating during tests.
- Employment and Worker Management: includes recruitment (placing targeted job advertisements, analyzing and filtering job applications, evaluating candidates), making employment decisions (promotion/termination, allocating tasks, monitoring and evaluating performance).
- Access to Government Services: includes access to public assistance (e.g., eligibility and allocation), evaluating creditworthiness, evaluating emergency calls/dispatching emergency services, and risk assessment in life/health insurance.
- Law Enforcement: includes assessing the risk of being a victim of a crime, polygraphs, assessing the reliability of legal evidence, assessing recidivism risk, and profiling specific groups in the course of an investigation.
- Migration and Asylum: includes polygraphs, assessing the risk of irregular migration, examining asylum applications, and identifying a person in the context of border control.
- Administration of Justice: includes researching or interpreting the law, and influencing the outcome of an election or voting behavior.
- Elections and Voting: includes analyzing legal texts, and assessing whether to vote for a candidate.

Provide the classification in the following JSON format:
{
    "EU Domain": []
}

"""


def make_openai_request(final_prompt, system_prompt):
    """
    Makes a request to the OpenAI Chat API to generate completions for a given prompt.

    Parameters:
    - final_prompt (str): The final prompt to be sent to the OpenAI Chat API.
    - system_prompt (str): The system prompt to be used for the request.

    Returns:
    - str: The response from the OpenAI Chat API containing the completion for the given prompt.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,  # lower temperature for more deterministic outputs
        top_p=0.1,  # lower top_p to decrease randomness
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def process_prompts(prompts):
    results = []
    for prompt, model in tqdm(prompts, desc="Processing Prompts"):
        # Process Content Domain
        formatted_prompt_content = f"User Prompt: {prompt}"
        response_content = json.loads(
            make_openai_request(formatted_prompt_content, SYSTEM_PROMPT_CONTENT_DOMAIN)
        )

        # Process Type of Service
        formatted_prompt_service = f"User Prompt: {prompt}"
        response_service = json.loads(
            make_openai_request(formatted_prompt_service, SYSTEM_PROMPT_SERVICE)
        )

        # Process Third Category (Placeholder for now)
        formatted_prompt_third = f"User Prompt: {prompt}"
        response_third = json.loads(
            make_openai_request(formatted_prompt_third, SYSTEM_PROMPT_THIRD_CATEGORY)
        )

        result = {
            "WildChat Example Prompt": prompt,
            "WildChat Model": model,
            "Content Domain": response_content["Content Domain"],
            "Types of Service": response_service["Type of Service"],
            "EU Domain": response_third["EU Domain"],  # Placeholder result
        }
        results.append(result)
    return results


# Process the prompts
categorized_prompts = process_prompts(prompts)

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(categorized_prompts)
df.to_csv("wildchat_analysis_results_revised_v4.csv", index=False)
