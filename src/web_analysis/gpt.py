import os
import openai
import json
import asyncio
import aiohttp
import requests
import tiktoken
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

class GPT(object):
    """
        fill in later
    """
    def __init__(self, 
                 language='en',
                 model='gpt-3.5-turbo',
                 prompt='scraping-policy'):
    
        self.load_API_key()

        self.language_code = language
        self.model = model
        self.prompt_id = prompt
        self.SYSTEM_PROMPT = "You are a smart and intelligent legal assistant. I will provide you with the Terms of Use/Service document for a website and you will answer legal questions about that document."
        self.USER_PROMPT_1 = "Are you clear about your role?"
        self.ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your task. Please provide me with the necessary information to get started."
        self.GUIDELINES_PROMPT_TEMPLATE = self.load_prompt_from_json()


                                                ####### loading methods #######

    def load_API_key(self):
        try:
            openai.api_key = os.environ['OPENAI_API_KEY']
            print("OpenAI API key successfully loaded!")
        except KeyError:
            print("Error: OPENAI_API_KEY environment variable is not set.")
        except openai.error.AuthenticationError:
            print("Error: Incorrect OpenAI API key.")

    def load_prompt_from_json(self):
        """Loads a specific prompt from a JSON file based on a provided key, defaults to 'scraping-policy' prompt.
            Other keys supported are: "scraping-policy", "AI-policy", "competing-services", "illicit-content", "type-of-license"
        """
        try:
            with open('data/prompt_templates.json', 'r') as file:
                prompts = json.load(file)["prompts"]
                for prompt in prompts:
                    if prompt["id"] == self.prompt_id:
                        return prompt["content"]
        except FileNotFoundError:
            print("prompt_templates.json file not found.")
        except json.JSONDecodeError:
            print("Error decoding prompts.json.")

        # return default prompt (scraping-policy) if no specific prompt is found or in case of error
        print(f"Prompt with key '{self.prompt_id}' not found. Using default prompt: 'scraping-policy'.")
        self.prompt_id = 'scraping-policy'
        return prompts[0]['content']

                                                
                                                ####### make OpeanAI requests #######

    def make_openai_request(self, final_prompt):
        """
        Makes a request to the OpenAI Chat API to generate completions for a given prompt.

        Parameters:
        - final_prompt (str): The final prompt to be sent to the OpenAI Chat API.

        Returns:
        - str: The response from the OpenAI Chat API containing the completion for the given prompt.
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=0.2,           # lower temperature for more deterministic outputs
            top_p=0.1,                 # lower top_p to decrease randomness
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT_1},
                {"role": "assistant", "content": self.ASSISTANT_PROMPT_1},
                {"role": "user", "content": final_prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip(" \n")

    async def make_openai_request_async(self, session, final_prompt):
        """
        Makes an asynchronous request to the OpenAI Chat API to generate completions for a given prompt.

        Parameters:
        - session (aiohttp.ClientSession): A session object used to make asynchronous requests.
        - final_prompt (str): The final prompt to be sent to the OpenAI Chat API.

        Returns:
        - str: The response from the OpenAI Chat API containing the completion for the given prompt.
        """
        url = "https://api.openai.com/v1/chat/completions"  # endpoint for gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,           # lower temperature for more deterministic outputs
            "top_p": 0.1,                 # lower top_p to decrease randomness
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT_1},
                {"role": "assistant", "content": self.ASSISTANT_PROMPT_1},
                {"role": "user", "content": final_prompt}
            ]
        }
        async with session.post(url, json=payload, headers=headers) as response:
            # print(f"status code: {response.status}")
            # status code of 200 indicates a successful connection
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content'].strip(" \n")
            else:
                return None

                                                ####### batched processing #######

    async def process_batch_async(self, session, batch, custom_guidelines_prompt=None):
        """
        Processes a single batch of prompts asynchronously.

        Parameters:
        - session (aiohttp.ClientSession): A session object used to make asynchronous requests.
        - batch (list of str): List of prompts to be processed in the batch.

        Returns:
        - list of str: List of responses from the OpenAI Chat API, each containing the completion for the corresponding prompt in the batch.
        """
        if custom_guidelines_prompt != None:
            try:
                tasks = [self.make_openai_request_async(session, custom_guidelines_prompt.format(prompt)) for prompt in batch]
            except Exception as e:
                print(f"Failed to format custom guidlines. Make sure guidelines are formatted with empty brackets for where prompts will be inerted. {e}")
        else:
            tasks = [self.make_openai_request_async(session, self.GUIDELINES_PROMPT_TEMPLATE.format(prompt)) for prompt in batch]
        return await asyncio.gather(*tasks)

    async def process_prompts_in_batches_async(self, prompts, batch_size=10, custom_guidelines_prompt=None):
        """
        Processes a list of prompts in batches asynchronously by sending them to the OpenAI Chat API.

        Parameters:
        - prompts (list of str): List of prompts to be processed.
        - batch_size (int, optional): Size of each batch. Defaults to 10.

        Returns:
        - list of dict: List of responses from the OpenAI Chat API, each containing the completion for the corresponding prompt.
        """
        final_responses = []
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                if custom_guidelines_prompt != None:
                    batch_responses = await self.process_batch_async(session, batch, custom_guidelines_prompt=custom_guidelines_prompt)
                    final_responses.append(batch_responses)
                else:
                    batch_responses = await self.process_batch_async(session, batch)
                    formatted_responses = []
                    for j, response in enumerate(batch_responses):
                        if response is not None:
                            try:
                                formatted_response = json.loads(response)
                                formatted_responses.append(formatted_response) 
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse response: {response}. Error: {e}")
                        # else:
                        #     formatted_responses.append(None)
                    final_responses.extend(formatted_responses)
                # final_responses.extend(batch_responses)
        return final_responses

                                                ####### getters and setters #######


    def get_token_count(self, msg):
        """
        Calculate and print the number of tokens in a given message using the model-specific encoding.

        Parameters:
        - msg (str): The text message to encode and count tokens.

        Returns:
        - None: Outputs the token count directly to the console.
        """
        encoding = tiktoken.encoding_for_model(self.model)
        token_count = len(encoding.encode(msg))
        print(f"The text contains {token_count} tokens.")

    def get_guidelines_prompt(self):
        """
        Retrieve the current guidelines prompt template stored in the class.

        Returns:
        - str: The guidelines prompt template.
        """
        return self.GUIDELINES_PROMPT_TEMPLATE

    def get_system_prompt(self):
        """
        Retrieve the current system prompt stored in the class.

        Returns:
        - str: The system prompt.
        """
        return self.SYSTEM_PROMPT

    def set_guidelines_prompt(self, new_prompt=None, default=False):
        """
        Set the guidelines prompt template to a new value or reset to default.

        Parameters:
        - new_prompt (str, optional): The new prompt template to set, if not resetting to default.
        - default (bool): If True, reset the guidelines prompt template to the default value.

        Returns:
        - None: Updates the guidelines prompt template in place.
        """
        if default == True:
            self.GUIDELINES_PROMPT_TEMPLATE = self.load_prompt_from_json()
        else:
            # test to make sure new guidlines can be properly formatted
            temp = new_prompt.format("test")
            if temp == new_prompt:
                print("Failed to format new guidlines prompt. Please try again and make sure guidelines are formatted with empty brackets for where prompts will be inerted. Ex: \"These guidelines are formatted correctly with brackets for where new prompt will go: \{\}.\"")
            elif new_prompt != None:
                self.GUIDELINES_PROMPT_TEMPLATE = new_prompt

    def set_system_prompt(self, new_prompt=None, default=False):
        """
        Set the system prompt to a new value or reset to default.

        Parameters:
        - new_prompt (str, optional): The new prompt to set, if not resetting to default.
        - default (bool): If True, reset the system prompt to the default value.

        Returns:
        - None: Updates the system prompt in place.
        """
        if default == True:
            self.SYSTEM_PROMPT = "You are a smart and intelligent legal assistant. I will provide you with a terms of service document and you will answer legal questions about the given document."
        elif new_prompt != None:
            self.SYSTEM_PROMPT = new_prompt