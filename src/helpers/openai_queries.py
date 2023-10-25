"""
This demonstrates calling the OpenAI API with multiprocessing
to parallelize a batch of requests (and also with exponential-backoff retry logic so
that failed requests are retried)
"""
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from multiprocessing import Pool
import openai
from functools import partial
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from helpers.io import read_json, write_json
# from src.helpers.time import to_hms

# assert os.getenv("OPENAI_API_KEY")
openai.api_key = "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" #os.getenv("OPENAI_API_KEY")


def to_hms(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


@retry(
    wait=wait_random_exponential(min=1, max=60), 
    stop=stop_after_attempt(3), 
    retry_error_callback=lambda retry_state: None
)
def _run_gpt_chat_query(prompt, model, max_tokens, temp, logit_bias):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temp,
        logit_bias=logit_bias
    )
    response_text = response["choices"][0]["message"]["content"].strip()
    return response_text


@retry(
    wait=wait_random_exponential(min=1, max=60), 
    stop=stop_after_attempt(3), 
    retry_error_callback=lambda retry_state: None
)
def _run_gpt_completion_query(prompt, model, max_tokens, temp, logit_bias):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temp,
        max_tokens=max_tokens,
        logit_bias=logit_bias
    )
    return response["choices"][0]["text"].strip()


def run_text_completion(
    prompts,
    model="gpt-3.5-turbo",
    batch_size=10,
    max_tokens=100,
    temp=0,
    logit_bias={},
    save_dir="openai_cache/",
    multiprocess=False,
    verbose=True,
):
    start_time = time.time()
    assert model in [
        "text-davinci-001",
        "text-davinci-002",
        "text-davinci-003", 
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-4",
    ]
    
    cache = {}
    cache_fp = f"{save_dir}/cache/openai_{model}_{max_tokens}_{temp}.json.gz"
    # if os.path.exists(cache_fp):
        # cache = read_json(cache_fp)

    results = {}
    for i, prompt in enumerate(prompts):
        if prompt in cache:
            results[i] = cache[prompt]
    prompt_idx = [i for i, prompt in enumerate(prompts) if i not in results]
    remaining_prompts = [prompts[idx] for idx in prompt_idx]
    if verbose:
        print(f"Found {len(results)} / {len(prompts)} responses in the cache.")
    
    # now run pool
    responses = []
    if len(remaining_prompts) > 0:
        llm_fn =  _run_gpt_chat_query if "turbo" in model or "gpt-4" in model else _run_gpt_completion_query
        # with Pool(batch_size) as pool:
        #     pooled = pool.map(partial(llm_fn, model=model, max_tokens=max_tokens, temp=temp), remaining_prompts)
        #     for i, (prompt, response) in enumerate(zip(remaining_prompts, pooled)):
        #         results[prompt_idx[i]] = response
        if multiprocess:
            pooled = process_map(partial(llm_fn, model=model, max_tokens=max_tokens, temp=temp, logit_bias=logit_bias),
                remaining_prompts, max_workers=batch_size)
            for i, (prompt, response) in enumerate(zip(remaining_prompts, pooled)):
                results[prompt_idx[i]] = response

            failed_responses = sum([1 if r is None else 0 for r in results.values()])
        else:
            for i, remaining_prompt in tqdm(enumerate(remaining_prompts)):
                response = llm_fn(remaining_prompt, model=model, max_tokens=max_tokens, temp=temp, logit_bias=logit_bias)
                results[prompt_idx[i]] = response
            failed_responses = sum([1 if r is None else 0 for r in results.values()])
        print(f"Completed {len(prompts)} with {failed_responses} failures.")

        # update cache
        # TODO: Why even load the cache and save the cache if you're just going to override it.
        for prompt, response in zip(prompts, responses):
            cache[prompt] = response
        write_json(cache, cache_fp, compress=True)

    responses = [results[i] for i in range(len(results))]
    end_time = time.time()
    print(f"Time taken = {to_hms(end_time - start_time)}")

    return responses


# NOTE:
# You can find an updated, more robust and feature-rich implementation
# in Zeno Build
# - Zeno Build: https://github.com/zeno-ml/zeno-build/
# - Implementation: https://github.com/zeno-ml/zeno-build/blob/main/zeno_build/models/providers/openai_utils.py

# import openai
import asyncio
from typing import Any

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

async def dispatch_openai_requests(
    messages_list, #: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    # top_p: float,
):
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def run_asyncio_openai(messages, openai_key, temperature, max_tokens, batch_size=10):
    openai.api_key = openai_key
    all_messages = [[{"role": "user", "content": message}] for message in messages]
    all_responses = []
    for chunk in divide_chunks(all_messages, batch_size):
        responses = asyncio.run(
            dispatch_openai_requests(
                messages_list=chunk,
                # messages_list=[
                #     [{"role": "user", "content": "Write a poem about asynchronous execution."}],
                #     [{"role": "user", "content": "Write a poem about asynchronous pirates."}],
                # ],
                model="gpt-3.5-turbo",
                temperature=temperature,
                max_tokens=max_tokens,
                # top_p=1.0,
            )
        )
        all_responses.extend(responses)
        time.sleep(20)

    predictions = [x['choices'][0]['message']['content'] for x in all_responses]
    return predictions