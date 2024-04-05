# import os
# import pandas as pd
import numpy as np
# from functools import partial
# from collections import Counter, defaultdict
# from helpers import io
import re


# ##########################################################################
# ############## Data Preparer Utils
# ##########################################################################

def convert_inputs_targets_to_messages(
    input_text,
    target_text,
    dset,
):
    """
    Converts standard [input/output] type rows into the universal messages format.
    """
    return [
        {"from": "user", "text": input_text.strip(), "parent": dset},
        {"from": "assistant", "text": target_text.strip(), "parent": 0},
    ]


# ##########################################################################
# ############## Data Preparer Functions
# ##########################################################################

def prepare_flan_collection(row):
    return convert_inputs_targets_to_messages(
        row["inputs"], row["targets"], row["task_name"]
    )


def prepare_xp3x(row):
    # "xp3x-multi_eurlex-ron_latn"
    # task_name = "xp3x-" + row["dataset"].split("/")[-1] + "-" + row["language"].replace("-", "_")
    if row["dataset"] in ["clue"]:
        task_name = row["config"]  # Set task_name to e.g. `c3` without the clue
    else:
        task_name = row["dataset"].split("/")[-1]
    task_name = row["language"] + "/" + task_name
    return convert_inputs_targets_to_messages(
        row["inputs"], row["targets"], task_name)


def prepare_commitpackft(row):
    lang_normalized = row["lang"].replace("'", "").replace("(", "").replace(")", "").replace(" ", "-").lower()
    return convert_inputs_targets_to_messages(
        # Could add some strong delimiters to separate the code from the text
        # e.g. ```prog_lang\n<old_contents>\n```\n\n<subject>
        row["old_contents"] + "\n\n" + row["subject"],
        row["new_contents"],
        lang_normalized,
    )


def prepare_cobra_frames(row):
    """
    CobraFrames dataset has a structure where each row is one of the elements in the frame for harmful statement.
    fomatting foces on the structure of the input and output given the row.
    The first 4 elements are context, speaker, listener, and statement check, serving as the context for the statement.
    The rest of the elements are the structured explanation for the statement
    """

    formatting = {
    "speechContext": "[Context of statement] {}[/]",
    "speakerIdentity": "[Speaker identity/characteristics] {}[/]",
    "listenerIdentity": "[Listener identity/characteristics] {}[/]",
    "statementCheck": "[The statement is complete and understandable] {}[/]",
    "relevantPowerDynamics": "[Relevant power dynamics] {}[/]",
    "conversationContext": "[Conversational context] {}[/]",
    "statement": "[Statement] {}[/]",
    "intent": "[Intent] {}[/]",
    "offensiveness": "[Offensiveness] {}[/]",
    "targetGroup": "[Targeted/referenced minority group] {}[/]",
    "implication": "[Implied meaning/stereotype] {}[/]",
    "targetGroupEmotionalReaction": "[Targeted minority group emotional reaction] {}[/]",
    "targetGroupCognitiveReaction": "[Targeted minority group cognitive reaction] {}[/]",
    }

    f = [   
            formatting[v].format(row[v])
            for v in row.keys() if v in formatting
        ]
    input_instructions = "Following the examples and complete the structured explanation for the given statement.\n\n" + row['examples'] 
    input_context = "\n".join(f[1:5])
    output = "\n".join(f[5:])
    return convert_inputs_targets_to_messages(
        input_instructions + "\n" + input_context,
        output,
        row["_source"]
    )


def prepare_dolly_15k(row):
    input_text = re.sub(r'\s*\[.*?\]\s*', '', "\n".join([row["context"], row["instruction"]]).strip())
    target_text = re.sub(r'\s*\[.*?\]\s*', '', row["response"])
    return convert_inputs_targets_to_messages(
        input_text, target_text, row["category"]
    )


def prepare_thai_gen_ai_dolly(row):
    input_text = "\n".join([row["context"], row["instruction"]]).strip() if row["context"] else row["instruction"]
    target_text = row["response"]
    return convert_inputs_targets_to_messages(
        input_text, target_text, row["category"]
    )


def prepare_laion_oig(row):
    # Rosey is there since unified_joke_explanations uses this instead of <bot> marker.
    turn_markers = ["<human>:", "<bot>:", "Rosey:"]
    turns = row["text"].strip()
    parent = row["_source"]

    # Take care of situation when a Background is provided.
    if turns.startswith("Background:"):
        # Remove the first human tag and make the background a part of the human input.
        turns = turns.replace("<human>: ", "\n", 1)
        turns = "<human>: " + turns

    # Append the turn markers with a separator for easy splitting on the turns.
    SEPARATOR = "<*>"
    for tm in turn_markers:
        turns = turns.replace(tm, f"{SEPARATOR}{tm}")
    turns = turns.split(SEPARATOR)

    messages = []
    for i, turn in enumerate(turns):
        if turn.strip():
            speaker = "user" if turn.startswith("<human>") else "assistant"
            # Remove the turn markers from the turns
            for tm in turn_markers:
                turn = turn.replace(tm, "")
            messages.append({
                "from": speaker,
                "text": turn.strip(),
                "parent": parent,
            })
            parent = i
    return messages


def prepare_self_instuct(row):
    return convert_inputs_targets_to_messages(
        row["prompt"], row["completion"], "self_instruct",
    )


def prepare_anthropic_hh_rlhf(row):
    SEPARATOR = "<*>"
    chosen_text = row['chosen'].replace(SEPARATOR, " ")
    rejected_text = row['rejected'].replace(SEPARATOR, " ")
    # [(text, user, score)]

    # Add placeholder markers for splitting.
    marked_chosen = chosen_text.replace('\n\nHuman:', f'{SEPARATOR}USER{SEPARATOR}').replace('\n\nAssistant:', f'{SEPARATOR}ASSISTANT{SEPARATOR}')
    marked_rejected = rejected_text.replace('\n\nHuman:', f'{SEPARATOR}USER{SEPARATOR}').replace('\n\nAssistant:', f'{SEPARATOR}ASSISTANT{SEPARATOR}')

    # Split the transcript into statements using the placeholder markers.
    chosen_seq = marked_chosen.split(SEPARATOR)[1:]
    reject_seq = marked_rejected.split(SEPARATOR)[1:]

    messages = []
    parent = "anthropic_hhrlhf"
    for chosen_turn, reject_turn in zip(chosen_seq, reject_seq):
        if chosen_turn == reject_turn and chosen_turn in ["USER", "ASSISTANT"]:
            # sometimes there is only 1 response, not 2.
            turn_type = chosen_turn
        elif chosen_turn == reject_turn:
            if len(messages) > 0:
                parent = 0 if parent == "anthropic_hhrlhf" else parent + 1
            messages.append({
                "from": turn_type.lower(),
                "text": chosen_turn.strip(),
                "parent": parent,
            })
            if turn_type.lower() == "assistant":
                messages[-1]["score"] = 1.0
        else:
            parent = 0 if parent == "anthropic_hhrlhf" else parent + 1
            messages.append({
                "from": turn_type.lower(),
                "text": chosen_turn.strip(),
                "parent": parent,
                "score": 1.0
            })
            messages.append({
                "from": turn_type.lower(),
                "text": reject_turn.strip(),
                "parent": parent,
                "score": 0.0
            })

    return messages


def prepare_stanford_human_preferences(row):
    return [
        {"from": "user", "text": row["history"].strip(), "parent": row["domain"]},
        {"from": "assistant", "text": row["human_ref_A"].strip(), "score": row["score_A"], "parent": 0},
        {"from": "assistant", "text": row["human_ref_B"].strip(), "score": row["score_B"], "parent": 0},
    ]


def prepare_open_assistant(dset):
    messages = []
    current_message_tree = None  # dset[0]["message_tree_id"]
    messageid_to_idx, current_dialog = {}, []
    dialog_idx = 0
    for row in dset:
        if current_message_tree != row["message_tree_id"]:
            if current_dialog and len(current_dialog) > 1:
                messages.append(current_dialog)
            current_message_tree = row["message_tree_id"]
            current_dialog = [{
                "from": "user" if row["role"] == "prompter" else "assistant",
                "text": row["text"].strip().replace("\"", ""),
                "parent": row['lang'],
            }]
            dialog_idx = 0
            messageid_to_idx = {}
        else:
            if row["parent_id"] in messageid_to_idx:
                current_dialog.append({
                    "from": "user" if row["role"] == "prompter" else "assistant",
                    "text": row["text"].strip().replace("\"", ""),
                    "parent": messageid_to_idx[row["parent_id"]],
                })
                dialog_idx += 1
        messageid_to_idx[row["message_id"]] = dialog_idx
    return messages


def prepare_oasst_octopack(row):
    messages = []
    for i, segment in enumerate(row["conversations"]):
        messages.append({
            "from": "user" if segment["role"] == "prompter" else "assistant",
            "text": segment["text"].strip().replace("\"", ""),
            "parent": i-1 if i else "octopack",
        })
    return messages


def prepare_longform(row):
    dset_id = row["source"]  # .replace(" ", "").replace("-", "").lower()
    return convert_inputs_targets_to_messages(
        row["input"], row["output"], dset_id,
    )


def prepare_gpteacher(row):
    inp = row["instruction"]
    if row["input"]:
        inp += "\n" + row["input"]
    return convert_inputs_targets_to_messages(
        inp, row["response"], row["_source"],
    )


def prepare_openai_summarization(row):
    instruction = "Summarize the above article:"
    text0 = row["summaries"][0]["text"].strip()
    text1 = row["summaries"][1]["text"].strip()
    return [
        {"from": "user", "text": row["info"]["post"].strip() + "\n\n\n" + instruction, "parent": "openai-summarize"},
        {"from": "assistant", "text": text0, "score": int(np.abs(row["choice"] - 1)), "parent": 0},
        {"from": "assistant", "text": text1, "score": row["choice"], "parent": 0},
    ]


def prepare_openai_webgpt(row):
    context0 = row["quotes_0"]["extract"][0].strip() + "\n\n\n" if row["quotes_0"]["extract"] else ""
    context1 = row["quotes_1"]["extract"][0].strip() + "\n\n\n" if row["quotes_1"]["extract"] else ""
    text0 = context0 + row["question"]["full_text"].strip()
    text1 = context1 + row["question"]["full_text"].strip()
    return [
        {"from": "user", "text": text0, "parent": row['dataset']},
        {"from": "user", "text": text1, "parent": row['dataset']},
        {"from": "assistant", "text": row["answer_0"].strip(), "score": row["score_0"], "parent": 0},
        {"from": "assistant", "text": row["answer_1"].strip(), "score": row["score_1"], "parent": 1},
    ]


def prepare_alpaca(row):
    inputs = " ".join([row["instruction"], row["input"]]).strip()
    return convert_inputs_targets_to_messages(
        inputs, row["output"], "alpaca",
    )


def prepare_everything_lm(row):
    inputs = " ".join([row["instruction"], row["input"]]).strip()
    return convert_inputs_targets_to_messages(
        inputs, row["output"], "everything_lm",
    )


def prepare_llama2_med_tuned_instructions(row):
    inputs = "\n".join([row["instruction"], row["input"]]).strip()
    return convert_inputs_targets_to_messages(
        inputs, row["output"], "llama2_med_tuned_instructions",
    )

def prepare_capybara(row):
    messages = []
    parent_id = 0
    dset = row["source"]
    for i, turn in enumerate(row["conversation"]):
        messages.append({
            "from": "user",
            "text": turn["input"].strip(),
            "parent": dset if i == 0 else parent_id,
        })
        if parent_id != 0:
            parent_id += 1
        messages.append({
            "from": "assistant",
            "text": turn["output"].strip(),
            "parent": parent_id,
        })
        parent_id += 1
    return messages

def prepare_evol_instruct(row):
    return convert_inputs_targets_to_messages(
        row['instruction'], row["output"], "evol_instruct",
    )


def prepare_deita_10k(row):
    messages = []
    for i, turn in enumerate(row["conversations"]):
        messages.append({
            "from": "user" if turn["from"] == "human" else "assistant",
            "text": turn["value"].strip(),
            "parent": row["source"] if i == 0 else i-1
        })
    return messages


def prepare_metamathqa(row):
    return convert_inputs_targets_to_messages(
        row["query"], row["response"], row["type"],
    )


def prepare_pure_dove(row):
    messages = []
    parent_id = 0
    for i, turn in enumerate(row["conversation"]):
        messages.append({
            "from": "user",
            "text": turn["input"].strip(),
            "parent": "pure_dove" if i == 0 else parent_id,
        })
        if parent_id != 0:
            parent_id += 1
        messages.append({
            "from": "assistant",
            "text": turn["output"].strip(),
            "parent": parent_id,
        })
        parent_id += 1
    return messages

def prepare_nectar(row):
    human = []
    assistant = []
    messages = []

    if row["turns"] == 1:
        input = row["prompt"].split("Assistant:")[0].strip()
        output = row["answers"][0]["answer"]
        return convert_inputs_targets_to_messages(
            input, output, "nectar",
        )
    else:
        # Split the conversation based on "Human:" and "Assistant:" tags
        segments = row["prompt"].split("Human: ")[1:]
        # Extract human and assistant texts
        for segment in segments:
            parts = segment.split("Assistant:")
            human.append(parts[0].strip())
            if len(parts) > 1:
                assistant.append(parts[1].strip())
        assistant.append(row["answers"][0]["answer"])
        parent_id = 0
        for index, (h, a) in enumerate(zip(human, assistant)):
            messages.append({
                "from": "user",
                "text": h.strip(),
                "parent": "nectar" if index == 0 else parent_id,
            })
            if parent_id != 0:
                parent_id += 1
            messages.append({
                "from": "assistant",
                "text": a.strip(),
                "parent": parent_id,
            })
            parent_id += 1
        return messages

def prepare_feedback_collection(row):
    return convert_inputs_targets_to_messages(
        row['instruction'], row["output"], "feedback_collection",
    )


def prepare_sharegpt_vicuna(row):
    parent = "sharegpt_vicuna"
    messages = []
    for i, turn in enumerate(row["conversations"]):
        messages.append({
            "from": "user" if turn["from"] == "human" else "assistant",
            "text": turn["value"].strip(),
            "parent": parent,
        })
        parent = i
    return messages


def prepare_code_alpaca(row):
    inputs = row["instruction"].strip()
    if row["input"]:
        inputs += "\n" + row["input"].strip()
    return convert_inputs_targets_to_messages(
        inputs, row["output"], "code_alpaca",
    )


def prepare_hc3(row, lang):
    # dset_id = f"hc3_{lang}-{row['source']}"
    messages = [{"from": "user", "text": row["question"].strip(), "parent": row['source']}]
    if len(row["human_answers"]) and row["human_answers"][0]:
        human_answer = row["human_answers"][0].strip()
        messages.append({"from": "assistant", "text": human_answer, "score": 1, "parent": 0})
    if len(row["chatgpt_answers"]) and row["chatgpt_answers"][0]:
        assistant_answer = row["chatgpt_answers"][0].strip()
        messages.append({"from": "assistant", "text": assistant_answer, "score": 0, "parent": 0})
    return messages


def prepare_hc3_en(row):
    return prepare_hc3(row, "en")


def prepare_hc3_zh(row):
    return prepare_hc3(row, "zh")


def prepare_camel_science(row):
    return convert_inputs_targets_to_messages(
        row["message_1"], row["message_2"], row["_source"],
    )


def prepare_cot_collection(row):
    return convert_inputs_targets_to_messages(
        row["source"], row["rationale"], row['_source']
    )


def prepare_gpt4all(row):
    source_to_dsetid = {
        "": "stackoverflow",
        "pacovaldez/stackoverflow-questions": "stackoverflow",
        "nomic-ai": "nomic",
        "laion/unified_chip2": "chip2",
        "unified_chip2": "chip2",
        "unified_unifiedskg_instructions": "unifiedskg",
        "output_unified_unifiedskg.jsonl": "unifiedskg",
        "unified_multi_sum": "unifiedmultisum",
        "unified_abstract_infill_output_0-100_000.jsonl": "abstractinfill",
        "unified_abstract_infill_output-100-000-x.jsonl": "abstractinfill",
        "unified_hc3_human": "hc3"
    }
    return convert_inputs_targets_to_messages(
        # row["prompt"], row["response"], f"nomicai-gpt4allj--{source_to_dsetid[row['source']]}"
        row["prompt"], row["response"], row['source']
    )


def prepare_evol_instruct_v2(row):
    return convert_inputs_targets_to_messages(
        row['conversations'][0]["value"], row['conversations'][1]["value"], "evol_instruct_v2",
    )


def prepare_gpt4_alpaca(row):
    inputs = row["instruction"].strip()
    if row["input"]:
        inputs += "\n" + row["input"].strip()
    return convert_inputs_targets_to_messages(
        inputs, row["output"], "gpt4alpaca",
    )


def prepare_thai_gen_ai_alpaca(row):
    inputs = row["instruction"].strip()
    if row["input"]:
        inputs += "\n" + row["input"].strip()
    return convert_inputs_targets_to_messages(
        inputs, row["output"], "thai_gen_ai_alpaca",
    )


def prepare_tasksource_instruct(row):
    # task_name = "tsi-" + row['task'].replace("-", "_").replace("/", "-")
    return convert_inputs_targets_to_messages(
        row["inputs"], row["targets"], row['task'],
    )


def prepare_stack_exchange_instruction(row):
    return convert_inputs_targets_to_messages(
        row["question"], row["response"], "stack-exchange-instruction",
    )


def prepare_unnatural_instructions(row):
    return convert_inputs_targets_to_messages(
        row['instances']['instruction_with_input'][0],
        row['instances']['output'][0],
        "unnatural_instructions",
    )


def prepare_starcoder_self_instruct(row):
    return convert_inputs_targets_to_messages(
        row['instruction'], row['output'],
        'starcoder-self-instruct'
    )

  
def prepare_thai_gen_ai_gpteacher(row):
    inputs = row["instruction"].strip()
    if row["input"]:
        inputs += "\n" + row["input"].strip()
    return convert_inputs_targets_to_messages(
        inputs, row["output"], "thai_gen_ai_gpteacher",
    )


def tinystories_get_example(it):
    buf = []
    try:
        line = next(it)
        while line:
            if line['text'].strip() == '':
                break
            else:
                buf.append(line['text'])
            line = next(it)
    except StopIteration:
        if len(buf) > 0:
            pass  # we have an example to process
        else:
            raise  # we don't

    buf = '\n'.join(buf)

    try:
        inpt_text, *tgt_text = re.split('\nStory: ', buf, re.MULTILINE)

        inpt_text = inpt_text + '\nStory: '
        tgt_text = '\n'.join(tgt_text)
    except Exception:
        print('\n'.join(re.split('^Story: ', buf)))
        raise

    return convert_inputs_targets_to_messages(inpt_text, tgt_text, 'tiny-stories')


def prepare_tiny_stories(dset):
    stories = []
    it = iter(dset)

    while True:
        try:
            stories += [tinystories_get_example(it)]
        except StopIteration:
            break

    return stories


def prepare_joke_explanation(row):
    inputs = row["joke"] + "\n\n" + "Explain this joke."
    return convert_inputs_targets_to_messages(inputs, row["explaination"], "joke-explanation")


def prepare_book_summaries(row):
    instruction = "Summarize the above text:"
    return convert_inputs_targets_to_messages(
        row["input"].strip() + "\n\n\n" + instruction, row["output"].strip(), "summary"
    )


def prepare_ultrachat(row):
    parent = "ultrachat"
    messages = []
    for i, script in enumerate(row["data"]):
        messages.append({
            "from": "user" if i % 2 == 0 else "assistant",
            "text": script.strip(),
            "parent": parent,
        })
        parent = i
    return messages


def prepare_wildchat(row):
    messages = []
    for i, script_dict in enumerate(row['conversation']):
        messages.append({
            'from': script_dict['role'],
            'text': script_dict['content'].strip(),
            'parent': row['model'] if i==0 else i-1
        })
    return messages

def prepare_airoboros(row):
    parent = "airoboros"
    messages = []
    for i, turn in enumerate(row['conversations']):
        messages.append({
            "from": "user" if turn["from"] == "human" else "assistant",
            "text": turn["value"].strip(),
            "parent": parent,
        })
        parent = i
    return messages

def prepare_lima(row):
    messages = []
    parent = row['source']
    for i, turn in enumerate(row['conversations']):
        messages.append({
            "from": "assistant" if i % 2 else "user",
            "text": turn.strip(),
            "parent": parent
        })
        parent = i
    return messages
   
def prepare_tool_llama(row):
    return convert_inputs_targets_to_messages(
        row['context'] + row['instruction'],
        row['response'],
        'toolbench',
    )

def prepare_mathinstruct(row):
    return convert_inputs_targets_to_messages(
        row["instruction"], row["output"], row["_source"]
    )

def prepare_gorilla(row):
    return convert_inputs_targets_to_messages(
        row['instruction'],
        row['response'],
        'gorilla-apibench',
    )

def prepare_baize_data(row):
    messages = []
    items = row["input"].split("[|Human|]")
    parent_id = -1
    for item in items:
        if item.strip() == "The conversation between human and AI assistant.":
            continue
        elif item.strip() == "":
            break
        sub_items = item.strip().split("[|AI|]")
        human_turn_item = {
            "from": "user",
            "text": sub_items[0].strip(),
            "parent": row["_source"] if parent_id == -1 else parent_id,
        }
        messages.append(human_turn_item)
        parent_id += 1
        agent_turn_item = {
            "from": "assistant",
            "text": sub_items[1].strip(),
            "parent": parent_id,
        }
        messages.append(agent_turn_item)
        parent_id += 1

    return messages


def prepare_open_orca(row):
    inputs = "".join([row['system_prompt'] + row['question']])
    outputs = row['response']
    return [
        {"from": "user", "text": inputs.strip(), "parent": row['source']},
        {"from": "assistant", "text": outputs.strip(), "parent": 0},
    ]


def prepare_selfee(row):
    outputs = row["outputs"]
    parsed_outputs = ''
    for index, elem in enumerate(outputs):
        feedback = elem["feedback"]
        output = elem["output"]
        feedback_number = index + 1 
        revision_number = index
        if index != 0:
            output = "\n\n### Revision {number}:\n{revision}".format(number=revision_number, revision=output) 
        parsed_outputs += output + "\n\n### Feedback {number}:\n{feedback}".format(number=feedback_number, feedback=feedback)
    return convert_inputs_targets_to_messages(
        row['instruction'], parsed_outputs, "selfee",
    )

def prepare_pmc_llama(row):
    inputs = "".join([row['instruction'] + row['input']])
    return convert_inputs_targets_to_messages(
        inputs,
        row['output'],
        row['source']
    )
  
def prepare_medical_meadow(row):
    inputs = "".join([row['instruction'] + row['input']])
    return convert_inputs_targets_to_messages(
        inputs,
        row['output'],
        row["_source"],
    )

def prepare_medinstruct(row):
    inputs = "".join([row['instruction'] + row['input']])
    return convert_inputs_targets_to_messages(
        inputs,
        row['output'],
        'medinstruct',
    )
      
def prepare_chatdoctor(row):
    return convert_inputs_targets_to_messages(
        row["inputs"], row["outputs"], row["_source"]
    )

def prepare_seabench(row):
    inputs = row["turns"][0].strip()
    outputs = row["chatgpt_response"].strip() if row["chatgpt_response"] else ""

    return [
        {"from": "user", "text": inputs, "parent": row["lang"]},
        {"from": "assistant", "text": outputs, "parent": 0},
    ]

  
def prepare_agentinstruct(row):
    datasets = row  # Based on the current structure, a row represents all datasets :TODO: might need to change this
    messages = []
    for dataset in datasets:
        for i, turn in enumerate(dataset["conversations"], start=-1):
            messages.append({
                "from": "user" if turn["from"] == "human" else "assistant",
                "text": turn["value"].strip(),
                "parent": dataset['id'].split('_')[0] if i == -1 else i,
            })
    return messages


def prepare_cidar(row):
    return convert_inputs_targets_to_messages(
        row['instruction'],
        row['output'],
        'cidar',
    )

def prepare_indic_instruct(row):
    ''' This dataset conatins lots of other datasets, each having their own format. A different prepare method is needed for each sub-dataset 
    Some datasets such as 'hh-rlhf', 'lm_sys', 'oasst1' have same forrmat and thus they have the prepare method below
    '''
    if row['dataset'] == 'anudesh' : 
        return convert_inputs_targets_to_messages(
            row['messages'][0]['content'], row['messages'][1]['content'], row['dataset']
        )

    if row['dataset'] == 'dolly' : 
        input_text = re.sub(r'\s*\[.*?\]\s*', '', "\n".join([row["context"], row["instruction"]]).strip())
        target_text = re.sub(r'\s*\[.*?\]\s*', '', row["response"])
        return convert_inputs_targets_to_messages(
            input_text, target_text, row["dataset"]
        )

    if row['dataset'] == 'flan_v2' : 
        return convert_inputs_targets_to_messages(
            row["inputs"], row["targets"], row['dataset']
        )

    if row['dataset'] in ['hh-rlhf', 'lm_sys', 'oasst1'] : 
        messages = []
        for i, turn in enumerate(row['messages']):
            messages.append({
                "from": turn["role"],
                "text": turn["content"].strip(),
                "parent": row['dataset'] if turn["role"]=='user' else 0,
            })
        return messages

    if row['dataset'] == 'nmt-seed' : 
        return convert_inputs_targets_to_messages(
            row["input_text"], row["output_text"], row['dataset']
        )

    if row['dataset'] == 'wikihow' : 
        input_text = row["intro"]
        for i, turn in enumerate(row["steps"]) : 
            input_text += '\n' + turn['description']
        input_text += row['messages'][0]['content']

        target_text = row['messages'][1]['content']
        return convert_inputs_targets_to_messages(
            input_text, target_text, row["dataset"]
        )


def prepare_pii_masking_200k(row):
    inputs = row["unmasked_text"] + "\n\n" + "Given the previous paragraph, please mask any personally " \
                                             "identifiable information using masks, such as [FIRSTNAME_1], [AGE_2]," \
                                             " [GENDER_1], or [COUNTRY_2],.."
    return convert_inputs_targets_to_messages(
        inputs,
        row['masked_text'],
        'pii-masking-200k'
    )


def prepare_no_robots(row):
    return convert_inputs_targets_to_messages(
        row["messages"][0]["content"],
        row["messages"][1]["content"],
        row["category"]
    )
  
  
def prepare_help_steer(row):
    return convert_inputs_targets_to_messages(
        row["prompt"],
        row["response"],
        "HelpSteer"
    )


def prepare_bactrianx(row):
    input_col = row["input"] or ""  # input can be None
    inputs = row["instruction"] + " " + input_col
    outputs = row["output"]
    return [
        {"from": "user", "text": inputs, "parent": row["_source"]},
        {"from": "assistant", "text": outputs, "parent": 0},
    ]
