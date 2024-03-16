import downloaders as downloaders
import preparers as preparers


COLLECTION_FN_MAPPER = {
    "Flan Collection (Super-NaturalInstructions)": {
        "download_function": downloaders.download_flan_collection_sni,
        "prepare_function": preparers.prepare_flan_collection,
    },
    "Flan Collection (Chain-of-Thought)": {
        "download_function": downloaders.download_flan_collection_cot,
        "prepare_function": preparers.prepare_flan_collection,
    },
    "Flan Collection (Dialog)": {
        "download_function": downloaders.download_flan_collection_dialog,
        "prepare_function": preparers.prepare_flan_collection,
    },
    "Flan Collection (Flan 2021)": {
        "download_function": downloaders.download_flan_collection_flan2021,
        "prepare_function": preparers.prepare_flan_collection,
    },
    "Flan Collection (P3)": {
        "download_function": downloaders.download_flan_collection_p3,
        "prepare_function": preparers.prepare_flan_collection,
    },
    "Dolly 15k": {
        "download_function": downloaders.download_dolly_15k,
        "prepare_function": preparers.prepare_dolly_15k,
    },
    "xP3x": {
        "download_function": downloaders.download_xp3x,
        "prepare_function": preparers.prepare_xp3x,
    },
    "CommitPackFT": {
        "download_function": downloaders.download_commitpackft,
        "prepare_function": preparers.prepare_commitpackft,
    },
    "Anthropic HH-RLHF": {
        "download_function": downloaders.download_anthropic_hh_rlhf,
        "prepare_function": preparers.prepare_anthropic_hh_rlhf,
    },
    "Self-Instruct": {
        "download_function": downloaders.download_self_instruct,
        "prepare_function": preparers.prepare_self_instuct,
    },
    "Stanford Human Preferences": {
        "download_function": downloaders.download_stanford_human_preferences,
        "prepare_function": preparers.prepare_stanford_human_preferences,
    },
    "Open Assistant": {
        "download_function": downloaders.download_open_assistant,
        "prepare_function": preparers.prepare_open_assistant,
        "custom_prepare": True,
    },
    "Open Assistant OctoPack": {
        "download_function": downloaders.download_open_assistant_octopack,
        "prepare_function": preparers.prepare_oasst_octopack,
    },
    "OpenAI (Summarize from Feedback)": {
        "download_function": downloaders.download_openai_summarization,
        "prepare_function": preparers.prepare_openai_summarization,
    },
    "OpenAI (WebGPT)": {
        "download_function": downloaders.download_openai_webgpt,
        "prepare_function": preparers.prepare_openai_webgpt,
    },
    "Longform": {
        "download_function": downloaders.download_longform,
        "prepare_function": preparers.prepare_longform,
    },
    "GPTeacher": {
        "download_function": downloaders.download_gpteacher,
        "prepare_function": preparers.prepare_gpteacher,
    },
    "Alpaca": {
        "download_function": downloaders.download_alpaca,
        "prepare_function": preparers.prepare_alpaca,
    },
    "MetaMathQA": {
        "download_function": downloaders.download_metamathqa,
        "prepare_function": preparers.prepare_metamathqa,
    },
    "EverythingLM": {
        "download_function": downloaders.download_everything_lm,
        "prepare_function": preparers.prepare_everything_lm,
    },
    "GPT-4-Alpaca": {
        "download_function": downloaders.download_gpt4_alpaca,
        "prepare_function": preparers.prepare_gpt4_alpaca,
    },
    "WizardLM Evol-Instruct": {
        "download_function": downloaders.download_evol_instruct,
        "prepare_function": preparers.prepare_evol_instruct,
    },
    "WizardLM Evol-Instruct V2": {
        "download_function": downloaders.download_evol_instruct_v2,
        "prepare_function": preparers.prepare_evol_instruct_v2,
    },
    "Pure-Dove": {
        "download_function": downloaders.download_pure_dove,
        "prepare_function": preparers.prepare_pure_dove,
    },
    "Llama2-MedTuned-Instructions": {
        "download_function": downloaders.download_llama2_med_tuned_instructions,
        "prepare_function": preparers.prepare_llama2_med_tuned_instructions,
    },
    "OIG": {
        "download_function": downloaders.download_laion_oig,
        "prepare_function": preparers.prepare_laion_oig,
    },
    "Thai Gen AI (Alpaca)": {
        "download_function": downloaders.download_thai_gen_ai_alpaca,
        "prepare_function": preparers.prepare_thai_gen_ai_alpaca,
    },
    "ShareGPT Vicuna": {
        "download_function": downloaders.download_sharegpt_vicuna,
        "prepare_function": preparers.prepare_sharegpt_vicuna,
    },
    "Code Alpaca": {
        "download_function": downloaders.download_code_alpaca,
        "prepare_function": preparers.prepare_code_alpaca,
    },
    "HC3 (English)": {
        "download_function": downloaders.download_hc3_en,
        "prepare_function": preparers.prepare_hc3_en,
    },
    "HC3 (Chinese)": {
        "download_function": downloaders.download_hc3_zh,
        "prepare_function": preparers.prepare_hc3_zh,
    },
    "Camel-AI Science": {
        "download_function": downloaders.download_camel_science,
        "prepare_function": preparers.prepare_camel_science,
    },
    "CoT Collection": {
        "download_function": downloaders.download_cot_collection,
        "prepare_function": preparers.prepare_cot_collection,
    },
    "NomicAI GPT4AllJ": {
        "download_function": downloaders.download_gpt4all,
        "prepare_function": preparers.prepare_gpt4all,
    },
    "Unnatural Instructions": {
        "download_function": downloaders.download_unnatural_instructions,
        "prepare_function": preparers.prepare_unnatural_instructions,
    },
    "StarCoder Self-Instruct": {
        "download_function": downloaders.download_starcoder_self_instruct,
        "prepare_function": preparers.prepare_starcoder_self_instruct,
    },
    "Thai Gen AI (GPTeacher)": {
        "download_function": downloaders.download_thai_gen_ai_gpteacher,
        "prepare_function": preparers.prepare_thai_gen_ai_gpteacher,
    },
    "Tiny Stories": {
        "download_function": downloaders.download_tiny_stories,
        "prepare_function": preparers.prepare_tiny_stories,
        "custom_prepare": True,
    },
    "Thai Gen AI (Dolly)": {
        "download_function": downloaders.download_thai_gen_ai_dolly,
        "prepare_function": preparers.prepare_thai_gen_ai_dolly,
    },
    "Tasksource Instruct": {
        "download_function": downloaders.download_tasksource_instruct,
        "prepare_function": preparers.prepare_tasksource_instruct,
    },
    "Tasksource Symbol-Tuning": {
        "download_function": downloaders.download_tasksource_symbol_tuning,
        "prepare_function": preparers.prepare_tasksource_instruct
    },
    "Stack Exchange Instruction": {
        "download_function": downloaders.download_stack_exchange_instruction,
        "prepare_function": preparers.prepare_stack_exchange_instruction,
    },
    "Joke Explanation": {
        "download_function": downloaders.download_joke_explanation,
        "prepare_function": preparers.prepare_joke_explanation,
    },
    "Book Summaries": {
        "download_function": downloaders.download_book_summaries,
        "prepare_function": preparers.prepare_book_summaries,
    },
    "UltraChat": {
        "download_function": downloaders.download_ultrachat,
        "prepare_function": preparers.prepare_ultrachat,
    },
    "Airoboros": {
        "download_function": downloaders.download_airoboros,
        "prepare_function": preparers.prepare_airoboros,
    },
    "LIMA": {
        "download_function": downloaders.download_lima,
        "prepare_function": preparers.prepare_lima,
    },
    "MathInstruct":{
        "download_function": downloaders.download_mathinstruct,
        "prepare_function": preparers.prepare_mathinstruct,
    },
    "Tool-Llama": {
        "download_function": downloaders.download_tool_llama,
        "prepare_function": preparers.prepare_tool_llama,
    },
    "Gorilla": {
        "download_function": downloaders.download_gorilla,
        "prepare_function": preparers.prepare_gorilla,
    },
    "Baize Chat Data": {
        "download_function": downloaders.download_baize_data,
        "prepare_function": preparers.prepare_baize_data,
    },
    "Open Orca": {
        "download_function": downloaders.download_open_orca,
        "prepare_function": preparers.prepare_open_orca,
    },
    "AgentInstruct": {
        "download_function": downloaders.download_agentinstruct,
        "prepare_function": preparers.prepare_agentinstruct,
    },
    "PII-Masking-200k": {
        "download_function": downloaders.download_pii_masking_200k,
        "prepare_function": preparers.prepare_pii_masking_200k,
    }
}
