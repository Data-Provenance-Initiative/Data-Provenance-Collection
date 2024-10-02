import downloaders as downloaders
import preparers as preparers


COLLECTION_FN_MAPPER = {
    "Open-Platypus": {
        "download_function": downloaders.download_open_platypus,
        "prepare_function": preparers.prepare_open_platypus,
    },
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
    "Capybara": {
        "download_function": downloaders.download_capybara,
        "prepare_function": preparers.prepare_capybara,
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
    "Open Assistant v2": {
        "download_function": downloaders.download_open_assistant_v2,
        "prepare_function": preparers.prepare_open_assistant,
        "custom_prepare": True,
    },
    "Open Assistant OctoPack": {
        "download_function": downloaders.download_open_assistant_octopack,
        "prepare_function": preparers.prepare_oasst_octopack,
    },
    "Indic-Instruct": {
        "download_function": downloaders.download_indic_instruct,
        "prepare_function": preparers.prepare_indic_instruct,
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
    "Glaive Code Assistant": {
        "download_function": downloaders.download_glaive_code_assistant,
        "prepare_function": preparers.prepare_glaive_code_assistant,
    },
    "Glaive Code Assistant v2": {
        "download_function": downloaders.download_glaive_code_assistant_v2,
        "prepare_function": preparers.prepare_glaive_code_assistant_v2,
    },
    "Glaive Code Assistant v3": {
        "download_function": downloaders.download_glaive_code_assistant_v3,
        "prepare_function": preparers.prepare_glaive_code_assistant_v3,
    },
    "Nectar": {
        "download_function": downloaders.download_nectar,
        "prepare_function": preparers.prepare_nectar,
    },
    "MetaMathQA": {
        "download_function": downloaders.download_metamathqa,
        "prepare_function": preparers.prepare_metamathqa,
    },
    "RiddleSense": {
        "download_function": downloaders.download_riddle_sense,
        "prepare_function": preparers.prepare_riddle_sense,
    },
    "EverythingLM": {
        "download_function": downloaders.download_everything_lm,
        "prepare_function": preparers.prepare_everything_lm,
    },
    "GPT-4-Alpaca": {
        "download_function": downloaders.download_gpt4_alpaca,
        "prepare_function": preparers.prepare_gpt4_alpaca,
    },
    "lmsys_chat_1m": {
        "download_function": downloaders.download_lmsys_chat_1m,
        "prepare_function": preparers.prepare_lmsys_chat_1m,
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
    "Feedback Collection": {
        "download_function": downloaders.download_feedback_collection,
        "prepare_function": preparers.prepare_feedback_collection,
    },
    "Preference Collection": {
        "download_function": downloaders.download_preference_collection,
        "prepare_function": preparers.prepare_preference_collection,
    },
    "MagPie-Pro": {
        "download_function": downloaders.download_magpie
        "prepare_function": preparers.prepare_magie
    },
    "Synthetic-GSM8K-Reflection": {
        "download_function":, downloaders.download_synthetic_gsm8k_reflection
        "prepare_function": preparers.download_synthetic_gsm8k_reflection
    }
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
    "SEACrowd": {
        "download_function": downloaders.download_seacrowd,
        "prepare_function": preparers.prepare_seacrowd,
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
        "prepare_function": preparers.prepare_tasksource_instruct,
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
    "UltraChat_200k": {
        "download_function": downloaders.download_ultrachat_200k,
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
    "MathInstruct": {
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
    "PMC-LLaMA Instructions": {
        "download_function": downloaders.download_pmc_llama,
        "prepare_function": preparers.prepare_pmc_llama,
    },
    "Medical Meadow": {
        "download_function": downloaders.download_medical_meadow,
        "prepare_function": preparers.prepare_medical_meadow,
    },
    "MedInstruct": {
        "download_function": downloaders.download_medinstruct,
        "prepare_function": preparers.prepare_medinstruct,
    },
    "Open Orca": {
        "download_function": downloaders.download_open_orca,
        "prepare_function": preparers.prepare_open_orca,
    },
    "ToxicChat": {
        "download_function": downloaders.download_toxicchat,
        "prepare_function": preparers.prepare_toxicchat,
    },
    "COIG": {
        "download_function": downloaders.download_coig,
        "prepare_function": preparers.prepare_coig,
    },
    # "COIG-Kun":{
    #     "download_function": downloaders.download_coig_kun,
    #     "prepare_function": preparers.prepare_coig_kun,
    # },
    "COIG-CQIA": {
        "download_function": downloaders.download_coig_cqia,
        "prepare_function": preparers.prepare_coig_cqia,
    },
    "SelFee": {
        "download_function": downloaders.download_selfee,
        "prepare_function": preparers.prepare_selfee,
    },
    "SeaBench": {
        "download_function": downloaders.download_seabench,
        "prepare_function": preparers.prepare_seabench,
    },
    "Deita 10K": {
        "download_function": downloaders.download_deita_10k,
        "prepare_function": preparers.prepare_deita_10k,
    },
    "ChatDoctor": {
        "download_function": downloaders.download_chatdoctor,
        "prepare_function": preparers.prepare_chatdoctor,
    },
    "AgentInstruct": {
        "download_function": downloaders.download_agentinstruct,
        "prepare_function": preparers.prepare_agentinstruct,
    },
    "Cidar": {
        "download_function": downloaders.download_cidar,
        "prepare_function": preparers.prepare_cidar,
    },
    "PII-Masking-200k": {
        "download_function": downloaders.download_pii_masking_200k,
        "prepare_function": preparers.prepare_pii_masking_200k,
    },
    "No Robots": {
        "download_function": downloaders.download_no_robots,
        "prepare_function": preparers.prepare_no_robots,
    },
    "HelpSteer": {
        "download_function": downloaders.download_help_steer,
        "prepare_function": preparers.prepare_help_steer,
    },
    "Bactrian-X": {
        "download_function": downloaders.download_bactrianx,
        "prepare_function": preparers.prepare_bactrianx,
    },
    "WildChat": {
        "download_function": downloaders.download_wildchat,
        "prepare_function": preparers.prepare_wildchat,
    },
    "PygmalionAI-PIPPA": {
        "download_function": downloaders.download_pippa,
        "prepare_function": preparers.prepare_pippa,
    },
    "CollectiveCognition": {
        "download_function": downloaders.download_collective_cognition,
        "prepare_function": preparers.prepare_collective_cognition,
    },
    "ChatbotArena": {
        "download_function": downloaders.download_chatbot_arena_conversations,
        "prepare_function": preparers.prepare_chatbot_arena_conversations,
    },
    "KIWI": {
        "download_function": downloaders.download_kiwi,
        "prepare_function": preparers.prepare_kiwi,
    },
    "Orca-Math": {
        "download_function": downloaders.download_orca_math,
        "prepare_function": preparers.prepare_orca_math,
    },
    "Cobra Frames": {
        "download_function": downloaders.download_cobra_frames,
        "prepare_function": preparers.prepare_cobra_frames,
    },
    "MathDial": {
        "download_function": downloaders.download_mathdial,
        "prepare_function": preparers.prepare_mathdial,
    },
    "10k Prompt Ranked": {
        "download_function": downloaders.download_10k_prompt_ranked,
        "prepare_function": preparers.prepare_10k_prompt_ranked,
    },
    "Aya Dataset": {
        "download_function": downloaders.download_aya_dataset,
        "prepare_function": preparers.prepare_aya_dataset,
    },
    "MegaWika": {
        "download_function": downloaders.download_megawika,
        "prepare_function": preparers.prepare_megawika,
    },
    "Gretel Text-to-SQL": {
        "download_function": downloaders.download_gretel_text_to_sql,
        "prepare_function": preparers.prepare_gretel_text_to_sql,
    },
    "ExpertQA": {
        "download_function": downloaders.download_expertqa,
        "prepare_function": preparers.prepare_expertqa,
    },
    "OpenMathInstruct-1": {
        "download_function": downloaders.download_openmath_instruct,
        "prepare_function": preparers.prepare_openmath_instruct,
    },
    "OpenGPT Healthcare": {
        "download_function": downloaders.download_opengpt_healthcare,
        "prepare_function": preparers.prepare_opengpt_healthcare,
    },
    "Conifer": {
        "download_function": downloaders.download_conifer,
        "prepare_function": preparers.prepare_conifer,
    },
    "Reasoning": {
        "download_function": downloaders.download_reasoning,
        "prepare_function": preparers.prepare_reasoning,
    },
    "DialogStudio": {
        "download_function": downloaders.download_dialogstudio,
        "prepare_function": preparers.prepare_dialogstudio,
    },
    "UltraFeedback Argilla": {
        "download_function": downloaders.download_ultraFeedback_argilla,
        "prepare_function": preparers.prepare_ultraFeedback_argilla,
    },
    "LongAlign-10k": {
        "download_function": downloaders.download_longalign_10k,
        "prepare_function": preparers.prepare_longalign_10k,
    },
    "Dynosaur": {
        "download_function": downloaders.download_dynosaur,
        "prepare_function": preparers.prepare_dynosaur,
    },
    "Lumos Planning": {
        "download_function": downloaders.download_lumos_planning,
        "prepare_function": preparers.prepare_lumos_planning,
    },
    "Lumos Grounding": {
        "download_function": downloaders.download_lumos_grounding,
        "prepare_function": preparers.prepare_lumos_grounding,
    },
    "InstAr": {
        "download_function": downloaders.download_inst_ar,
        "prepare_function": preparers.prepare_inst_ar,
    }
}