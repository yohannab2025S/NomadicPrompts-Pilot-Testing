"""Central settings for the Nomadic Prompts study. EDIT the TODO items, then freeze this file.
Any change after the plan is submitted must be recorded in deviations_log.md."""
import os

SEED = 42
FORMAT_LINES = {
    "formal_fallacies": 'End your reply with a final line of exactly "Answer: valid" or "Answer: invalid".',
    "logical_deduction": 'End your reply with a final line of exactly "Answer: (X)" where X is the option letter.',
    "word_sorting": 'End your reply with a final line of exactly "Answer: " followed by the sorted words separated by single spaces.',
}

# TODO: set the final tasks after baselines.py. n_a = Part A (optimization) size, n_b = Part B (held-out) size.
# logical_deduction pools the 3-, 5- and 7-object BBH variants (750 items).
TASKS = {
    "logical_deduction": dict(hf_configs=["logical_deduction_three_objects", "logical_deduction_five_objects",
                                          "logical_deduction_seven_objects"], n_a=100, n_b=200),
    "formal_fallacies": dict(hf_configs=["formal_fallacies"], n_a=100, n_b=150),
    "word_sorting": dict(hf_configs=["word_sorting"], n_a=100, n_b=150),
}

# TODO: fill model_id with the exact strings from the ARC model list / provider docs.
# 'provider_group' = developer (for same- vs cross-provider); 'size' = large/small.
# 'variants' lists reasoning settings to test in baselines.py; after baselines choose ONE as FINAL_SETTING.
# Env vars: ARC_BASE_URL, ARC_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, OPENAI_COMPAT_* for gpt-oss-20b host.
MODELS = {
    "gpt-oss-120b": dict(provider_group="OpenAI", size="large", host="arc", base_url_env="ARC_BASE_URL", key_env="ARC_API_KEY",
                         variants={"thinking-low": "gpt-oss-120b-thinking-low", "thinking-high": "gpt-oss-120b-thinking-high"}),
    "gpt-oss-20b": dict(provider_group="OpenAI", size="small", host="TODO (ARC GPU or paid host)", base_url_env="OSS20_BASE_URL", key_env="OSS20_API_KEY",
                        variants={"default": "TODO-gpt-oss-20b"}),
    "deepseek-large": dict(provider_group="DeepSeek", size="large", host="DeepSeek API", base_url_env="DEEPSEEK_BASE_URL", key_env="DEEPSEEK_API_KEY",
                           variants={"default": "TODO-deepseek-v4-pro"}),
    "deepseek-v4.1-flash": dict(provider_group="DeepSeek", size="small", host="arc", base_url_env="ARC_BASE_URL", key_env="ARC_API_KEY",
                                variants={"thinking-low": "DeepSeek-V4.1-Flash-thinking-low", "thinking-high": "DeepSeek-V4.1-Flash"}) , 
    "glm-5.3": dict(provider_group="Z.ai", size="unclassified", host="arc", base_url_env="ARC_BASE_URL", key_env="ARC_API_KEY",
                        variants={"default": "GLM-5.3", "thinking-high": "GLM-5.3-thinking-high"}),
}
# TODO after baselines: one fixed reasoning setting per model.
FINAL_SETTING = {
    "gpt-oss-120b": "thinking-low",
    "deepseek-v4.1-flash": "thinking-low",
    "glm-5.3": "thinking-high",
}

TEMPERATURE = 0
MAX_TOKENS = 8000          # high enough that reasoning models are not truncated; log finish_reason
MAX_CONCURRENT_ARC = 10    # ARC limit
MAX_CONCURRENT_OTHER = 10
WRITER_MODEL = "claude-sonnet-4-6"
WRITER_TEMPERATURE = 1.0

# TODO: USD per 1M tokens (prompt, completion). Use 0 for free ARC. Cost is computed from COMPLETION tokens.
PRICES = {m: dict(prompt=0.0, completion=0.0) for m in MODELS}
PRICES["writer"] = dict(prompt=3.0, completion=15.0)  # TODO verify current Sonnet price

LOG_PATH = "logs/calls.jsonl"
