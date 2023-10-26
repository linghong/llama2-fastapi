# Temporary setup for testing purposes.
# Currently supports model names "meta-llama/Llama-2-7b-chat-hf" and "microsoft/phi-1_5".
# To extend functionality, add additional models of interest to the "models" dictionary below.
models = {
    "Llama-2-7b-chat-hf": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "type": "chat",
        "require_auth": True,
        "trust_remote_code": False,
        "additional_packages": [],
        "preload": False,
        "prompt_template": """<s>[INST]<SYS>>\n{{ system }}\n<</SYS>>\n\n{% for item in instructions %}[INST]{{ item.question }}[/INST]{{ item.answer }}{% endfor %}</s>""",
    },
    "phi-1_5": {
        "name": "microsoft/phi-1_5",
        "type": "base",
        "require_auth": False,
        "trust_remote_code": True,
        "additional_packages": ["einops"],
        "preload": True,
        "prompt_template": [],
    },
}
