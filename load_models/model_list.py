# This is a temporary code, using model_name = 'meta-llama/Llama-2-7b-chat-hf' or "microsoft/phi-1_5" for testing purpose
models = {
  'Llama-2-7b-chat-hf': {
    "name": 'meta-llama/Llama-2-7b-chat-hf',
    "type": "base",
    "require_auth": True,
    "trust_remote_code": False,
    "additional_packages": [],
    "prompt_template": '''<s>[INST]<SYS>>\n{{ system }}\n<</SYS>>\n\n{% for item in instructions %}[INST]{{ item.question }}[/INST]{{ item.answer }}{% endfor %}</s>'''
  },
  "phi-1_5": {
    "name": "microsoft/phi-1_5",
    "type": "base",
    "require_auth": False,
    "trust_remote_code": True,
    "additional_packages": ["einops"],
    "prompt_template":[]
  }
}