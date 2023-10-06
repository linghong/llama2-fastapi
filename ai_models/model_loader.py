import os
from transformers import BitsAndBytesConfig, AutoConfig, AutoTokenizer, AutoModelForCausalLM, GPTQConfig

from config import HUGGINGFACE_ACCESS_TOKEN

# for base model that requires auth, such as llama2, with no quantization
def load_model_base(model_name: str, cache_dir: str = "./ai_models"):
    hf_auth = HUGGINGFACE_ACCESS_TOKEN
    if not os.path.exists(model_path): 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            use_auth_token=hf_auth
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            use_auth_token=hf_auth
        )
        print(f"Model loaded from Hugging Face")

        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print(f"Model saved locally for future use")
    
    else:
        model_path = os.path.join(cache_dir, model_name)       
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
        )
        print(f"Model loaded from local")

    return tokenizer, model    
