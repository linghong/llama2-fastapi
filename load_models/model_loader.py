import os
from transformers import AutoTokenizer, AutoModelForCausalLM


from config import HUGGINGFACE_ACCESS_TOKEN

def load_model(
    model_name: str, 
    require_auth: bool = False, 
    trust_remote: bool = False,
    cache_dir: str = "./load_models/models"
):
    hf_auth = HUGGINGFACE_ACCESS_TOKEN if require_auth else None

    model_path = os.path.join(cache_dir, model_name)
    
    if not os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            use_auth_token=hf_auth,
            trust_remote_code=trust_remote
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            use_auth_token=hf_auth,
            trust_remote_code=trust_remote
        )
        
        if require_auth:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded from Hugging Face")

        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print(f"Model saved locally for future use")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote
        )
        print(f"Model loaded from local")

    return model, tokenizer

def load_models(models):
    loaded_models = {}
    
    for key, value in models.items():
        model_name = value['name']
        require_auth = value['require_auth']
        trust_remote_code = value['trust_remote_code']

        model, tokenizer = load_model(model_name, require_auth, trust_remote_code)
        
        loaded_models[key] = {
            'model': model,
            'tokenizer': tokenizer
        }
    
    return loaded_models
