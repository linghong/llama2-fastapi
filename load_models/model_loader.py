import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import HUGGINGFACE_ACCESS_TOKEN, CACHE_DIR

def load_model(
    model_name: str, 
    require_auth: bool = False, 
    trust_remote: bool = False,
    cache_dir: str = CACHE_DIR
):
    """
    Loads a given model from Hugging Face or a local cache.

    Parameters:
    - model_name (str): Name of the model to be loaded.
    - require_auth (bool): Whether authentication is required. Default is False.
    - trust_remote (bool): Whether to trust remote code. Default is False.
    - cache_dir (str): Directory to store cached models. Default is "./load_models/models".

    Returns:
    - tuple: Loaded model and its tokenizer.
    """

    hf_auth = HUGGINGFACE_ACCESS_TOKEN if require_auth else None
    model_path = os.path.join(cache_dir, model_name)
    
    try:
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
            
            logging.info(f"Model {model_name} loaded from Hugging Face")

            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            logging.info(f"Model {model_name} saved locally for future use")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote
            )
            logging.info(f"Model {model_name} loaded from local")
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {str(e)}")
        return None, None
    return model, tokenizer

def load_models(models):
    """
    Preloads models based on a given dictionary of model configurations.

    Parameters:
    - models (dict): A dictionary containing model configurations.

    Returns:
    - dict: A dictionary containing the loaded models and tokenizers.
    """

    loaded_models = {}
    
    for key, value in models.items():       
        preload = value["preload"]
        if preload:
            model_name = value['name']
            require_auth = value['require_auth']
            trust_remote_code = value['trust_remote_code']

            try:
                model, tokenizer = load_model(model_name, require_auth, trust_remote_code)
                
                if model is not None and tokenizer is not None:
                    loaded_models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer
                    }
                else:
                    logging.warning(f'Skipped loading model: {model_name}')
            except Exception as e:
                logging.error(f"An error occurred while loading the model {model_name}: {str(e)}")
 
    return loaded_models
