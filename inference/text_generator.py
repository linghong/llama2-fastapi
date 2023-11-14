import logging
from transformers import pipeline
from jinja2 import Template


def generate_text_pipeline(model, tokenizer, prompt, max_new_tokens=300):
    """
    Generates text using the given model and tokenizer with pipeline.

    Parameters:
    - model: Preloaded model for text generation.
    - tokenizer: Preloaded tokenizer for text generation.
    - prompt (str): The text prompt to begin generation.
    - max_new_tokens (int): The maximum number of tokens for the generated text.

    Returns:
    - str: The generated text.
    """

    if model is None or tokenizer is None:
        logging.error("Model or tokenizer is None in generate_text_pipeline.")
        return None

    try:
        generate_text = pipeline(
            "text-generation",  # task name must be the first argument
            model=model.to("cuda"),
            tokenizer=tokenizer,
            device=0,
            config={
                "max_new_tokens": max_new_tokens,
                "return_full_text": True,
                "do_sample": True,
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "repetition_penalty": 1.1,
            },
        )
        res = generate_text(prompt)
        prompt_len = len(prompt)
        ai_response = res[0]["generated_text"][prompt_len:]

    except Exception as e:
        logging.error(f"An error occured in generate_text_pipeline: {str(e)}")
        return None
    return ai_response


def generate_text_phi1_5(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generates text using the given phi-1_5 model and tokenizer.

    Parameters:
    - model: Preloaded phi-1_5 model for text generation.
    - tokenizer: Preloaded tokenizer for text generation.
    - prompt (str): The text prompt to begin generation.
    - max_new_tokens (int): The maximum number of tokens for the generated text. Default is 50.

    Returns:
    - str: The generated text.
    """

    if model is None or tokenizer is None:
        logging.error("Model or tokenizer is None in generate_text_phi1_5.")
        return None

    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

        outputs = model.generate(**inputs, max_length=max_new_tokens)
        generated_text = tokenizer.batch_decode(outputs)[0]
    except Exception as e:
        logging.error(f"An error occurred in generate_text_phil5: {str(e)}")
        return None
    return generated_text


def create_prompt(
    models, full_model_name, base_prompt, question, chat_history, fetched_text
):
    """
    Creates a text generation prompt based on specified parameters.

    Parameters:
    - models (dict): Dictionary of available models and their configurations.
    - model_name (str): The name of the model to use for generating the prompt.
    - base_prompt (str): The base prompt comes from the user.
    - question (str): The user's question.
    - chat_history: Previous chat history.
    - fetched_text (str): Fetched text from a vector database, if any.

    Returns:
    - str: The generated prompt.
    """

    try:
        model_name = full_model_name.split("/")[1]
        template_str = models[model_name]["prompt_template"]
        template = Template(template_str)

        system_prompt = "You are an AI assistant, skilled and equipped with a specialized data source as well as a vast reservoir of general knowledge. When a user presents a question, they can prompt you to extract relevant information from this data source. If information is obtained, it will be flagged with '''fStart and closed with fEnd'''. Only use the fetched data if it is directly relevant to the user's question and can contribute to a reasonable correct answer. Otherwise, rely on your pre-existing knowledge to provide the best possible response. Also, only give answer for the question asked, don't provide text not related to the user's question. "
        system_combined = system_prompt + base_prompt

        if fetched_text != "":
            user_message = f"{question}\n'''fStart {fetched_text} fEnd'''"
        else:
            user_message = question

        prompt = template.render(
            system=system_combined, instructions=chat_history, question=user_message
        )
    except Exception as e:
        logging.error(f"An error occurred in create_prompt: {str(e)}")
        return False, str(e)
    return True, prompt
