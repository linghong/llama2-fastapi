from transformers import pipeline
from jinja2 import Template

def generate_text_pipeline(model, tokenizer, prompt, max_new_tokens=300):
    generate_text = pipeline(
        'text-generation',  # task name must be the first argument
        model=model, 
        tokenizer=tokenizer,
        config={
            'max_new_tokens': max_new_tokens,
            'return_full_text': True,
            'do_sample': True,
            'temperature': 0.1,
            'top_p': 0.95,
            'top_k': 40,
            'repetition_penalty': 1.1
        }
    )
    res = generate_text(prompt)
    ai_response = (res[0]["generated_text"])

    return ai_response

def generate_text_phi1_5(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(
        prompt, 
        return_tensors='pt', return_attention_mask=False
    )
    
    outputs = model.generate(**inputs, max_length=max_new_tokens)
    generated_text = tokenizer.batch_decode(outputs)[0]
   
    return generated_text

# this is a temporary solution
def create_prompt(models, model_name, base_prompt, question, chat_history, fetched_text):
    # Get the template for the specified model
    template_str = models[model_name]['prompt_template']
    template = Template(template_str)

    system_prompt = "You are an AI assistant, skilled and equipped with a specialized data source as well as a vast reservoir of general knowledge. When a user presents a question, they can prompt you to extract relevant information from this data source. If information is obtained, it will be flagged with '''fStart and closed with fEnd'''. Only use the fetched data if it is directly relevant to the user's question and can contribute to a reasonable correct answer. Otherwise, rely on your pre-existing knowledge to provide the best possible response. Also, only give answer for the question asked, don't provide text not related to the user's question. "

    if fetched_text != "":
        user_message = f"{question}\n'''fStart {fetched_text} fEnd'''"
    else:
        user_message = question

    combined_chat_history = chat_history.push({"question":  user_message})

    prompt = template.render(system=system_prompt + base_prompt, instructions=combined_chat_history)
    
    return prompt