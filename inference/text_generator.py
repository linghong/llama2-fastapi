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

# this is temporary solution
def create_prompt(models, model_name, chat_history):
    # Get the template for the specified model
    template_str = models[model_name]['prompt_template']
    template = Template(template_str)

    system_prompt="You are an AI assistant"
    prompt = template.render(system=system_prompt, instructions=chat_history)
    
    return prompt