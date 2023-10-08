from transformers import pipeline

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
    inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
    
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.batch_decode(outputs)[0]
   
    return generated_text
