import os
from openai import File, FineTuningJob

async def upload_training_file(file_content):
    try:
        print('sending data to OpenAI...')
        res = File.create(
            file=file_content,
            purpose='fine-tune' 
        )
        return res["id"]
    except Exception as e:
        print(f"An error occurred: {str(e)}")

async def fine_tune_openai_model(file_id, fine_tuning_model, epochs):
    try:
        res = FineTuningJob.create(
            training_file=file_id,
            model=fine_tuning_model,
            epochs=epochs
        )
        return  res["fine_tuned_model"]
    except Exception as e:
        print(f"An error occurred: {str(e)}")

