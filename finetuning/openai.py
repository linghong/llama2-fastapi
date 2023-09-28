import openai
import os

async def upload_training_file(file_content):
    try:
        print('sending data to OpenAI...')
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        openai.api_key = os.environ["OPENAI_API_KEY"]

        res = openai.File.create(
            file=file_content,
            purpose='fine-tune' 
        )
        return res["id"]
    except Exception as e:
        print(f"An error occurred: {str(e)}")

async def fine_tune_openai_model(file_id, fine_tuning_model, epochs):
    try:
        res = openai.FineTuningJob.create(
            training_file=file_id,
            model=fine_tuning_model,
            epochs=epochs
        )
        return  res["fine_tuned_model"]
    except Exception as e:
        print(f"An error occurred: {str(e)}")

