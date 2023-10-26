import logging
from openai import File, FineTuningJob


async def upload_training_file(file_content):
    try:
        logging.info("sending data to OpenAI...")
        res = File.create(file=file_content, purpose="fine-tune")
        return res["id"]
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")


async def fine_tune_openai_model(file_id, fine_tuning_model, suffix, n_epochs):
    try:
        parameters = {
            "training_file": file_id,
            "model": fine_tuning_model,
        }
        if suffix != "":
            parameters["suffix"] = suffix
        if n_epochs != "":
            parameters["hyperparameters"] = {"n_epochs": n_epochs}

        res = FineTuningJob.create(**parameters)
        return res["fine_tuned_model"]
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")
