import json
import logging
from datetime import timedelta
from typing import Annotated, Optional
import openai
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    status,
    Header,
    Depends,
    Form,
    File,
)
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

from config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    API_SECRET_KEY,
    OPENAI_API_KEY,
    YOUR_CLIENT_SITE_ADDRESS,
)
from database import fake_users_db
from finetuning.openai import fine_tune_openai_model, upload_training_file
from finetuning.validation import validate_data_format, validate_messages
from load_models.model_list import models
from load_models.model_loader import load_model, load_models
from inference.text_generator import (
    create_prompt,
    generate_text_phi1_5,
    generate_text_pipeline,
)
from models import ChatMessages, FineTuningSpecs, Token, User
from user_auth import authenticate_user, create_access_token, get_current_active_user


logging.basicConfig(
    filename="application.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting a new instance of Smartchat FastAPI application...")

if not API_SECRET_KEY:
    logging.error(f"Unable to Fetch API Secert Key")
    raise Exception("Unable to Fetch API Secert Key")

app = FastAPI()

loaded_models = load_models(models)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", YOUR_CLIENT_SITE_ADDRESS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)


async def get_api_secret_key(authorization: str = Header(...)):
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(
            status_code=401, detail="Unauthorized: Invalid API Secret Key"
        )
    api_secret_key = authorization[len(prefix) :]

    if api_secret_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=401, detail="Unauthorized: Invalid API Secret Key"
        )
    return api_secret_key


@app.post("/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return current_user


@app.get("/")
async def root(current_user: User = Depends(get_current_active_user)):
    if current_user:
        return {"message": "Hello World"}
    return {"message": "Unauthorized"}


@app.post("/api/chat/opensourcemodel")
async def chat(
    chat_messages: ChatMessages, api_secret_key: str = Depends(get_api_secret_key)
):
    model_name = chat_messages.selected_model
    question = chat_messages.question

    # Ensure that 'model_name' is a valid key in 'loaded_models'
    model_key = model_name.split("/").pop()
    if model_key not in models.keys():
        raise HTTPException(status_code=400, detail="Invalid model name")

    if model_name not in loaded_models.keys():
        current_model = models[model_key]
        require_auth = current_model["require_auth"]
        trust_remote_code = current_model["trust_remote_code"]

        model, tokenizer = load_model(model_name, require_auth, trust_remote_code)
        loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}

    model = loaded_models[model_name]["model"]
    tokenizer = loaded_models[model_name]["tokenizer"]

    try:
        if model_name == "microsoft/phi-1_5":
            generated_text = generate_text_phi1_5(model, tokenizer, question)

        else:
            chat_history = chat_messages.chat_history
            base_prompt = chat_messages.base_prompt
            fetched_text = chat_messages.fetched_text
            success, prompt = create_prompt(
                models, model_name, base_prompt, question, chat_history, fetched_text
            )
            generated_text = generate_text_pipeline(model, tokenizer, prompt)

        return {"success": True, "message": generated_text}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/finetuning/openai")
async def finetune(
    file: UploadFile = File(...),
    fine_tuning_model: str = Form(..., alias="finetuning"),
    suffix: str = Optional[str],
    n_epochs: int = Form(..., alias="epochs"),
    api_secret_key: str = Depends(get_api_secret_key),
):
    file_content = await file.read()
    file_str = file_content.decode("utf-8")
    file_list = [json.loads(line) for line in file_str.splitlines() if line]

    data_format_errors = validate_data_format(file_list)
    messages_errors = validate_messages(file_list)

    if not messages_errors and not data_format_errors:
        try:
            openai.api_key = OPENAI_API_KEY

            file_submit_result = await upload_training_file(file_content)
            file_id = file_submit_result["id"]

            res = await fine_tune_openai_model(
                file_id, fine_tuning_model, suffix, n_epochs
            )
            fine_tuning_job_id = res["id"]

            return {
                "success": True,
                "id": fine_tuning_job_id,
                "message": "Your request has been successfully sent to OpenAI",
            }
        except Exception as e:
            raise HTTPException(detail=str(e))
    else:
        errors = {"data_format": validate_data_format, "messages": validate_messages}
        return {"success": False, "id": fine_tuning_job_id, "error": errors}


@app.post("/api/finetuning/peft")
async def finetune(
    file: UploadFile = File(...),
    fine_tuning_model: str = Form(..., alias="finetuning"),
    epochs: int = Form(...),
    batch_size: Optional[int] = Form(None, alias="batchSize"),
    learning_rate_multiplier: Optional[float] = Form(
        None, alias="learningRateMultiplier"
    ),
    prompt_loss_weight: Optional[float] = Form(None, alias="promptLossWeight"),
    api_secret_key: str = Depends(get_api_secret_key),
):
    content = await file.read()
    specs = FineTuningSpecs(
        fine_tuning_model=fine_tuning_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
        prompt_loss_weight=prompt_loss_weight,
    )
    # process and validate the uploaded file/data
    # run fine-tuning work (DeepSpeed ZeRO, LoRA, Flash Attention)
    # pass
