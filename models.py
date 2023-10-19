from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class ChatMessage(BaseModel):
    question: str
    answer: str

class ChatMessages(BaseModel):
    question: str
    chat_history: List[ChatMessage] = Field(alias="chatHistory")
    selected_model: str = Field(alias="selectedModel")
    fetched_text: str = Field(alias="fetchedText")

class FineTuningSpecs(BaseModel):
    finetuning: str
    epochs: Optional[int]
    batchsize: Optional[int]
    learning_rate_multiplier: Optional[float]
    prompt_loss_weight: Optional[float]