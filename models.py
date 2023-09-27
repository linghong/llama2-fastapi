from enum import Enum
from pydantic import BaseModel
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
    chatHistory: List[ChatMessage]
    selectedModel: str

class FineTuningSpecs(BaseModel):
    finetuning: str
    epochs: Optional[int]
    batchsize: Optional[int]
    learning_rate_multiplier: Optional[float]
    prompt_loss_weight: Optional[float]