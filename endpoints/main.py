from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Status(BaseModel):  # Dynamic stuff
    state: str  # unstarted, voting, training, waiting
    round: int


class AccuracyDP(BaseModel):
    round: int
    accuracy: float


class AccuracySamples(BaseModel):
    data: List[AccuracyDP]


class Info(BaseModel):  # stuff that won't change
    name: str
    identity: str
    driver_type: str


class DataInfo(BaseModel):
    name: str
    train_size: int
    test_size: int


class Vote(BaseModel):
    round: int
    vote: bool
    am_proposer: bool


class VoteRecord(BaseModel):
    votes: List[Vote]


class Stats(BaseModel):
    average_round_time: float
    average_train_time: float
    average_evaluate_time: float


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/colearn/info/", response_model=Info)
async def create_item():
    return {}


@app.get("/colearn/status", response_model=Status)
def read_root():
    return {}


@app.get("/colearn/accuracy/{dataset}", response_model=AccuracySamples)
def read_root(dataset: str, start: Optional[int] = None, end: Optional[int] = None,
              limit: Optional[int] = None):
    return {}


@app.get("/colearn/votes/", response_model=VoteRecord)
def read_root(start: Optional[int] = None, end: Optional[int] = None, limit: Optional[int] = None):
    return {}


@app.get("/colearn/datainfo", response_model=DataInfo)
def read_root():
    return {}


@app.get("/colearn/stats", response_model=Stats)
def read_root():
    return {}


@app.post('/colearn/start')
def start(starting_info: Dict[Any, Any]):
    return {}


@app.post('/colearn/stop')
def stop():
    return {}
