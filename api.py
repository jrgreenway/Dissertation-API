from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, Request, Response
from pydantic import BaseModel
from api_types import RuleRequest, rule_translator
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import argparse
import torch
from utils import format_text, start_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
MODEL = args.model
PORT = args.port

api = FastAPI()


@asynccontextmanager
async def lifespan(MODEL):
    global model
    global tokeniser
    global device
    model, tokeniser = start_model(MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    yield


@api.post("/get")
async def get_rule(request: RuleRequest):

    inputs = format_text(tokeniser, request)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class label
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return {"label": rule_translator[predicted_class_id]}


@api.get("/shutdown")
async def shutdown():
    def shutdown_server():
        import os
        import signal

        os.kill(os.getpid(), signal.SIGINT)

    BackgroundTasks(shutdown_server())
    return Response()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="127.0.0.1", port=int(PORT))
