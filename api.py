from fastapi import FastAPI, HTTPException
from api_types import RuleRequest, rule_translator, ModelRequest
import torch
from utils import format_text, start_model


PORT = 8000
model = None
tokeniser = None
device = None

api = FastAPI()


@api.post("/model")
async def load_model(request: ModelRequest):
    # This function uses start_model to load a saved model, on request from the client.
    print(request)
    global model
    global tokeniser
    global device
    model, tokeniser = start_model(request.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return {"model": request.model}


@api.post("/get")
async def get_rule(request: RuleRequest):
    # This method retrieves the LLM's prediction of the rule, with request containing the relevent ship infomation from the client.
    print(request)
    if model is None or tokeniser is None:
        raise HTTPException(
            status_code=400, detail="Model and tokeniser are not initialized"
        )

    inputs = format_text(tokeniser, request)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return {"label": rule_translator[predicted_class_id]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="127.0.0.1", port=int(PORT))
