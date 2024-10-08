import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from api_types import RuleRequest


def start_model(model_name):
    # A helper function which loads pre-saved models and tokenisers.
    model_name = f"models/{model_name}/"
    tokeniser = AutoTokenizer.from_pretrained(model_name + "token")
    model = AutoModelForSequenceClassification.from_pretrained(model_name + "model")
    return model, tokeniser


def format_text(tokeniser, text: RuleRequest):
    # This functin formats and tokenises the POST request body into the predecided sentence format, returning the model inputs.
    formatted_text = (
        f"The primary ship has a position (500, 500), "
        f"a heading of 0 degrees, and a speed of 10 knots. "
        f"The secondary ship has a position ({int(text.X_2)}, {int(text.Y_2)}), "
        f"a heading of {int(text.Heading_2)} degrees, and a speed of {int(text.Speed_2)} knots."
    )

    inputs = tokeniser.encode_plus(
        formatted_text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return inputs
