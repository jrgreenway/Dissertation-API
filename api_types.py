from enum import Enum
from pydantic import BaseModel


class RuleRequest(BaseModel):
    X_2: float
    Y_2: float
    Heading_2: float
    Speed_2: float


rule_translator = {0: 13, 1: 14, 2: 15}
