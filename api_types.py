from enum import Enum
from pydantic import BaseModel


# These are some BaseModel derived classes to help handle POST requests.
class RuleRequest(BaseModel):
    X_2: float
    Y_2: float
    Heading_2: float
    Speed_2: float


class ModelRequest(BaseModel):
    model: str


# As the models classify text into numbers ranging 0-2, this dictionary helps map them to the corresponding COLREG rules.

rule_translator = {0: 13, 1: 14, 2: 15}
