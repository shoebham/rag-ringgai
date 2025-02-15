from pydantic import BaseModel
from typing import Dict, Optional

class Query(BaseModel):
    question: str
    context: Optional[Dict] = None 