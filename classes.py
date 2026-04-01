from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

class ABCUnit(BaseModel):
    antecedent: Optional[str] = Field(None, description="O contexto imediato anterior")
    behavior: str = Field(description="O comportamento divergente exato extraído do texto")
    consequence: Optional[str] = Field(None, description="A resposta imediata posterior")
    antecedent_snippet: Optional[str] = Field(description="Trecho original contendo o antecedente envolto EXATAMENTE por <A> e </A>. Null se não houver.")
    behavior_snippet: str = Field(description="Trecho original contendo o comportamento envolto EXATAMENTE por <B> e </B>.")
    consequence_snippet: Optional[str] = Field(description="Trecho original contendo a consequência envolta EXATAMENTE por <C> e </C>. Null se não houver.")
    justification: str = Field(description="Justificativa clínica para este rótulo")

class ABCEvaluationState(TypedDict):
    observation: str
    identified_behaviors: List[str]
    final_episodes: List[ABCUnit]