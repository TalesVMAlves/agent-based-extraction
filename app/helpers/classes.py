from typing import List, Optional, TypedDict, Dict, Any
from pydantic import BaseModel, Field, create_model

class AgentState(TypedDict):
    observation: str
    extracted_data: Dict[str, Any]
    errors: List[str]

def create_task_model(task_name: str, entities: List[str]) -> type[BaseModel]:
    """
    Cria um modelo Pydantic dinâmico para uma TASK.
    Gera as chaves de extração e snippet dinamicamente para cada entidade na lista.
    """
    fields = {}
    for ent in entities:
        fields[ent] = (Optional[str], Field(None, description=f"A extração exata para {ent}."))
        fields[f"{ent}_snippet"] = (Optional[str], Field(None, description=f"O trecho envolto com a tag <{ent.upper()}>."))
        
    fields["justification"] = (Optional[str], Field(None, description="Justificativa técnica/clínica."))
    
    return create_model(task_name.capitalize(), **fields)

class MetricasClasse(BaseModel):
    analise: str = Field(description="Explique brevemente se o modelo acertou ou errou esta classe e por quê.")
    cobertura_semantica: int = Field(description="1 = Capturou corretamente a ideia principal. 0 = Omitiu ou errou feio.")
    pureza_semantica: int = Field(description="1 = Extração limpa. 0 = Vazamento de fronteira/inversão de papel.")

def create_judge_model(entities: List[str]) -> type[BaseModel]:
    """
    Cria a classe Pydantic de avaliação do Juiz dinamicamente.
    """
    fields = {
        'analise_geral': (str, Field(description="Síntese da qualidade da extração da cadeia.")),
        'nota_final': (int, Field(description="Nota de 1 a 5 baseada na utilidade clínica/técnica."))
    }
    
    for ent in entities:
        key_name = ent.lower()
        fields[key_name] = (MetricasClasse, Field(description=f"Métricas para a classe {ent.capitalize()}"))
        
    return create_model('AvaliacaoJuizDinamica', **fields)