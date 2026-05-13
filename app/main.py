import os
import json
import yaml
import asyncio
import pandas as pd
import mlflow
import copy
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import warnings

from app.services.agent import Agent
from app.helpers.evaluation import run_unified_evaluation
from app.helpers.utils import parse_argilla_responses, extrair_spans_do_agente

logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

def load_merged_config(llm_path: str, experiment_path: str) -> dict:
    """Faz o merge das configurações de LLM e do Experimento específico."""
    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_config = yaml.safe_load(f)
    with open(experiment_path, 'r', encoding='utf-8') as f:
        exp_config = yaml.safe_load(f)
    return {**llm_config, **exp_config}

async def process_dataset(df_test, agent: Agent, entities_names):
    results = []
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Extraindo com Agente"):
        text = str(row["text"])
        true_entities = row.get("entities", [])
        
        try:
            ## O estado começa apenas com a observação
            output = await agent.graph.ainvoke({"observation": text, "extracted_data": {}})
            extracted_dict = output.get("extracted_data", {})
        except Exception as e:
            print(f"Erro na extração do texto índice {idx}: {e}")
            extracted_dict = {}
            
        pred_dicts = extrair_spans_do_agente(text, extracted_data=extracted_dict, entities_names=entities_names)
        
        results.append({
            "text": text,
            "true_entities": json.dumps(true_entities, ensure_ascii=False),
            "pred_entities": json.dumps(pred_dicts, ensure_ascii=False),
            "extracted_raw_dict": json.dumps(extracted_dict, ensure_ascii=False)
        })
        
    return pd.DataFrame(results)

async def main():
    load_dotenv()

    LLM_CONFIG_PATH = 'config/llm_config.yaml'
    experiment_configs = [
        'config/experiments/ibama_1_node.yaml',
        'config/experiments/ibama_2_nodes.yaml',
        'config/experiments/ibama_3_nodes.yaml',
    ]

    providers_to_test = [
        "openai", 
        # "aws"
        ]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    
    mlflow.langchain.autolog(log_traces=True)

    for exp_path in experiment_configs:
        print(f"Carregando configuração: {exp_path}")
        try:
            config = load_merged_config(LLM_CONFIG_PATH, exp_path)
            entities_names = []
            for task in config['domain']['tasks']:
                entities_names.extend(task['entities'])

            mlflow.set_experiment(config['experiment']['name'])
            dataset_path = config['experiment']['dataset_path']
            df_labeled = pd.read_csv(dataset_path)
            df_labeled['entities'] = df_labeled['span_label.responses'].apply(parse_argilla_responses)
    
            split_idx = int(len(df_labeled) * config['experiment']['train_split_ratio'])
            df_test = df_labeled.iloc[split_idx:].copy()

            for provider in providers_to_test:
                current_config = copy.deepcopy(config)
                current_config['llm']['provider'] = provider
                
                original_run_name = current_config['experiment']['mlflow_run']
                run_name_with_model = f"{original_run_name}_{provider}"
                current_config['experiment']['mlflow_run'] = run_name_with_model
                
                print(f"\n Iniciando: {run_name_with_model}")
                agent = Agent(current_config)

                df_preds = await process_dataset(df_test, agent, entities_names)
                await run_unified_evaluation(df_preds, current_config, agent.juiz_chain)
        except FileNotFoundError:
            print(f"O arquivo '{exp_path}' não foi encontrado. Pulando...")
            continue
        except Exception as e:
            print(f"Erro no experimento '{exp_path}': {str(e)}")
            continue
if __name__ == "__main__":
    asyncio.run(main())