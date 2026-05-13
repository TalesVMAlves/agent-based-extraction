import os
import yaml
import pandas as pd
import mysql.connector
import argilla as rg
from dotenv import load_dotenv

load_dotenv()
with open(".config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cfg_db = config['database']
cfg_data = config['dataset_db']
mapping = cfg_data['column_mapping']

def coletar_dados():
    total_limit = cfg_data['limit_train'] + cfg_data['limit_test']
    
    conn = mysql.connector.connect(
        host=cfg_db['host'],
        port=cfg_db['port'],
        user=os.getenv(cfg_db['user_env_var']),
        password=os.getenv(cfg_db['pass_env_var']),
        database=cfg_db['db_name']
    )
    
    query = cfg_data['query'].format(total_limit=total_limit)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df_notas = coletar_dados()
client = rg.Argilla(api_url=config['argilla']['api_url'], api_key=config['argilla']['api_key'])

settings = rg.Settings(
    guidelines=cfg_data['guidelines'],
    fields=[rg.TextField(name="text", title="Nota Clínica")],
    questions=[
        rg.SpanQuestion(
            name="span_label",
            field="text",
            labels=cfg_data['labels'],
            allow_overlapping=False
        )
    ]
)

dataset = client.datasets(name=cfg_data['name'])
if dataset is None:
    dataset = rg.Dataset(name=cfg_data['name'], settings=settings)
    dataset.create()

print("Enviando registros...")
records = [
    rg.Record(
        fields={"text": row[mapping['text']]},
        metadata={
            "id_banco": int(row[mapping['id_banco']]), 
            "data_nota": str(row[mapping['data_nota']])
        }
    )
    for _, row in df_notas.iterrows()
]

dataset.records.log(records)
print(f"Sucesso! {len(records)} registros enviados para {cfg_data['name']}.")