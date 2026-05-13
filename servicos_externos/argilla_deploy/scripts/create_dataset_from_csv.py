import yaml
import pandas as pd
import argilla as rg

with open(".config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cfg_argilla = config['argilla']
cfg_csv = config['dataset_csv']

client = rg.Argilla(
    api_url=cfg_argilla['api_url'], 
    api_key=cfg_argilla['api_key']
)

df = pd.read_csv(cfg_csv['file_path']).dropna(subset=[cfg_csv['text_column']])

settings = rg.Settings(
    guidelines=cfg_csv['guidelines'],
    fields=[
        rg.TextField(name="text", title="Texto para Análise")
    ],
    questions=[
        rg.SpanQuestion(
            name="span_label",
            field="text",
            labels=cfg_csv['labels'],
            title="Selecione as entidades:",
            allow_overlapping=False
        )
    ]
)

dataset = client.datasets(name=cfg_csv['name'])
if dataset is None:
    print(f"Criando novo dataset: {cfg_csv['name']}")
    dataset = rg.Dataset(name=cfg_csv['name'], settings=settings)
    dataset.create()

print("Preparando registros...")
records = [
    rg.Record(fields={"text": str(row[cfg_csv['text_column']]).strip()})
    for _, row in df.iterrows()
]

dataset.records.log(records)
print(f"Sucesso! {len(records)} registros enviados para o Argilla.")