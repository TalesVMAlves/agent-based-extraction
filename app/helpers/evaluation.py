import json
from tqdm import tqdm
import math
import os
import mlflow
import pandas as pd
from app.helpers.utils import preparar_dados_para_juiz
from app.helpers.classes import create_judge_model

def calcular_iou(span_true, span_pred):
    start_max = max(span_true['start'], span_pred['start'])
    end_min = min(span_true['end'], span_pred['end'])
    intersecao = max(0, end_min - start_max)
    
    uniao = max(span_true['end'], span_pred['end']) - min(span_true['start'], span_pred['start'])
    return intersecao / uniao if uniao > 0 else 0

def evaluate_relaxed_from_dataframe(df_preds, entities_names, iou_threshold=0.3):
    print(f"Avaliação | IoU >= {iou_threshold}")
    
    labels = [e.upper() for e in entities_names]
    metrics = {label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in labels}
    
    for _, row in df_preds.iterrows():
        true_ents = json.loads(row["true_entities"]) if isinstance(row["true_entities"], str) else row["true_entities"]
        pred_ents = json.loads(row["pred_entities"]) if isinstance(row["pred_entities"], str) else row["pred_entities"]
        true_spans = [{"start": t["start"], "end": t["end"], "label": t["label"].upper()} for t in true_ents]
        pred_spans = [{"start": p["start"], "end": p["end"], "label": p["label"].upper()} for p in pred_ents]
        
        for label in labels:
            trues_label = [t for t in true_spans if t["label"] == label]
            preds_label = [p for p in pred_spans if p["label"] == label]
            
            if len(trues_label) == 0 and len(preds_label) == 0:
                metrics[label]["TN"] += 1
                continue
            
            matched_trues = set()
            matched_preds = set()
            
            for i, p_span in enumerate(preds_label):
                melhor_iou = 0
                melhor_true_idx = -1
                
                for j, t_span in enumerate(trues_label):
                    iou = calcular_iou(t_span, p_span)
                    if iou > melhor_iou:
                        melhor_iou = iou
                        melhor_true_idx = j
                
                if melhor_iou >= iou_threshold:
                    metrics[label]["TP"] += 1
                    matched_trues.add(melhor_true_idx)
                    matched_preds.add(i)
                else:
                    metrics[label]["FP"] += 1
                    
            metrics[label]["FN"] += len(trues_label) - len(matched_trues)

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    final_metrics = {}
    
    for label, counts in metrics.items():
        tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) 
        mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
        
        prefix = label[:4].lower() 
        final_metrics[f"iou_{prefix}_precision"] = precision
        final_metrics[f"iou_{prefix}_recall"] = recall
        final_metrics[f"iou_{prefix}_f1"] = f1
        final_metrics[f"iou_{prefix}_mcc"] = mcc

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    micro_mcc_den = math.sqrt((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn))
    micro_mcc = ((total_tp * total_tn) - (total_fp * total_fn)) / micro_mcc_den if micro_mcc_den > 0 else 0
    
    final_metrics["iou_global_precision"] = micro_p
    final_metrics["iou_global_recall"] = micro_r
    final_metrics["iou_global_f1"] = micro_f1
    final_metrics["iou_global_mcc"] = micro_mcc
    
    return final_metrics

async def executar_juiz_semantico(df_juiz_dados, juiz_chain, entities_names):
    resultados = []
    schema_dinamico = create_judge_model(entities_names)
    
    for _, row in tqdm(df_juiz_dados.iterrows(), total=len(df_juiz_dados), desc="Avaliando com LLM Judge"):
        try:
            resposta_bruta = await juiz_chain.ainvoke({
                "relato": row["relato"],
                "gabarito": json.dumps(row["gabarito"], ensure_ascii=False),
                "predicao": json.dumps(row["predicao"], ensure_ascii=False)
            })
            
            if isinstance(resposta_bruta, dict):
                resposta = schema_dinamico(**resposta_bruta)
            else:
                resposta = resposta_bruta
                
            res_dict = {
                "doc_id": row["doc_id"],
                "nota_final": resposta.nota_final,
                "analise_juiz": resposta.analise_geral
            }
            
            for ent in entities_names:
                ent_key = ent.lower()
                prefix = ent[:4].lower()
                
                metricas_obj = getattr(resposta, ent_key, None)
                
                if metricas_obj:
                    res_dict[f"judge_{prefix}_cobertura"] = metricas_obj.cobertura_semantica
                    res_dict[f"judge_{prefix}_pureza"] = metricas_obj.pureza_semantica
                else:
                    res_dict[f"judge_{prefix}_cobertura"] = None
                    res_dict[f"judge_{prefix}_pureza"] = None
                    
            resultados.append(res_dict)
            
        except Exception as e:
            erro_dict = {
                "doc_id": row["doc_id"],
                "nota_final": None,
                "analise_juiz": str(e)
            }
            for ent in entities_names:
                prefix = ent[:4].lower()
                erro_dict[f"judge_{prefix}_cobertura"] = None
                erro_dict[f"judge_{prefix}_pureza"] = None
                
            resultados.append(erro_dict)
            
    df_resultados_juiz = pd.DataFrame(resultados)
    
    metrics = {col: df_resultados_juiz[col].mean(skipna=True) for col in df_resultados_juiz.columns if col.startswith("judge_") and "nota" not in col}
    cobertura_cols = [col for col in df_resultados_juiz.columns if col.endswith("_cobertura")]
    pureza_cols = [col for col in df_resultados_juiz.columns if col.endswith("_pureza")]
    if cobertura_cols:
        metrics["judge_global_cobertura"] = df_resultados_juiz[cobertura_cols].mean(skipna=True).mean()
    if pureza_cols:
        metrics["judge_global_pureza"] = df_resultados_juiz[pureza_cols].mean(skipna=True).mean()
    metrics["nota_final"] = df_resultados_juiz["nota_final"].mean(skipna=True)
    
    return df_resultados_juiz, metrics

async def run_unified_evaluation(df_preds, config, juiz_chain, caminho_imagem=None):
    run_name = config['experiment']['mlflow_run']
    entities_names = []
    for task in config['domain']['tasks']:
        entities_names.extend(task['entities'])
    print(f"Iniciando Job: {run_name} ({config['experiment']['architecture']})")
    
    with mlflow.start_run(run_name=run_name):
        os.makedirs("outputs", exist_ok=True)

        provider = config['llm']['provider']
        flat_params = {
            "architecture": config['experiment']['architecture'],
            "domain": config['experiment']['name'].split('_')[0],
            "model": config['llm']['models'][provider]['model_name'],
            "num_tasks": len(config['domain']['tasks'])
        }
        with open(f"outputs/experiments/config_{run_name}.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        mlflow.log_artifact(f"outputs/experiments/config_{run_name}.json", artifact_path="configs")
        mlflow.log_params(flat_params)

        if caminho_imagem and os.path.exists(caminho_imagem):
            mlflow.log_artifact(caminho_imagem, artifact_path="arquitetura_grafo")
        
        print("\nCalculando métricas IoU...")
        iou_thresh = config['experiment'].get("iou_threshold", 0.3)
        metricas_iou = evaluate_relaxed_from_dataframe(df_preds, entities_names, iou_threshold=iou_thresh)
        mlflow.log_metrics(metricas_iou)

        print("\nCalculando métricas do Juiz...")
        df_juiz_preparado = preparar_dados_para_juiz(df_preds, entities_names)
        df_res_juiz, metricas_juiz = await executar_juiz_semantico(df_juiz_preparado, juiz_chain, entities_names)
        mlflow.log_metrics(metricas_juiz)

        caminho_pred = "outputs/agent_predictions_gen.csv"
        caminho_juiz = "outputs/auditoria_juiz_gen.csv"
        
        df_preds.to_csv(caminho_pred, index=False, encoding='utf-8')
        df_res_juiz.to_csv(caminho_juiz, index=False, encoding='utf-8')
        
        mlflow.log_artifact(caminho_pred, artifact_path="datasets_resultados")
        mlflow.log_artifact(caminho_juiz, artifact_path="datasets_resultados")
        
        print("\nJob concluído e registrado no MLflow!")