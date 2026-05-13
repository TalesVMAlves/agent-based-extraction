import pandas as pd
import re
import ast
import json

def parse_argilla_responses(response_str):
    if pd.isna(response_str): return []
    matches = re.findall(r"\{'end': \d+, 'label': '[^']+', 'start': \d+\}", response_str)
    parsed = []
    for m in matches:
        try: parsed.append(ast.literal_eval(m))
        except: pass
    return parsed

def localizar_span_ancorado(texto_completo, snippet_tagueado, tag):
    if not snippet_tagueado or not texto_completo: return None
    
    # 1. Tenta achar a marcação completa com a tag
    pattern = rf"(.*?)<{tag}>(.*?)</{tag}>(.*)"
    match = re.search(pattern, snippet_tagueado, re.DOTALL | re.IGNORECASE)
    
    if match:
        prefixo, entidade, sufixo = match.groups()
        snippet_limpo = prefixo + entidade + sufixo 
        
        inicio_snippet = texto_completo.find(snippet_limpo)
        if inicio_snippet != -1:
            start_idx = inicio_snippet + len(prefixo)
            return {"start": start_idx, "end": start_idx + len(entidade), "text": entidade}
        
    #     inicio_entidade = texto_completo.find(entidade)
    #     if inicio_entidade != -1:
    #         return {"start": inicio_entidade, "end": inicio_entidade + len(entidade), "text": entidade}
            
    # limpo = snippet_tagueado.replace(f"<{tag}>", "").replace(f"</{tag}>", "").strip()
    # inicio = texto_completo.find(limpo)
    # if inicio != -1 and len(limpo) > 0:
    #     return {"start": inicio, "end": inicio + len(limpo), "text": limpo}
        
    return None

def extrair_spans_do_agente(text, extracted_data, entities_names):
    """Extrai os spans iterando dinamicamente sobre a lista de entidades da configuração."""
    res = []
    if not extracted_data: return res
    
    for ent in entities_names:
        tag = ent.upper()
        snippet = extracted_data.get(f"{ent}_snippet")
        
        span = localizar_span_ancorado(text, snippet, tag)
        if span:
            span["label"] = tag
            res.append(span)
            
    res_unicos = []
    vistos = set()
    for r in res:
        tupla_id = (r["start"], r["end"], r["label"])
        if tupla_id not in vistos:
            vistos.add(tupla_id)
            res_unicos.append(r)
            
    res_unicos.sort(key=lambda x: x["start"])
    return res_unicos

def preparar_dados_para_juiz(df_preds, entities_names):
    """Prepara os dicionários Gabarito e Predição de forma agnóstica de domínio."""
    dados_juiz = []
    
    for idx, row in df_preds.iterrows():
        texto_original = row["text"]
        
        true_ents = json.loads(row["true_entities"]) if isinstance(row["true_entities"], str) else row["true_entities"]
        pred_ents = json.loads(row["pred_entities"]) if isinstance(row["pred_entities"], str) else row["pred_entities"]
        true_dict = {e.upper(): [] for e in entities_names}
        pred_dict = {e.upper(): [] for e in entities_names}
        
        for t in true_ents: 
            label = t["label"].upper()
            if label in true_dict:
                true_dict[label].append(texto_original[t["start"]:t["end"]])
                
        for p in pred_ents: 
            label = p["label"].upper()
            if label in pred_dict:
                pred_dict[label].append(p["text"])
        
        dados_juiz.append({
            "doc_id": idx,
            "relato": texto_original,
            "gabarito": true_dict,
            "predicao": pred_dict
        })
        
    return pd.DataFrame(dados_juiz)