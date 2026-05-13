# Guia de Serviços Externos: MLflow e Argilla

Este documento detalha a infraestrutura de serviços externos necessários para o desenvolvimento, rastreamento de experimentos e curadoria de dados do projeto de Extração de Entidades (NER).

---

## 1. MLflow: Rastreamento e Gestão de Experimentos

### Utilidade e Necessidade
O **MLflow** é a ferramenta responsável pelo tracking do nosso agente de Inteligência Artificial. Toda vez que executamos um processo de extração, o MLflow registra as métricas de avaliação, os parâmetros utilizados e os arquivos de saída.

**Por que é essencial para os experimentos ABA e IBAMA?**
Nosso projeto possui diferentes configurações de arquitetura (1, 2 ou 3 nós/agents). O MLflow centraliza essas execuções, permitindo comparar de forma objetiva qual dessas configurações tem a melhor precisão e o melhor custo/benefício para a base de dados do IBAMA ou para notas clínicas ABA, colocando todas as rodadas lado a lado.

### Como iniciar o MLflow
Assumindo que você está no diretório raiz do projeto, inicie o contêiner executando:

```bash
docker compose -f servicos_externos/mlflow/docker-compose.yml up -d
```

A interface ficará disponível no seu navegador local em `http://localhost:5000`.

---

## 2. Argilla: Curadoria e Anotação de Dados

### Utilidade e Necessidade
O **Argilla** é a nossa plataforma de validação humana. Precisamos revisar o que o agente extraiu (Antecedentes, Comportamentos, Consequências, Causas, Fenômenos, etc.) e possivelmente corrigir essas anotações para criar o nosso "Padrão Ouro" (Gold Standard).

### Como iniciar o Argilla
A partir do diretório raiz do projeto, suba os serviços do Argilla:

```bash
docker compose -f servicos_externos/argilla_deploy/docker-compose.yaml up -d
```

A interface ficará disponível no navegador em `http://localhost:6900`. 

**Credenciais de Acesso Padrão:**
- **Usuário:** `argilla`
- **Senha:** `12345678`
*(Estes valores podem ser alterados no arquivo `servicos_externos/argilla_deploy/docker-compose.yaml`. Inclusive com variáveis de ambiente definidas no .env ou um arquivo de config, por simplicidade mantive com um padrão do próprio serviço.)*.

---

### Configuração e Ingestão de Dados no Argilla

A pasta `servicos_externos/argilla_deploy/` contém scripts para enviar nossos dados ao Argilla. O comportamento desses scripts é controlado pelo arquivo **`.config.yaml`**, que permite alterar mapeamentos sem mexer no código Python.

#### O Arquivo `.config.yaml`
Ele centraliza as definições:
- **Secão Argilla:** URL, API Key e Workspace de destino.
- **Secão Dataset CSV:** Nome do dataset, caminho do arquivo `.csv`, coluna de texto e labels de entidades permitidas.
- **Secão Dataset DB:** Query SQL (com controle de limite), e mapeamento de colunas (associando o ID do banco e a data da nota como metadados para rastreabilidade), pode ser necessário alterar o query a depender do objetivo ou tabela de interesse.

#### Script 1: Ingestão via CSV (`create_dataset_from_csv.py`)
Um exemplo simples utilizando um csv já preparado (dataset em `artefatos/ner_datasets/`). Ele limpa os dados nulos e envia os registros em lote. Garanta que o Argilla esteja up, execute o script python, acesse o serviço no navegador, faça o login com as credenciais e na tela inicial você verá o conjunto de dados exportado "collectaba_obs_diarias_abc" essa tarefa está definida como classificação dos trechos dos textos.

**Execução a partir da raiz:**
```bash
python servicos_externos/argilla_deploy/scripts/create_dataset_from_csv.py
```

#### Script 2: Ingestão via Banco de Dados (`create_dataset_from_db.py`)
Utilizado para extrair notas clínicas, relatórios diretamente do banco de dados e enviá-las ao Argilla. 
**Atenção:** Requer um arquivo `.env` configurado com as credenciais do banco de dados.

**Execução a partir da raiz:**
```bash
python servicos_externos/argilla_deploy/scripts/create_dataset_from_db.py
```