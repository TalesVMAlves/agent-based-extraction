# Extração da Tríplice Contingência (ABC) via Agentes LLM

## Visão Geral do Produto
Este projeto propõe uma arquitetura baseada em LLMs para transformar observações diárias e relatos clínicos não estruturados em inteligência acionável. O sistema extrai a Tríplice Contingência (Antecedente -> Comportamento -> Consequência) de cada relato de sessão, permitindo a construção de Grafos de Conhecimento que ajudam os terapeutas a visualizar padrões de gatilho, auditar a fidelidade do tratamento e avaliar a eficácia das intervenções aplicadas.

---

## O Problema de Negócio e Limitações Técnicas
Relatos de sessão em ABA são narrativas complexas que frequentemente aglutinam a demanda do terapeuta, a ação do paciente e o manejo corretivo num mesmo parágrafo. Quando exigimos que um modelo de IA tradicional leia um texto e extraia todas essas informações simultaneamente, ele sofre de sobrecarga cognitiva. Isso resulta em dois problemas principais:

* **Confusão de Papéis Lógicos:** A IA frequentemente inverte as classes, por exemplo, colocando o manejo do terapeuta no campo do antecedente, ou confundindo a recusa do paciente com o gatilho da crise.
* **Vazamento de Fronteira (Boundary Bleed):** Devido à verbosidade inerente dos LLMs, o modelo extrai parágrafos inteiros em vez de focar apenas no gatilho isolado ou no manejo específico, arruinando a estruturação e a padronização dos dados clínicos.

---

## A Nossa Solução: Pipeline Multiagente Sequencial
Para resolver a sobrecarga cognitiva e as falhas generativas, adotamos a Decomposição de Tarefas utilizando o framework LangGraph. O nosso sistema roteia o raciocínio da IA em passos cognitivos isolados:

* **Passo 1 (Comportamento):** O fluxo inicia com um agente focado exclusivamente em encontrar a ação alvo de mudança, seja ela ativa (ex: "se jogou no chão") ou passiva/de oposição (ex: "recusa em realizar a demanda").
* **Passo 2 (Antecedente):** Usando o Comportamento como âncora no tempo, o modelo olha regressivamente para o texto e busca a demanda, transição ou mudança ambiental que ocorreu imediatamente antes e engatilhou a ação.
* **Passo 3 (Consequência):** O modelo então olha progressivamente para o texto e busca o manejo contingente do terapeuta ou a resposta ambiental aplicada imediatamente após o comportamento (ex: "ajuda física total", "redirecionamento").

**Inovação Técnica (Contextual Anchoring):** Para garantir que a IA não perca a rastreabilidade em textos longos, ela é forçada a retornar snippets (pequenos trechos de 5 a 10 palavras) do relato original com tags estruturadas (como `<Comportamento>`). Isso cria uma assinatura única para o evento e assegura que a métrica clínica pertence de fato àquele contexto.

---

## Como Avaliamos o Sucesso (LLM-as-a-Judge)
Métricas tradicionais de software (correspondência exata de tokens) falham ao avaliar IA Generativa, pois penalizam o modelo por usar variações linguísticas. Para garantir o rigor clínico dos dados extraídos, desenvolvemos um juiz automatizado (LLM-as-a-Judge) que avalia os resultados contra o padrão-ouro humano através de duas dimensões semânticas:

* **Cobertura Semântica (Recall):** O modelo capturou o significado central da contingência clínica sem perder informações essenciais?
* **Pureza Semântica (Precision):** O modelo respeitou rigorosamente as fronteiras das classes (ex: não incluiu o verbo de recusa dentro da chave do antecedente)?

---

## Resultados e Impacto Alcançado
O nosso estudo comprovou que a orquestração multiagente é superior às abordagens monolíticas de extração:

* Ao isolar o foco atencional do modelo, a Pureza Semântica na extração do Comportamento saltou de 83% para 95% em modelos de ponta, mitigando drasticamente as alucinações e os vazamentos de fronteira.
---

## Entrega de Valor: Grafos de Conhecimento e Análise em Dupla Camada
A principal entrega de valor para a coordenação clínica é a conversão dos dados extraídos em Grafos de Conhecimento. Utilizando um Node Registry, o sistema agrupa extrações similares em macrocategorias funcionais, resultando em diagramas de fluxo de contingência (Alluvial / Sankey) que suportam uma análise de dupla camada:

* **Camada Macroscópica (Descoberta de Padrões):** O supervisor pode visualizar tendências sistêmicas, descobrindo estatisticamente quais Antecedentes engatilham comportamentos específicos com maior frequência para um paciente.
* **Camada Microscópica (Rastreio e Auditoria):** A preservação do texto original permite um zoom-in clínico. O analista pode auditar a eficácia das intervenções (Efficacy Tracking), verificando se o manejo (Consequência) da equipe de aplicadores tem sido efetivo em regular o comportamento ou se está reforçando-o inadvertidamente.