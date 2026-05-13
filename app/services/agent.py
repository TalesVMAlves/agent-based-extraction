import os
import yaml
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from app.helpers.classes import AgentState, create_task_model, create_judge_model
class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.architecture = config['experiment']['architecture']
        self.tasks_config = config['domain']['tasks']
        self.entities_names = []
        for task in self.tasks_config:
            self.entities_names.extend(task['entities'])
        
        llm_provider = config['llm']['provider']
        llm_settings = config['llm']['models'][llm_provider]
        
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=llm_settings['model_name'], 
                temperature=llm_settings['temperature'],
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
        elif llm_provider == "aws":
            self.llm = ChatBedrockConverse(
                model=llm_settings['model_name'],
                region_name=llm_settings['region'],
                temperature=llm_settings['temperature']
            )
        else:
            raise ValueError(f"Provider '{llm_provider}' não é suportado.")
        
        self.graph = self._build_graph()
        self._setup_judge()

    def _setup_judge(self):
        with open(self.config['domain']['judge']['prompt_path'], 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        prompt = ChatPromptTemplate.from_template(prompt_text)
        schema_dinamico = create_judge_model(self.entities_names)
        
        self.juiz_chain = prompt | self.llm.with_structured_output(schema_dinamico)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        first_node = True
        previous_node = None
        
        for task_dict in self.tasks_config:
            node_name = task_dict['name']
            builder.add_node(node_name, self._create_task_node(task_dict))
            if first_node:
                builder.set_entry_point(node_name)
                first_node = False
            else:
                builder.add_edge(previous_node, node_name)
            previous_node = node_name
            
        builder.add_edge(previous_node, END)
        return builder.compile()

    def _create_task_node(self, task_dict):
        async def node_func(state: AgentState):
            task_name = task_dict['name']
            entities_to_extract = task_dict['entities']
            
            with open(task_dict['prompt_path'], 'r', encoding='utf-8') as f:
                system_prompt_text = f.read()
                
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt_text),
                ("human", "Relato: '{text}'\nContexto do Evento prévio: {contexto}")
            ])
            
            schema = create_task_model(task_name, entities_to_extract)
            chain = prompt | self.llm.with_structured_output(schema)
            current_data = state.get("extracted_data", {})
            erros_atuais = state.get("errors", [])
            try:
                result = await chain.ainvoke({
                    "text": state["observation"],
                    "contexto": str(current_data)
                })
                if isinstance(result, dict):
                    current_data.update(result)
                else:
                    current_data.update(result.model_dump())
                    
            except Exception as e:
                erro_msg = f"Erro crítico no nó '{task_name}': {str(e)}"
                print(f"\n {erro_msg}") 
                erros_atuais.append(erro_msg)
                
            return {"extracted_data": current_data, "errors": erros_atuais}
        
        return node_func