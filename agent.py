import ingest, run_localGPT, utils

class Agent:
    def __init__(self, name, embeddings_dir, device_type, use_history, model_type, persist_dir, promptTemplate_type, agent_state):
        self.name = name  
        self.embeddings_dir = embeddings_dir  
        self.persist_dir = persist_dir
        self.promptTemplate_type = promptTemplate_type
        self.agent_state = agent_state
        # self.opening_statement = opening_statement
        self.qa = run_localGPT.retrieval_qa_pipline(device_type, use_history, self.persist_dir, promptTemplate_type=model_type)

    def ask(self, query):
        complete_query = query
        res = self.qa(complete_query)
        return res