# python -m unittest test_source_retrieval.py -v
import unittest
import os
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma  # Updated import
from main import (
    load_embeddings,
    CHROMA_SETTINGS,
    PERSIST_DIRECTORY_X,
    PERSIST_DIRECTORY_Y,
    MODEL_ID,
    MODEL_BASENAME,
    load_model,
    get_prompt_template
)
from langchain.chains import RetrievalQA

# Choose agent
AGENT = "X"  # or "Y"
PERSIST_DIRECTORY = PERSIST_DIRECTORY_X if AGENT == "X" else PERSIST_DIRECTORY_Y

class TestSourceRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embeddings = load_embeddings()
        cls.llm = load_model("cuda", MODEL_ID, MODEL_BASENAME)
        
    def test_retriever_in_isolation(self):
        """Test if retriever finds documents without QA chain"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )
        
        query = "Preisnachlass wegen verspäteter Lieferung"
        docs = db.similarity_search(query, k=3)
        
        print(f"\nRetriever found {len(docs)} documents directly:")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        self.assertGreater(len(docs), 0)

    def test_qa_chain_sources(self):
        """Test if QA chain preserves sources"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )
        
        retriever = db.as_retriever(search_kwargs={"k": 3})
        prompt, memory = get_prompt_template(promptTemplate_type="llama")
        
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = qa({"query": "Preisnachlass wegen verspäteter Lieferung"})
        
        print("\nQA Chain Results:")
        print(f"Answer: {result['result']}")
        print(f"Sources returned: {len(result['source_documents'])}")
        
        self.assertGreater(len(result['source_documents']), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)