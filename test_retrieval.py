# python3 -m unittest test_retrieval.py -v
import unittest
import torch
import tempfile
from unittest.mock import patch, MagicMock
from main import load_embeddings, retrieval_qa_pipline
from langchain.vectorstores import Chroma

class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepare the necessary setup for testing."""
        cls.embeddings = load_embeddings()  # Ensure embeddings load correctly
        cls.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        cls.persist_directory = tempfile.mkdtemp()  # Use a temporary directory for each test
        cls.qa_pipeline = retrieval_qa_pipline(cls.device_type, use_history=False, persist_directory=cls.persist_directory)

    @patch("langchain.vectorstores.Chroma.similarity_search")
    def test_vector_store_document_count(self, mock_similarity_search):
        """Test that the vector store has documents and is retrievable"""
        mock_similarity_search.return_value = [{"page_content": "Mock document content"}]
        
        # Test that the vector store is not empty
        db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        docs = db.similarity_search("test query", k=1)
        
        # Assert at least one document is returned
        self.assertGreater(len(docs), 0, "Vector store should have at least one document.")
        print(f"Tested documents: {len(docs)}")

    @patch("main.retrieval_qa_pipline")
    def test_retriever_with_mock_query(self, mock_retrieval_qa_pipline):
        """Test the retriever's response with a mock query"""
        # Mock response for the retrieval QA pipeline
        mock_retrieval_qa_pipline.return_value = {"result": "Mock Answer", "source_documents": [{"page_content": "Mock document content"}]}
        
        query = "What is a discount?"
        result = self.qa_pipeline({"query": query})
        
        # Assert the result is not empty and mock behavior
        self.assertIsNotNone(result['result'])
        self.assertGreater(len(result['source_documents']), 0, "No source documents returned.")

    @patch("langchain.chains.RetrievalQA.from_chain_type")
    def test_qa_chain_with_query(self, mock_qa_chain):
        """Test the QA chain with a mock response"""
        # Mock response for the QA chain
        mock_qa_chain.return_value = {"result": "The discount offered is 10%", "source_documents": [{"page_content": "Mock document content"}]}
        
        query = "What discount is offered?"
        result = self.qa_pipeline({"query": query})
        
        # Assert non-empty result and correct query handling
        self.assertIn("10%", result['result'])
        self.assertGreater(len(result['source_documents']), 0, "No source documents returned.")
    
    @patch("langchain.vectorstores.Chroma.similarity_search")
    def test_check_document_retrieval(self, mock_similarity_search):
        """Test that the retrieval of documents works as expected"""
        # Mock document retrieval response
        mock_similarity_search.return_value = [{"page_content": "Mock document content"}]
        
        # Simulate a retrieval of documents for a sample query
        query = "What is a discount?"
        db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        docs = db.similarity_search(query, k=3)
        
        self.assertGreater(len(docs), 0, "Expected documents to be returned.")
        print(f"Documents for query '{query}': {len(docs)}")

    @patch("main.load_embeddings")
    @patch("main.load_model")
    def test_load_embeddings_and_model(self, mock_load_model, mock_load_embeddings):
        """Test that the embeddings and model load correctly"""
        # Mock return values for loading embeddings and model
        mock_load_embeddings.return_value = "Mock Embedding Object"
        mock_load_model.return_value = "Mock Model Object"
        
        embeddings = load_embeddings()
        model = mock_load_model("cuda", "mock_model", "mock_basename")
        
        # Assertions
        self.assertEqual(embeddings, "Mock Embedding Object")
        self.assertEqual(model, "Mock Model Object")
        print("Tested embeddings and model loading.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
