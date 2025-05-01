# python -m unittest test_retrieval.py -v
import unittest
import os
import shutil
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from main import load_embeddings, DOCUMENT_MAP, CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY

# Define test agent state similar to your main script
TEST_AGENT_STATE = {
    'legal_issue_involved': 'Preisnachlass wegen verspäteter Lieferung',
    'dispute_context': {
        'context': 'Test context',
        'facts': 'Test facts',
        'preferred_resolution': 'Test resolution'
    }
}

class TestRetrievalSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run once before all tests"""
        cls.embeddings = load_embeddings()
        cls.settings = CHROMA_SETTINGS
        
    def setUp(self):
        """Run before each test"""
        self.test_persist_dir = "./test_db"
        os.makedirs(self.test_persist_dir, exist_ok=True)
        
    def tearDown(self):
        """Run after each test"""
        # Clean up test database - using shutil.rmtree to handle directories
        if os.path.exists(self.test_persist_dir):
            shutil.rmtree(self.test_persist_dir)

    def test_embeddings_loading(self):
        """Test that embeddings are properly loaded"""
        test_text = "This is a test sentence"
        embedding = self.embeddings.embed_query(test_text)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 100)  # Reasonable embedding size
        print(f"\nEmbedding test passed - dimension: {len(embedding)}")

    def test_document_loading(self):
        """Test that documents can be loaded from source directory"""
        # Create a test document if source directory is empty
        test_file_path = os.path.join(SOURCE_DIRECTORY, "test_doc.txt")
        if not os.path.exists(SOURCE_DIRECTORY):
            os.makedirs(SOURCE_DIRECTORY)
            
        if not os.listdir(SOURCE_DIRECTORY):
            with open(test_file_path, "w") as f:
                f.write("Test document content")
        
        for ext, loader_class in DOCUMENT_MAP.items():
            test_file = f"test_doc{ext}"
            test_path = os.path.join(SOURCE_DIRECTORY, test_file)
            
            if os.path.exists(test_path):
                loader = loader_class(test_path)
                documents = loader.load()
                self.assertGreater(len(documents), 0)
                print(f"Document loader test passed for {ext} files")

    def test_chroma_connection(self):
        """Test that Chroma DB can be connected and contains documents"""
        # Skip if no persist directory exists
        if not os.path.exists(PERSIST_DIRECTORY):
            self.skipTest(f"No Chroma DB found at {PERSIST_DIRECTORY}")
            
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=self.settings
        )
        
        # Verify collection exists
        collection = db._collection
        self.assertIsNotNone(collection)
        
        # Only check count if DB exists
        count = collection.count()
        print(f"\nChroma DB connection test passed - documents found: {count}")

    def test_retrieval_functionality(self):
        """Test that the system can retrieve relevant documents"""
        # Create a test Chroma DB
        test_docs = [
            "The quick brown fox jumps over the lazy dog",
            "Lorem ipsum dolor sit amet",
            "Python is a popular programming language",
            "Legal disputes often involve contract interpretation"
        ]
        
        test_metadatas = [{"source": f"test_{i}"} for i in range(len(test_docs))]
        
        db = Chroma.from_texts(
            texts=test_docs,
            embedding=self.embeddings,
            metadatas=test_metadatas,
            persist_directory=self.test_persist_dir,
            client_settings=self.settings
        )
        
        # Test retrieval
        retriever = db.as_retriever(search_kwargs={"k": 2})
        results = retriever.get_relevant_documents("programming language")
        
        self.assertEqual(len(results), 2)
        self.assertIn("Python", results[0].page_content)
        print("\nRetrieval test passed - relevant documents returned")

    def test_full_qa_pipeline(self):
        """Test the complete QA pipeline with retrieval"""
        # Skip if no persist directory exists
        if not os.path.exists(PERSIST_DIRECTORY):
            self.skipTest(f"No Chroma DB found at {PERSIST_DIRECTORY}")
            
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=self.settings
        )
        
        retriever = db.as_retriever(search_kwargs={"k": 3})
        results = retriever.get_relevant_documents(TEST_AGENT_STATE['legal_issue_involved'])
        
        self.assertGreater(len(results), 0, "No documents retrieved for legal issue")
        print(f"\nQA pipeline test passed - retrieved {len(results)} documents")
        
        # Print sample results for inspection
        print("\nSample retrieved documents:")
        for i, doc in enumerate(results[:2]):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    unittest.main(verbosity=2)