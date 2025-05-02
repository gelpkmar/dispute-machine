# python -m unittest test_german_retrieval.py -v
import unittest
from langchain_community.vectorstores import Chroma
from main import (
    load_embeddings,
    CHROMA_SETTINGS,
    MODEL_ID,
    MODEL_BASENAME,
    load_model,
    get_prompt_template,
    PERSIST_DIRECTORY_X,
    PERSIST_DIRECTORY_Y,
)

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Choose agent
AGENT = "X"  # or "Y"
PERSIST_DIRECTORY = PERSIST_DIRECTORY_X if AGENT == "X" else PERSIST_DIRECTORY_Y

class TestGermanRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n[INFO] Initializing test for AGENT: {AGENT}")
        cls.embeddings = load_embeddings()
        cls.llm = load_model("cuda", MODEL_ID, MODEL_BASENAME)

    def test_db_document_count(self):
        """Ensure DB is not empty and print a document"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )
        count = db._collection.count()
        print(f"[DEBUG] Anzahl der Dokumente in der DB: {count}")

        docs = db.similarity_search("Test", k=1)
        if docs:
            print(f"[DEBUG] Beispielinhalt: {docs[0].page_content[:150]}...")
        else:
            print("[WARN] Keine Dokumente beim Beispielabruf gefunden.")

        self.assertGreater(count, 0, "Die Datenbank sollte mindestens ein Dokument enthalten")

    def test_german_retriever(self):
        """Teste den Retriever mit deutschen Anfragen"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )

        queries = [
            "Welcher Preisnachlass wird angeboten?",
            "Was sind die Bedingungen für die verspätete Lieferung?",
            "Welche Lösungen werden im Vertrag vorgeschlagen?"
        ]

        for query in queries:
            docs = db.similarity_search(query, k=2)
            print(f"\n[QUERY] '{query}': {len(docs)} Dokument(e) gefunden")
            for i, doc in enumerate(docs):
                print(f"  [Doc {i+1}] {doc.page_content[:100]}... (Quelle: {doc.metadata.get('source', 'unbekannt')})")
            self.assertGreater(len(docs), 0, f"Keine Dokumente gefunden für: {query}")

    def test_german_qa_chain(self):
        """Teste die QA-Kette mit deutschen Eingaben"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )

        de_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Beantworte die folgende Frage auf Deutsch basierend auf dem gegebenen Kontext.
            Kontext: {context}
            Frage: {question}
            Antwort:
            """
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": de_prompt}
        )

        test_fragen = [
            "Welche Kompensation wird bei Verspätung angeboten?",
            "Wie hoch ist der vorgeschlagene Preisnachlass?",
            "Welche rechtlichen Grundlagen gelten in diesem Fall?"
        ]

        for frage in test_fragen:
            result = qa({"query": frage})
            print(f"\n[FRAGE] {frage}")
            print(f"[ANTWORT] {result['result']}")
            print(f"[QUELLEN] {len(result['source_documents'])}")

            self.assertTrue(result['result'].strip(), "Die Antwort sollte nicht leer sein")
            self.assertGreater(len(result['source_documents']), 0, "Es sollten Quellen zurückgegeben werden")

    def test_llm_german_capability(self):
        """Teste die Deutschfähigkeiten des LLM direkt"""
        test_fragen = [
            "Erkläre mir den Begriff 'Preisnachlass' in einfachem Deutsch.",
            "Was sind die typischen Gründe für Lieferverzögerungen?",
            "Nenne drei wichtige Punkte eines Liefervertrags."
        ]

        for frage in test_fragen:
            antwort = self.llm(f"Frage: {frage}\nAntwort:")
            print(f"\n[FRAGE] {frage}")
            print(f"[ANTWORT] {antwort}")
            self.assertTrue(antwort.strip(), "Die Antwort sollte nicht leer sein")
            self.assertGreater(len(antwort.split()), 3, "Die Antwort sollte ausführlich sein")

    def test_no_relevant_documents(self):
        """Testet, wie das System auf irrelevante Anfragen reagiert"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )
        query = "Was ist die Hauptstadt von Narnia?"
        docs = db.similarity_search(query, k=3)
        print(f"\n[IRRELEVANT] Anfrage: '{query}' - {len(docs)} Dokumente gefunden")
        self.assertIsInstance(docs, list)

    def test_document_order_by_similarity(self):
        """Prüft, ob Dokumente nach Ähnlichkeit sortiert sind"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )
        results = db.similarity_search_with_score("Preisnachlass", k=3)
        scores = [score for _, score in results]
        print("\n[SCORES] für 'Preisnachlass':", scores)
        self.assertTrue(all(scores[i] <= scores[i+1] for i in range(len(scores)-1)), "Scores sind nicht sortiert")

    def test_retriever_k_variations(self):
        """Testet verschiedene Werte für k bei der Suche"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )
        for k in [1, 5]:
            docs = db.similarity_search("Preisnachlass", k=k)
            print(f"[RETRIEVER] k={k}: {len(docs)} Dokumente gefunden")
            self.assertLessEqual(len(docs), k)

    def test_qa_repeatability(self):
        """Stellt sicher, dass QA bei wiederholten Anfragen konsistent ist (locker)"""
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS
        )

        de_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Beantworte die folgende Frage auf Deutsch basierend auf dem gegebenen Kontext.
            Kontext: {context}
            Frage: {question}
            Antwort:
            """
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": de_prompt}
        )

        frage = "Welche Kompensation wird bei Verspätung angeboten?"
        result1 = qa({"query": frage})["result"].strip()
        result2 = qa({"query": frage})["result"].strip()

        print(f"\n[REPEAT-TEST] Antwort 1: {result1}")
        print(f"[REPEAT-TEST] Antwort 2: {result2}")

        self.assertNotEqual(result1, "")
        self.assertNotEqual(result2, "")
        self.assertTrue(
            len(set(result1.split()) & set(result2.split())) > 5,
            "Antworten sollten eine signifikante inhaltliche Überlappung haben"
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
