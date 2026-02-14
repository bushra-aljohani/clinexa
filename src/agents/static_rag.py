"""
Static RAG with HYBRID Retrieval - PubMedQA
============================================

Uses PubMedQA for static retrieval to compare with dynamic PubMed API.
Features: Train/test split, context extraction, persistence, hybrid search.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

try:
    from llama_index.retrievers.bm25 import BM25Retriever
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

load_dotenv()


class StaticRAG:
    """HYBRID RAG with PubMedQA - for comparison with dynamic PubMed API."""
    
    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        db_name: str = "HealthcareDataset",
        collection_name: str = "sampled_pubmedqa",
        vector_weight: float = 0.6,
        sparse_weight: float = 0.4,
        limit: int = 1000,
        train_split: float = 0.8,
        persist_dir: str = "./rag_storage",
        force_rebuild: bool = False
    ):
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        self.db_name = db_name
        self.collection_name = collection_name
        self.vector_weight = vector_weight
        self.sparse_weight = sparse_weight
        self.train_split = train_split
        self.persist_dir = Path(persist_dir)
        self.force_rebuild = force_rebuild
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Static RAG (PubMedQA)")
        print(f"  DB: {db_name}/{collection_name}")
        print(f"  Split: {train_split*100:.0f}% train")
        
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        vector_path = self.persist_dir / "vector_index"
        test_path = self.persist_dir / "test_docs.pkl"
        
        if not force_rebuild and vector_path.exists() and test_path.exists():
            print(f"  📂 Loading from disk...")
            self._load_from_disk()
        else:
            print(f"  🔨 Building indexes...")
            self.train_docs, self.test_docs = self._load_documents(limit)
            self._build_vector_index()
            self._build_bm25_index()
            self._save_to_disk()
        
        print(f"✓ Ready: {len(self.train_docs)} train, {len(self.test_docs)} test")
    
    
    def _load_documents(self, limit: int):
        """Load PubMedQA documents and split."""
        docs = list(self.collection.find({}, {
            "_id": 1, "question": 1, "context": 1,
            "long_answer": 1, "final_decision": 1
        }).limit(limit))
        
        if not docs:
            raise ValueError("No documents in MongoDB!")
        
        print(f"    Loaded {len(docs)} docs")
        
        import random
        random.seed(42)
        random.shuffle(docs)
        split_idx = int(len(docs) * self.train_split)
        
        # Extract context text from dict if needed
        def extract_context(doc):
            context = doc.get("context", "")
            if isinstance(context, dict):
                # PubMedQA context is dict with "contexts" key
                return " ".join(context.get("contexts", []))
            return str(context) if context else ""
        
        # Train docs
        train_documents = []
        for doc in docs[:split_idx]:
            text = extract_context(doc)
            if not text or len(text) < 50:
                continue
            
            train_documents.append(Document(
                text=text,
                metadata={
                    "_id": str(doc["_id"]),
                    "question": doc.get("question", ""),
                    "source": "pubmedqa"
                }
            ))
        
        # Test docs
        test_documents = [{
            "_id": str(doc["_id"]),
            "question": doc.get("question", ""),
            "context": extract_context(doc),
            "long_answer": doc.get("long_answer", ""),
            "final_decision": doc.get("final_decision", "")
        } for doc in docs[split_idx:]]
        
        print(f"    Split: {len(train_documents)} train, {len(test_documents)} test")
        return train_documents, test_documents
    
    
    def _build_vector_index(self):
        print(f"    Building vector index...")
        self.vector_index = VectorStoreIndex.from_documents(
            self.train_docs,
            show_progress=True
        )
        self.vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=10
        )
        print(f"    ✓ Vector index built")
    
    
    def _build_bm25_index(self):
        if not HAS_BM25:
            print(f"    ⚠️ BM25 not available, will rebuild each time")
            self.bm25_retriever = None
            return
        
        print(f"    Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=list(self.vector_index.docstore.docs.values()),
            similarity_top_k=10
        )
        print(f"    ✓ BM25 index built")
    
    
    def _save_to_disk(self):
        print(f"    💾 Persisting...")
        
        vector_path = self.persist_dir / "vector_index"
        self.vector_index.storage_context.persist(persist_dir=str(vector_path))
        print(f"      ✓ Vector → {vector_path}")
        
        # Note: BM25 rebuilds from vector index (can't pickle Cython objects)
        
        test_path = self.persist_dir / "test_docs.pkl"
        with open(test_path, 'wb') as f:
            pickle.dump(self.test_docs, f)
        print(f"      ✓ Test docs → {test_path}")
    
    
    def _load_from_disk(self):
        vector_path = self.persist_dir / "vector_index"
        storage_context = StorageContext.from_defaults(persist_dir=str(vector_path))
        self.vector_index = load_index_from_storage(storage_context)
        self.vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=10
        )
        self.train_docs = list(self.vector_index.docstore.docs.values())
        print(f"      ✓ Vector ({len(self.train_docs)} docs)")
        
        # Rebuild BM25 from loaded vector index
        if HAS_BM25:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.train_docs,
                similarity_top_k=10
            )
            print(f"      ✓ BM25 rebuilt")
        else:
            self.bm25_retriever = None
        
        test_path = self.persist_dir / "test_docs.pkl"
        with open(test_path, 'rb') as f:
            self.test_docs = pickle.load(f)
        print(f"      ✓ Test docs ({len(self.test_docs)} docs)")
    
    
    def retrieve_dense(self, query: str, top_k: int = 5):
        nodes = self.vector_retriever.retrieve(query)
        return [{
            'text': node.get_content(),
            'metadata': {**node.metadata, 'retrieval_method': 'dense'},
            'score': node.score if hasattr(node, 'score') else 0.0
        } for node in nodes[:top_k]]
    
    
    def retrieve_sparse(self, query: str, top_k: int = 5):
        if not self.bm25_retriever:
            return []
        
        nodes = self.bm25_retriever.retrieve(query)
        return [{
            'text': node.get_content(),
            'metadata': {**node.metadata, 'retrieval_method': 'sparse'},
            'score': node.score if hasattr(node, 'score') else 0.0
        } for node in nodes[:top_k]]
    
    
    def retrieve_hybrid(self, query: str, top_k: int = 10):
        dense = self.retrieve_dense(query, top_k)
        sparse = self.retrieve_sparse(query, top_k)
        
        def normalize(results):
            if not results:
                return results
            scores = [r['score'] for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                for r in results:
                    r['score'] = 1.0
            else:
                for r in results:
                    r['score'] = (r['score'] - min_s) / (max_s - min_s)
            return results
        
        dense = normalize(dense)
        sparse = normalize(sparse)
        
        doc_scores = {}
        for r in dense:
            doc_id = r['metadata'].get('_id', id(r))
            doc_scores[doc_id] = {'result': r, 'dense': r['score'], 'sparse': 0.0}
        
        for r in sparse:
            doc_id = r['metadata'].get('_id', id(r))
            if doc_id in doc_scores:
                doc_scores[doc_id]['sparse'] = r['score']
            else:
                doc_scores[doc_id] = {'result': r, 'dense': 0.0, 'sparse': r['score']}
        
        hybrid = []
        for doc_id, data in doc_scores.items():
            score = self.vector_weight * data['dense'] + self.sparse_weight * data['sparse']
            result = data['result'].copy()
            result['score'] = score
            result['metadata']['retrieval_method'] = 'hybrid'
            result['metadata']['dense_score'] = data['dense']
            result['metadata']['sparse_score'] = data['sparse']
            hybrid.append(result)
        
        hybrid.sort(key=lambda x: x['score'], reverse=True)
        return hybrid[:top_k]
    
    
    def retrieve(self, query: str, top_k: int = 5, method: str = "hybrid"):
        if method == "dense":
            return self.retrieve_dense(query, top_k)
        elif method == "sparse":
            return self.retrieve_sparse(query, top_k)
        elif method == "hybrid":
            return self.retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    
    def get_test_questions(self, num: int = 5):
        import random
        if num >= len(self.test_docs):
            return self.test_docs
        return random.sample(self.test_docs, num)
    
    
    def close(self):
        if self.client:
            self.client.close()


def test_static_rag():
    print("=" * 80)
    print("STATIC RAG TEST - PubMedQA HYBRID Retrieval")
    print("=" * 80)
    
    rag = StaticRAG(limit=1000, force_rebuild=True)  # Use all 1000 docs
    
    test_questions = rag.get_test_questions(num=3)
    
    print(f"\n📝 Testing {len(test_questions)} held-out questions...\n")
    
    for i, test_item in enumerate(test_questions, 1):
        query = test_item['question']
        answer = test_item.get('final_decision', 'N/A')
        
        print(f"\n{'=' * 80}")
        print(f"Test {i}/{len(test_questions)}")
        print('=' * 80)
        print(f"Q: {query}")
        print(f"A: {answer}")
        
        results = rag.retrieve(query, top_k=3, method="hybrid")
        
        print(f"\n🔍 HYBRID (0.6 vector + 0.4 BM25):")
        for j, r in enumerate(results, 1):
            print(f"\n  [{j}] Score: {r['score']:.4f} (Dense: {r['metadata'].get('dense_score', 0):.3f} | BM25: {r['metadata'].get('sparse_score', 0):.3f})")
            print(f"      {r['text'][:200]}...")
    
    rag.close()
    
    print(f"\n{'=' * 80}")
    print("✓ Completed! Next run loads from disk (instant)")
    print("=" * 80)


if __name__ == "__main__":
    test_static_rag()
