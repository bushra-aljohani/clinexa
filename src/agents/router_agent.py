"""
Router Agent - Routes queries to appropriate retrieval method

Uses LlamaIndex RouterRetriever with PydanticSingleSelector to dynamically
decide between Static RAG and Dynamic PubMed API based on query characteristics.

Architecture:
    Query → Query Analyzer → Router Agent → [Static RAG | Dynamic API] → Results
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from llama_index.core.retrievers import RouterRetriever, BaseRetriever
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import RetrieverTool
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from sentence_transformers import SentenceTransformer

# Handle imports for both direct execution and module import
try:
    from agents.query_analyzer import QueryAnalyzer
    from agents.static_rag import StaticRAG
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.query_analyzer import QueryAnalyzer
    from agents.static_rag import StaticRAG

load_dotenv()


class SemanticScorer:
    """
    Shared semantic similarity scorer using sentence-transformers.
    
    Uses 'all-MiniLM-L6-v2' model (80MB, cached locally).
    Computes cosine similarity between query and documents.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            print("[SemanticScorer] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("  ✓ Model loaded (cached for future use)")
    
    def compute_similarity(self, query: str, document: str) -> float:
        """
        Compute semantic similarity between query and document.
        
        Args:
            query: User query
            document: Document text
            
        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        try:
            # Encode both texts
            query_emb = self._model.encode(query, convert_to_numpy=True)
            doc_emb = self._model.encode(document, convert_to_numpy=True)
            
            # Cosine similarity
            similarity = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            
            return float(similarity)
        
        except Exception as e:
            print(f"[WARN] Similarity computation failed: {e}")
            return 0.5  # Neutral score on error


class DynamicPubMedRetriever(BaseRetriever):
    """
    Dynamic PubMed retriever using E-utilities API.
    
    Searches live PubMed database for recent articles (2024-2026).
    Free API with rate limits: 3 req/sec (no key) or 10 req/sec (with key).
    """
    
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None, relevance_threshold: float = 0.3):
        """
        Initialize Dynamic PubMed retriever.
        
        Args:
            email: Email for PubMed API (required)
            api_key: PubMed API key (optional, increases rate limits to 10 req/sec)
            relevance_threshold: Minimum semantic similarity score to include result (0-1)
        """
        import requests
        self.requests = requests
        
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email or os.getenv("PUBMED_EMAIL", "medagent@example.com")
        self.api_key = api_key or os.getenv("PUBMED_API_KEY")
        self.rate_limit = 10 if self.api_key else 3  # req/sec
        self.relevance_threshold = relevance_threshold
        
        # Initialize semantic scorer (shared singleton)
        self.scorer = SemanticScorer()
        
        # Initialize BaseRetriever
        super().__init__()
        
        print(f"[DynamicPubMedRetriever] Initialized with E-utilities API")
        print(f"  Email: {self.email}")
        print(f"  API Key: {'✓ Present' if self.api_key else '✗ Not provided'}")
        print(f"  Rate Limit: {self.rate_limit} req/sec")
        print(f"  Relevance Threshold: {self.relevance_threshold}")
    
    def _search_pubmed(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search PubMed and return PMIDs.
        
        Uses esearch.fcgi endpoint with date filtering (2024-2026).
        
        Args:
            query: Search query
            max_results: Maximum number of PMIDs to return
            
        Returns:
            List of PMIDs
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
            "tool": "MedAgent-HYBRID-RAG",
            "mindate": "2024/01/01",
            "maxdate": "2026/12/31",
            "datetype": "pdat"  # Publication date
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = self.requests.get(
                f"{self.base_url}esearch.fcgi",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            return pmids
        
        except Exception as e:
            print(f"[ERROR] PubMed search failed: {e}")
            return []
    
    def _fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch article details from PMIDs.
        
        Uses efetch.fcgi endpoint with abstract retrieval.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dicts with title, abstract, pmid, year
        """
        if not pmids:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
            "tool": "MedAgent-HYBRID-RAG"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = self.requests.get(
                f"{self.base_url}efetch.fcgi",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            articles = []
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract PMID
                    pmid = article.find(".//PMID").text
                    
                    # Extract title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    # Extract abstract
                    abstract_texts = article.findall(".//AbstractText")
                    if abstract_texts:
                        abstract = " ".join([at.text or "" for at in abstract_texts])
                    else:
                        abstract = "No abstract available."
                    
                    # Extract year
                    year_elem = article.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None else "Unknown"
                    
                    # Extract journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else "Unknown"
                    
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "journal": journal
                    })
                
                except Exception as e:
                    print(f"[WARN] Failed to parse article: {e}")
                    continue
            
            return articles
        
        except Exception as e:
            print(f"[ERROR] PubMed fetch failed: {e}")
            return []
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Required by BaseRetriever. Routes to retrieve() method.
        
        Args:
            query_bundle: Query bundle from LlamaIndex
            
        Returns:
            List of NodeWithScore objects
        """
        return self.retrieve(query_bundle.query_str, top_k=10)
    
    def retrieve(self, query_str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve from PubMed API.

        Process:
        1. Search PubMed with esearch (get PMIDs)
        2. Fetch article details with efetch (get abstracts)
        3. Convert to NodeWithScore objects

        Args:
            query_str: Search query (str or QueryBundle)
            top_k: Number of results to return

        Returns:
            List of NodeWithScore objects
        """
        # Handle QueryBundle objects from LlamaIndex
        if hasattr(query_str, 'query_str'):
            query_text = query_str.query_str
        else:
            query_text = str(query_str)
        
        print(f"[DynamicPubMedRetriever] Searching PubMed for: '{query_text}'")
        
        # Step 1: Search for PMIDs
        pmids = self._search_pubmed(query_text, max_results=top_k)
        
        if not pmids:
            print(f"  → No results found")
            return []
        
        print(f"  → Found {len(pmids)} PMIDs: {pmids[:3]}...")
        
        # Step 2: Fetch abstracts
        articles = self._fetch_abstracts(pmids)
        
        if not articles:
            print(f"  → Failed to fetch article details")
            return []
        
        print(f"  → Retrieved {len(articles)} articles")
        
        # Step 3: Compute semantic similarity scores
        print(f"  → Computing semantic similarity scores...")
        scored_articles = []
        for article in articles:
            # Combine title + abstract for scoring
            doc_text = f"{article['title']} {article['abstract']}"
            
            # Compute semantic similarity
            score = self.scorer.compute_similarity(query_text, doc_text)
            
            # Filter by relevance threshold
            if score >= self.relevance_threshold:
                scored_articles.append((article, score))
        
        # Sort by score (descending)
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  → {len(scored_articles)}/{len(articles)} articles passed relevance threshold ({self.relevance_threshold})")
        
        # Step 4: Convert to NodeWithScore
        nodes_with_scores = []
        for article, score in scored_articles[:top_k]:
            # Combine title + abstract
            text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
            
            # Create node
            node = NodeWithScore(
                node=TextNode(
                    text=text,
                    metadata={
                        "source": "pubmed_api",
                        "pmid": article['pmid'],
                        "title": article['title'],
                        "year": article['year'],
                        "journal": article['journal'],
                        "retrieval_method": "dynamic_api",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/",
                        "semantic_score": score
                    }
                ),
                score=score  # Use semantic similarity score
            )
            nodes_with_scores.append(node)
        
        return nodes_with_scores


class CustomStaticRetriever(BaseRetriever):
    """
    Wrapper around StaticRAG to make it compatible with LlamaIndex RouterRetriever.
    Converts StaticRAG.retrieve() output to NodeWithScore objects.
    Adds semantic scoring for consistency with Dynamic retriever.
    """
    
    def __init__(self, static_rag: StaticRAG, relevance_threshold: float = 0.3):
        """
        Initialize wrapper around StaticRAG.
        
        Args:
            static_rag: Initialized StaticRAG instance
            relevance_threshold: Minimum semantic similarity score (0-1)
        """
        self.static_rag = static_rag
        self.relevance_threshold = relevance_threshold
        self.scorer = SemanticScorer()
        
        # Initialize BaseRetriever
        super().__init__()
        
        print(f"[CustomStaticRetriever] Initialized with {len(static_rag.vector_index.docstore.docs)} docs")
        print(f"  Relevance Threshold: {self.relevance_threshold}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Required by BaseRetriever. Routes to retrieve() method.
        
        Args:
            query_bundle: Query bundle from LlamaIndex
            
        Returns:
            List of NodeWithScore objects
        """
        return self.retrieve(query_bundle.query_str, top_k=10)
    
    def retrieve(self, query_str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve from Static RAG with semantic score validation.
        
        Args:
            query_str: Search query (str or QueryBundle)
            top_k: Number of results to return
            
        Returns:
            List of NodeWithScore objects
        """
        # Handle QueryBundle objects from LlamaIndex
        if hasattr(query_str, 'query_str'):
            query_text = query_str.query_str
        else:
            query_text = str(query_str)
        
        # Get results from Static RAG (gets more than top_k for filtering)
        results = self.static_rag.retrieve(query_text, top_k=top_k * 2, method="hybrid")
        
        # Recompute semantic scores for consistency
        scored_results = []
        for result in results:
            # Compute semantic similarity
            semantic_score = self.scorer.compute_similarity(query_text, result['text'])
            
            # Filter by relevance threshold
            if semantic_score >= self.relevance_threshold:
                # Blend hybrid score with semantic score (favor semantic)
                blended_score = 0.7 * semantic_score + 0.3 * result['score']
                result['semantic_score'] = semantic_score
                result['hybrid_score'] = result['score']
                result['score'] = blended_score
                scored_results.append(result)
        
        # Sort by blended score
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert to NodeWithScore objects
        nodes_with_scores = []
        for result in scored_results[:top_k]:
            # Add semantic score to metadata
            metadata = result.get('metadata', {})
            metadata['semantic_score'] = result['semantic_score']
            metadata['hybrid_score'] = result['hybrid_score']
            
            node = NodeWithScore(
                node=TextNode(
                    text=result['text'],
                    metadata=metadata
                ),
                score=result['score']
            )
            nodes_with_scores.append(node)
        
        return nodes_with_scores


class RouterAgent:
    """
    Router Agent that intelligently selects between Static RAG and Dynamic API.
    
    Uses LlamaIndex RouterRetriever with PydanticSingleSelector to dynamically
    choose the best retrieval method based on query characteristics.
    
    Routing Logic:
        - Temporal queries ("latest", "recent", dates) → Dynamic PubMed API
        - General medical knowledge queries → Static RAG
        - Complex queries → Can route to both (if using MultiSelector)
    """
    
    def __init__(
        self,
        static_rag: Optional[StaticRAG] = None,
        dynamic_retriever: Optional[DynamicPubMedRetriever] = None,
        llm_model: str = "gpt-4",
        use_query_analyzer: bool = True
    ):
        """
        Initialize Router Agent.
        
        Args:
            static_rag: StaticRAG instance (creates new if None)
            dynamic_retriever: DynamicPubMedRetriever instance (creates new if None)
            llm_model: LLM model for routing decisions
            use_query_analyzer: Whether to use Query Analyzer for pre-routing analysis
        """
        print("\n" + "="*70)
        print("ROUTER AGENT INITIALIZATION")
        print("="*70)
        
        # Initialize LLM for routing
        self.llm = OpenAI(model=llm_model)
        print(f"✓ LLM initialized: {llm_model}")
        
        # Initialize Query Analyzer (optional pre-routing)
        self.use_query_analyzer = use_query_analyzer
        if use_query_analyzer:
            self.query_analyzer = QueryAnalyzer()
            print("✓ Query Analyzer enabled")
        
        # Initialize Static RAG
        if static_rag is None:
            print("\n[1/2] Initializing Static RAG...")
            self.static_rag = StaticRAG(limit=1000, force_rebuild=False)
        else:
            self.static_rag = static_rag
        print("✓ Static RAG ready")
        
        # Initialize Dynamic retriever
        if dynamic_retriever is None:
            print("\n[2/2] Initializing Dynamic PubMed retriever...")
            self.dynamic_retriever = DynamicPubMedRetriever()
        else:
            self.dynamic_retriever = dynamic_retriever
        print("✓ Dynamic retriever ready")
        
        # Wrap retrievers for RouterRetriever compatibility
        self.static_retriever = CustomStaticRetriever(self.static_rag)
        
        # Create RetrieverTools with detailed descriptions
        self.static_tool = RetrieverTool.from_defaults(
            retriever=self.static_retriever,
            description=(
                "Use this for GENERAL medical knowledge queries about established "
                "medical concepts, drugs, procedures, and diseases. This retriever "
                "searches a curated PubMedQA knowledge base with 800 indexed documents. "
                "Best for: definitions, mechanisms, standard treatments, disease information. "
                "DO NOT use for queries about recent research, latest studies, or specific date ranges."
            )
        )
        
        self.dynamic_tool = RetrieverTool.from_defaults(
            retriever=self.dynamic_retriever,
            description=(
                "Use this for RECENT or TEMPORAL medical queries that require the latest "
                "research, recent studies, or information from specific time periods. "
                "This retriever searches live PubMed API for papers published 2024-2026. "
                "Best for: 'latest research on...', 'recent studies about...', "
                "'what's new in...', queries with dates or temporal keywords. "
                "Use when query mentions: latest, recent, new, current, 2024, 2025, 2026."
            )
        )
        
        # Create RouterRetriever with PydanticSingleSelector
        self.router = RouterRetriever(
            selector=PydanticSingleSelector.from_defaults(llm=self.llm),
            retriever_tools=[
                self.static_tool,
                self.dynamic_tool
            ]
        )
        
        print("\n✓ Router configured with 2 retrieval methods")
        print("="*70)
        print()
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to extract entities and characteristics.
        
        Args:
            query: User query
            
        Returns:
            Query analysis dict with entities, query type, temporal indicators
        """
        if not self.use_query_analyzer:
            return {"query": query}
        
        analysis = self.query_analyzer.analyze(query)
        return analysis
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Route query to appropriate retriever and get results.
        
        Args:
            query: User query
            top_k: Number of results to return
            return_metadata: Whether to include routing metadata
            
        Returns:
            Dict with:
                - results: List of retrieved documents
                - routing_decision: Which retriever was selected
                - query_analysis: Query analysis (if enabled)
                - retrieval_method: static_rag or dynamic_api
        """
        print(f"\n{'='*70}")
        print(f"ROUTING QUERY")
        print(f"{'='*70}")
        print(f"Query: {query}")
        print()
        
        # Optional: Pre-analyze query
        query_analysis = None
        if self.use_query_analyzer:
            print("[Step 1] Analyzing query...")
            query_analysis = self.analyze_query(query)
            print(f"  - Entities: {query_analysis.get('entities', {})}")
            print(f"  - Query Type: {query_analysis.get('query_type', 'unknown')}")
            print(f"  - Is Temporal: {query_analysis.get('is_temporal', False)}")
            print()
        
        # Route and retrieve
        print("[Step 2] Routing to retriever...")
        nodes_with_scores = self.router.retrieve(query)
        
        # Determine which retriever was used
        retrieval_method = "unknown"
        if nodes_with_scores:
            first_node_source = nodes_with_scores[0].node.metadata.get("source", "")
            if first_node_source == "pubmedqa":
                retrieval_method = "static_rag"
                print(f"  → Routed to: STATIC RAG")
            elif first_node_source == "pubmed_api":
                retrieval_method = "dynamic_api"
                print(f"  → Routed to: DYNAMIC API")
        
        print(f"  → Retrieved {len(nodes_with_scores)} results")
        print()
        
        # Format results
        results = []
        for i, node_with_score in enumerate(nodes_with_scores[:top_k]):
            result = {
                "rank": i + 1,
                "text": node_with_score.node.text,
                "score": node_with_score.score,
                "metadata": node_with_score.node.metadata
            }
            results.append(result)
        
        # Build response
        response = {
            "query": query,
            "results": results,
            "retrieval_method": retrieval_method,
            "num_results": len(results)
        }
        
        if return_metadata:
            response["query_analysis"] = query_analysis
            response["routing_decision"] = retrieval_method
        
        print(f"{'='*70}\n")
        return response
    
    def close(self):
        """Close connections."""
        if hasattr(self, 'static_rag'):
            self.static_rag.close()


def test_router_agent():
    """Test Router Agent with different query types."""
    print("\n" + "="*80)
    print("TESTING ROUTER AGENT")
    print("="*80)
    
    # Initialize router
    router = RouterAgent(
        llm_model="gpt-4",
        use_query_analyzer=True
    )
    
    # Test queries
    test_queries = [
        {
            "query": "What are the effects of metformin on diabetes?",
            "expected_route": "static_rag",
            "reason": "General medical knowledge query"
        },
        {
            "query": "What is the latest research on COVID-19 treatments in 2025?",
            "expected_route": "dynamic_api",
            "reason": "Temporal query with date"
        },
        {
            "query": "Recent studies about cancer immunotherapy",
            "expected_route": "dynamic_api",
            "reason": "Contains temporal keyword 'recent'"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Route: {test_case['expected_route']}")
        print(f"Reason: {test_case['reason']}")
        print()
        
        # Retrieve
        response = router.retrieve(test_case['query'], top_k=3)
        
        # Print results
        print(f"\n[RESULTS]")
        print(f"Routed to: {response['retrieval_method'].upper()}")
        print(f"Match expected: {'✓' if response['retrieval_method'] == test_case['expected_route'] else '✗'}")
        print()
        
        for result in response['results']:
            print(f"Rank {result['rank']} (Score: {result['score']:.3f})")
            print(f"  {result['text'][:150]}...")
            print()
    
    router.close()
    
    print("\n" + "="*80)
    print("ROUTER AGENT TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_router_agent()
