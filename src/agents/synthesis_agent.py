"""
Synthesis Agent - Generates medical answers from retrieved contexts

Uses GPT-4 (baseline) with plan to swap in MedGemma-27B for comparison.
Implements citation management, medical reasoning, and confidence scoring.

Architecture:
    Retrieved Contexts → Synthesis Agent (GPT-4) → Cited Medical Answer
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from llama_index.core import Document
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore, TextNode

load_dotenv()


class SynthesisAgent:
    """
    Synthesis Agent that generates medical answers from retrieved contexts.
    
    Features:
        - Citation management (tracks which sources used)
        - Medical reasoning (explicit thought process)
        - Confidence scoring (knows when to say "I don't know")
        - Model swappable (GPT-4 → MedGemma for A/B testing)
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.1,
        response_mode: str = "compact"
    ):
        """
        Initialize Synthesis Agent.
        
        Args:
            model: LLM model to use (gpt-4, gpt-3.5-turbo, or medgemma-27b)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            response_mode: Response synthesis mode
                - "compact": Concatenate contexts, single LLM call
                - "refine": Iteratively refine answer
                - "tree_summarize": Build tree of summaries
        """
        print("\n" + "="*70)
        print("SYNTHESIS AGENT INITIALIZATION")
        print("="*70)
        
        self.model_name = model
        self.temperature = temperature
        self.response_mode = response_mode
        
        # Check if using DeepSeek
        if model == "deepseek-chat":
            # Use DeepSeek API with OpenAI compatibility
            self.llm = OpenAI(
                model="deepseek-chat",
                temperature=temperature,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                api_base="https://api.deepseek.com",  # DeepSeek endpoint
                is_function_calling_model=False,
                is_chat_model=True
            )
            print(f"✓ LLM initialized: {model} (temp={temperature}) [DeepSeek API]")
        else:
            # Use OpenAI
            self.llm = OpenAI(
                model=model,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            print(f"✓ LLM initialized: {model} (temp={temperature})")
        
        # Initialize response synthesizer
        self.synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=response_mode,
            use_async=False,
            streaming=False
        )
        print(f"✓ Response synthesizer: {response_mode} mode")
        print("="*70 + "\n")
    
    def _format_context_with_citations(
        self, 
        contexts: List[Dict[str, Any]]
    ) -> tuple[List[NodeWithScore], Dict[str, str]]:
        """
        Format retrieved contexts with citation markers.
        
        Args:
            contexts: List of context dicts from router
                [{"text": "...", "score": 0.8, "metadata": {...}}, ...]
        
        Returns:
            (nodes_with_scores, citation_map)
        """
        nodes = []
        citation_map = {}
        
        for i, ctx in enumerate(contexts):
            citation_id = f"[{i+1}]"
            
            # Extract source info
            metadata = ctx.get('metadata', {})
            source = metadata.get('source', 'unknown')
            
            # Build citation reference
            if source == 'pubmedqa':
                doc_id = metadata.get('_id', 'unknown')
                citation_ref = f"PubMedQA:{doc_id}"
            elif source == 'pubmed_api':
                pmid = metadata.get('pmid', 'unknown')
                title = metadata.get('title', 'Unknown Title')
                year = metadata.get('year', 'N/A')
                citation_ref = f"PMID:{pmid} ({year}): {title}"
            else:
                citation_ref = f"Source:{source}"
            
            citation_map[citation_id] = citation_ref
            
            # Create node with citation marker
            text_with_citation = f"{citation_id} {ctx['text']}"
            
            node = NodeWithScore(
                node=TextNode(
                    text=text_with_citation,
                    metadata=metadata
                ),
                score=ctx.get('score', 1.0)
            )
            nodes.append(node)
        
        return nodes, citation_map
    
    def _build_medical_prompt(
        self,
        query: str,
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build specialized medical prompt with instructions.
        
        Args:
            query: User's medical question
            query_analysis: Optional query analysis from QueryAnalyzer
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a medical AI assistant. Answer the following medical question based ONLY on the provided context.

QUESTION: {query}
"""
        
        if query_analysis:
            entities = query_analysis.get('entities', {})
            query_type = query_analysis.get('query_type', 'unknown')
            
            prompt += f"""
QUERY ANALYSIS:
- Query Type: {query_type}
- Medical Entities Identified:
  • Conditions: {', '.join(entities.get('conditions', [])) or 'None'}
  • Drugs: {', '.join(entities.get('drugs', [])) or 'None'}
  • Procedures: {', '.join(entities.get('procedures', [])) or 'None'}
  • Symptoms: {', '.join(entities.get('symptoms', [])) or 'None'}
"""
        
        prompt += """
INSTRUCTIONS:
1. Base your answer ONLY on the provided context (marked with [1], [2], etc.)
2. Cite specific sources using [number] notation
3. If the context doesn't contain enough information, explicitly state this
4. Use clear medical terminology but explain complex concepts
5. Structure your answer logically (what, why, how)
6. Highlight key points about safety, efficacy, or clinical relevance
7. If multiple sources agree/disagree, note this

ANSWER:"""
        
        return prompt
    
    def _extract_citations_used(self, answer: str) -> List[str]:
        """
        Extract which citation numbers were used in the answer.
        
        Args:
            answer: Generated answer text
        
        Returns:
            List of citation IDs used (e.g., ["[1]", "[3]", "[5]"])
        """
        # Find all [number] patterns
        citations = re.findall(r'\[\d+\]', answer)
        return list(set(citations))  # Unique citations
    
    def _estimate_confidence(
        self,
        answer: str,
        contexts_used: int,
        total_contexts: int
    ) -> float:
        """
        Estimate confidence in the generated answer.
        
        Factors:
            - How many contexts were cited
            - Presence of uncertainty phrases
            - Answer length and completeness
        
        Args:
            answer: Generated answer
            contexts_used: Number of contexts cited
            total_contexts: Total contexts available
        
        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Citation coverage
        if total_contexts > 0:
            citation_ratio = contexts_used / total_contexts
            confidence += 0.3 * citation_ratio
        
        # Factor 2: Uncertainty phrases (lower confidence)
        uncertainty_phrases = [
            "not enough information",
            "unclear from the context",
            "cannot determine",
            "insufficient evidence",
            "more research needed",
            "may or may not"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase.lower() in answer.lower():
                confidence -= 0.15
                break
        
        # Factor 3: Answer completeness (length heuristic)
        if len(answer) < 100:
            confidence -= 0.1  # Very short answer
        elif len(answer) > 300:
            confidence += 0.1  # Detailed answer
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def synthesize(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate medical answer from retrieved contexts.
        
        Args:
            query: User's medical question
            contexts: Retrieved contexts from router
                [{"text": "...", "score": 0.8, "metadata": {...}}, ...]
            query_analysis: Optional query analysis with entities
        
        Returns:
            {
                "answer": str,              # Generated answer
                "citations_used": List[str], # Citations actually used
                "citation_map": Dict,        # Map citation IDs to sources
                "confidence": float,         # Confidence score (0-1)
                "num_contexts": int,         # Total contexts provided
                "model": str,                # Model used for generation
                "reasoning": str             # Brief reasoning summary
            }
        """
        print(f"\n{'='*70}")
        print(f"SYNTHESIZING ANSWER")
        print(f"{'='*70}")
        print(f"Query: {query}")
        print(f"Contexts: {len(contexts)} retrieved")
        
        if not contexts:
            return {
                "answer": "I don't have enough information to answer this question. No relevant contexts were retrieved.",
                "citations_used": [],
                "citation_map": {},
                "confidence": 0.0,
                "num_contexts": 0,
                "model": self.model_name,
                "reasoning": "No contexts available"
            }
        
        # Step 1: Format contexts with citations
        print(f"\n[Step 1] Formatting contexts with citation markers...")
        nodes, citation_map = self._format_context_with_citations(contexts)
        print(f"  → {len(nodes)} contexts formatted")
        print(f"  → Citations: {list(citation_map.keys())}")
        
        # Step 2: Build medical prompt
        print(f"\n[Step 2] Building medical prompt...")
        prompt = self._build_medical_prompt(query, query_analysis)
        
        # Step 3: Build full prompt with contexts
        print(f"\n[Step 3] Building full prompt with contexts...")
        full_prompt = prompt + "\n\nCONTEXT:\n"
        for node in nodes:
            full_prompt += f"\n{node.node.text}\n"
        
        # Step 4: Generate answer with direct LLM call
        print(f"\n[Step 4] Generating answer with {self.model_name}...")
        try:
            from llama_index.core.llms import ChatMessage
            
            messages = [
                ChatMessage(role="system", content="You are a medical AI assistant that provides evidence-based answers with proper citations."),
                ChatMessage(role="user", content=full_prompt)
            ]
            
            response = self.llm.chat(messages)
            answer = response.message.content
            print(f"  → Generated {len(answer)} characters")
            
        except Exception as e:
            print(f"  ✗ Synthesis failed: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "citations_used": [],
                "citation_map": citation_map,
                "confidence": 0.0,
                "num_contexts": len(contexts),
                "model": self.model_name,
                "reasoning": f"Synthesis error: {str(e)}"
            }
        
        # Step 4: Extract citations used
        print(f"\n[Step 5] Analyzing citations...")
        citations_used = self._extract_citations_used(answer)
        print(f"  → {len(citations_used)} citations used: {citations_used}")
        
        # Step 5: Estimate confidence
        print(f"\n[Step 6] Computing confidence...")
        confidence = self._estimate_confidence(
            answer=answer,
            contexts_used=len(citations_used),
            total_contexts=len(contexts)
        )
        print(f"  → Confidence: {confidence:.2f}")
        
        # Build reasoning summary
        reasoning = f"Used {len(citations_used)}/{len(contexts)} contexts. "
        if confidence >= 0.7:
            reasoning += "High confidence based on multiple supporting sources."
        elif confidence >= 0.5:
            reasoning += "Moderate confidence, some uncertainty present."
        else:
            reasoning += "Low confidence, limited context or high uncertainty."
        
        print(f"\n{'='*70}\n")
        
        return {
            "answer": answer,
            "citations_used": citations_used,
            "citation_map": citation_map,
            "confidence": confidence,
            "num_contexts": len(contexts),
            "model": self.model_name,
            "reasoning": reasoning
        }


def test_synthesis_agent():
    """Test Synthesis Agent with sample contexts."""
    print("\n" + "="*80)
    print("TESTING SYNTHESIS AGENT")
    print("="*80)
    
    # Initialize agent
    agent = SynthesisAgent(
        model="gpt-4",
        temperature=0.1,
        response_mode="compact"
    )
    
    # Test case 1: Medical question with contexts
    print("\n" + "="*80)
    print("TEST CASE 1: Effects of Metformin")
    print("="*80)
    
    query = "What are the effects of metformin on diabetes?"
    
    # Mock contexts (as if from router)
    contexts = [
        {
            "text": "Metformin is a first-line medication for type 2 diabetes. It works by reducing glucose production in the liver and improving insulin sensitivity in peripheral tissues. Clinical trials show it reduces HbA1c by 1-2% and may aid in weight loss.",
            "score": 0.85,
            "metadata": {
                "source": "pubmedqa",
                "_id": "doc_12345",
                "question": "How does metformin work?"
            }
        },
        {
            "text": "Long-term metformin use is associated with vitamin B12 deficiency in some patients. Regular monitoring is recommended. Common side effects include gastrointestinal upset, which often improves with continued use.",
            "score": 0.72,
            "metadata": {
                "source": "pubmedqa",
                "_id": "doc_67890",
                "question": "What are metformin side effects?"
            }
        },
        {
            "text": "Recent studies show metformin may have cardiovascular benefits beyond glucose control, potentially reducing risk of heart disease in diabetic patients.",
            "score": 0.68,
            "metadata": {
                "source": "pubmed_api",
                "pmid": "41234567",
                "title": "Metformin and Cardiovascular Outcomes",
                "year": "2025"
            }
        }
    ]
    
    # Query analysis (mock)
    query_analysis = {
        "entities": {
            "drugs": ["metformin"],
            "conditions": ["diabetes"],
            "procedures": [],
            "symptoms": []
        },
        "query_type": "drug_information",
        "is_temporal": False
    }
    
    # Synthesize answer
    result = agent.synthesize(
        query=query,
        contexts=contexts,
        query_analysis=query_analysis
    )
    
    # Display results
    print("\n" + "="*80)
    print("SYNTHESIS RESULTS")
    print("="*80)
    print(f"\nQUESTION: {query}\n")
    print(f"ANSWER:\n{result['answer']}\n")
    print(f"\nCITATIONS USED: {result['citations_used']}")
    print(f"\nSOURCE MAPPING:")
    for cit_id, source in result['citation_map'].items():
        used = "✓" if cit_id in result['citations_used'] else "✗"
        print(f"  {used} {cit_id}: {source}")
    print(f"\nCONFIDENCE: {result['confidence']:.2f}")
    print(f"REASONING: {result['reasoning']}")
    print(f"MODEL: {result['model']}")
    
    print("\n" + "="*80)
    print("SYNTHESIS AGENT TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_synthesis_agent()
