"""
Query Analysis Agent
====================

Extracts medical entities and classifies queries using DeepSeek/GPT-4.

Features:
- Medical entity extraction (conditions, drugs, procedures, symptoms)
- Query classification (type, specialty, complexity)
- Temporal detection (latest, recent, 2024)
- Structured JSON output with validation
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class QueryAnalyzer:
    """
    Analyzes medical queries to extract entities and classify intent.
    """
    
    def __init__(
        self, 
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Query Analyzer with specified model.
        
        Args:
            model: Model name (default: deepseek-chat)
            api_key: API key (reads from env if not provided)
            base_url: API base URL (for DeepSeek, Claude, etc.)
        """
        self.model = model
        
        # Configure OpenAI client based on model
        if model.startswith("deepseek"):
            self.client = OpenAI(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=base_url or "https://api.deepseek.com"
            )
        elif model.startswith("gpt-"):
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        else:
            # Generic OpenAI-compatible endpoint
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
    
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze a medical query and extract structured information.
        
        Args:
            query: User's medical question
            
        Returns:
            Dictionary with extracted entities and classifications
        """
        
        system_prompt = """You are a medical query analysis expert. Your task is to analyze medical questions and extract structured information.

Extract and classify the following:

1. **Medical Entities:**
   - conditions: diseases, disorders, syndromes (e.g., "type 2 diabetes", "hypertension")
   - drugs: medications, treatments (e.g., "metformin", "insulin")
   - procedures: medical procedures, tests (e.g., "colonoscopy", "MRI")
   - symptoms: patient symptoms (e.g., "chest pain", "fatigue")

2. **Query Classification:**
   - specialty: medical specialty (e.g., "cardiology", "oncology", "endocrinology")
   - query_type: one of [clinical_question, treatment_options, diagnosis, drug_information, research, clinical_trials, general_info]
   - complexity: one of [simple, moderate, complex]

3. **Temporal Analysis:**
   - is_temporal: true if query asks about recent/latest/new information
   - temporal_keywords: list of temporal words found (e.g., ["latest", "recent", "2024"])

Return ONLY valid JSON matching this structure:
{
  "entities": {
    "conditions": [],
    "drugs": [],
    "procedures": [],
    "symptoms": []
  },
  "specialty": "string",
  "query_type": "string",
  "is_temporal": boolean,
  "temporal_keywords": [],
  "complexity": "string"
}"""

        user_prompt = f"""Analyze this medical query:

"{query}"

Extract entities and classify the query. Return valid JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            
            # Validate and add metadata
            validated = self._validate_analysis(analysis, query)
            
            return validated
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print(f"Raw response: {content}")
            # Return fallback structure
            return self._fallback_analysis(query)
        
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return self._fallback_analysis(query)
    
    
    def _validate_analysis(self, analysis: Dict, query: str) -> Dict:
        """
        Validate and enrich analysis output.
        """
        # Ensure all required fields exist
        validated = {
            "entities": {
                "conditions": analysis.get("entities", {}).get("conditions", []),
                "drugs": analysis.get("entities", {}).get("drugs", []),
                "procedures": analysis.get("entities", {}).get("procedures", []),
                "symptoms": analysis.get("entities", {}).get("symptoms", [])
            },
            "specialty": analysis.get("specialty", "general_medicine"),
            "query_type": analysis.get("query_type", "general_info"),
            "is_temporal": analysis.get("is_temporal", False),
            "temporal_keywords": analysis.get("temporal_keywords", []),
            "complexity": analysis.get("complexity", "moderate"),
            "metadata": {
                "original_query": query,
                "model_used": self.model,
                "timestamp": datetime.now().isoformat(),
                "total_entities": sum(len(v) for v in analysis.get("entities", {}).values())
            }
        }
        
        return validated
    
    
    def _fallback_analysis(self, query: str) -> Dict:
        """
        Return basic analysis when LLM fails.
        """
        # Simple keyword-based detection
        query_lower = query.lower()
        
        temporal_words = ["latest", "recent", "new", "current", "updated", "2024", "2025", "2026"]
        found_temporal = [word for word in temporal_words if word in query_lower]
        
        return {
            "entities": {
                "conditions": [],
                "drugs": [],
                "procedures": [],
                "symptoms": []
            },
            "specialty": "general_medicine",
            "query_type": "general_info",
            "is_temporal": len(found_temporal) > 0,
            "temporal_keywords": found_temporal,
            "complexity": "moderate",
            "metadata": {
                "original_query": query,
                "model_used": self.model,
                "timestamp": datetime.now().isoformat(),
                "total_entities": 0,
                "fallback": True
            }
        }
    
    
    def analyze_batch(self, queries: List[str]) -> List[Dict]:
        """
        Analyze multiple queries in batch.
        
        Args:
            queries: List of medical questions
            
        Returns:
            List of analysis results
        """
        results = []
        for i, query in enumerate(queries, 1):
            print(f"Analyzing query {i}/{len(queries)}...")
            result = self.analyze(query)
            results.append(result)
        
        return results


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_query_analyzer():
    """
    Test Query Analyzer with sample medical questions.
    """
    print("=" * 80)
    print("QUERY ANALYZER TEST")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = QueryAnalyzer(model="deepseek-chat")
    print(f"\n✓ Initialized with model: {analyzer.model}")
    
    # Test queries
    test_queries = [
        "What are the latest treatments for type 2 diabetes?",
        "Can metformin cause lactic acidosis?",
        "What is the standard protocol for colonoscopy screening?",
        "Recent advances in immunotherapy for lung cancer",
        "Side effects of statins"
    ]
    
    print(f"\n📝 Testing with {len(test_queries)} queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {query}")
        print('─' * 80)
        
        result = analyzer.analyze(query)
        
        # Display results
        print(f"\n🔍 Entities:")
        for entity_type, entities in result['entities'].items():
            if entities:
                print(f"  - {entity_type}: {', '.join(entities)}")
        
        print(f"\n📊 Classification:")
        print(f"  - Specialty: {result['specialty']}")
        print(f"  - Type: {result['query_type']}")
        print(f"  - Complexity: {result['complexity']}")
        
        print(f"\n⏰ Temporal:")
        print(f"  - Is temporal: {result['is_temporal']}")
        if result['temporal_keywords']:
            print(f"  - Keywords: {', '.join(result['temporal_keywords'])}")
        
        print(f"\n📦 Metadata:")
        print(f"  - Total entities: {result['metadata']['total_entities']}")
        print(f"  - Model: {result['metadata']['model_used']}")
    
    print(f"\n{'=' * 80}")
    print("✓ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_query_analyzer()
