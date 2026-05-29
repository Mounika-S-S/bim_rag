from groq import Groq
import os
import hashlib
import json
from functools import lru_cache

class LLMClient:

    def __init__(self):
        self.custom_url = os.getenv("CUSTOM_LLM_URL", "").strip()
        self.model_name = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant").strip()
        
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        
        if self.custom_url:
            # Custom vLLM / Text Generation Inference endpoint
            api_key = api_key or "dummy-key"
            self.client = Groq(api_key=api_key, base_url=self.custom_url)
        else:
            if not api_key:
                raise ValueError("GROQ_API_KEY is missing. Add it to .env and restart the backend.")
            self.client = Groq(api_key=api_key)
            
        # Simple in-memory cache for responses
        self._response_cache = {}

    def _get_cache_key(self, query, context):
        """Generate cache key from query and context"""
        content = f"{query}|{context}"
        return hashlib.md5(content.encode()).hexdigest()

    @lru_cache(maxsize=100)  # Cache up to 100 responses
    def reason(self, query, context):

        # Check cache first
        cache_key = self._get_cache_key(query, context)
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        prompt = f"""
You are a STRICT BIM compliance reasoning engine. Answer questions based ONLY on the provided context (L1-L5). Do NOT invent values.

For compliance questions, structure your answer as:
- Element Name (and type)
- Compliance Status: COMPLIANT / NON-COMPLIANT
- Layer Responsible (L4 = code-book regulation, L5 = company rule)
- Actual Value (what the building has)
- Required Value (what the rule demands)
- Reason (why it fails or passes)

For listing questions ("show all", "list all"):
- List EVERY element mentioned in the context.
- Include violation count per element.

For general queries, cost-cutting, or planning:
- Suggest practical, safe recommendations based on context.
- Structure answers clearly with sections.
- Be comprehensive but concise.
- If information is insufficient, say so clearly instead of guessing.

Question:
{query}

Context:
{context}

Answer clearly:
"""
       

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800  # Increased for detailed responses
        )

        result = response.choices[0].message.content

        # Cache the result
        self._response_cache[cache_key] = result

        return result
