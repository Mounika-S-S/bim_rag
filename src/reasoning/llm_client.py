from groq import Groq
import os
import hashlib
import json
from functools import lru_cache

class LLMClient:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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
You are an expert BIM compliance assistant. Answer questions based on the provided context from BIM layers (L1-L5).

Guidelines:
- Use the context to provide accurate, detailed answers.
- For compliance questions, identify violations clearly.
- For cost-cutting or planning, suggest practical, safe recommendations.
- Structure answers clearly with sections if needed.
- Be comprehensive but concise.
- If information is insufficient, say so.

Question: {query}

Context:
{context}

Answer:
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
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