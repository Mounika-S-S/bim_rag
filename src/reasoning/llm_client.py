from groq import Groq
from openai import OpenAI
import os
import hashlib
import json
from functools import lru_cache

class LLMClient:

    def __init__(self):
        self.custom_url = os.getenv("CUSTOM_LLM_URL")
        self.custom_model = os.getenv("LLM_MODEL_NAME", "llama3.1-bim-rag-lora")

        if self.custom_url:
            # Connect to a self-hosted GCP vLLM Endpoint (which uses OpenAI compatible API)
            self.client_type = "openai"
            self.client = OpenAI(
                base_url=self.custom_url,
                api_key=os.getenv("CUSTOM_LLM_API_KEY", "sk-no-key-required") 
            )
        else:
            # Fallback to Groq API
            self.client_type = "groq"
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

        messages = [
            {"role": "user", "content": prompt}
        ]

        if self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.custom_model,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
        else:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )

        result = response.choices[0].message.content

        # Cache the result
        self._response_cache[cache_key] = result

        return result