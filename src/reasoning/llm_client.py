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

        prompt = f"""You are an expert BIM (Building Information Modelling) compliance assistant.
Your job is to answer questions about building elements, their properties, and regulatory compliance.

STRICT RULES — you MUST follow these exactly:
1. NEVER invent, guess or assume numerical values. Only state values that are explicitly present in the provided context.
2. If a property or element is NOT in the context, say clearly: "This property/element is not found in the project data."
3. For compliance questions: state COMPLIANT or NON-COMPLIANT with the exact values and the rule reference.
4. For non-compliant items: always state (a) what value was found, (b) what value is required, (c) the gap, (d) a practical suggestion.
5. Structure your answer with clear sections. Use bullet points for lists.
6. If the context says MISSING_PROPERTY — tell the user the property does not exist in L1/L2 data AND list what properties ARE available.
7. Do not repeat the question back — go straight to the answer.

Question: {query}

Context (from BIM project data):
{context}

Answer:"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        if self.client_type == "openai":
            # Self-hosted Colab vLLM — no API token limits, use generous context
            response = self.client.chat.completions.create(
                model=self.custom_model,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,   # no cost, no rate limit — use full context
            )
        else:
            # Groq free-tier fallback — keep within 6000 TPM limit
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.3,
                max_tokens=1500,   # safe for Groq free tier
            )

        result = response.choices[0].message.content

        # Cache the result
        self._response_cache[cache_key] = result

        return result