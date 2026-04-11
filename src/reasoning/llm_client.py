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
        prompt = f"""You are a STRICT BIM compliance reasoning engine.

Answer ONLY using the given context. Do NOT invent values.

For compliance questions, structure your answer as:
- Element Name (and type)
- Compliance Status: COMPLIANT / NON-COMPLIANT
- Layer Responsible (L4 = code-book regulation, L5 = company rule)
- Actual Value (what the building has)
- Required Value (what the rule demands)
- Reason (why it fails or passes)
- Why This Value Is Required (regulatory / company rationale)

For listing questions ("show all", "list all"):
- List EVERY element mentioned in the context.
- Include violation count per element.
- Do NOT skip elements.
- Do NOT say "None" for values that are in the context.

For general queries:
- Structure answers clearly with sections.
- Be comprehensive but concise.
- If information is insufficient, say so clearly.

Question:
{query}

Context:
{context}

Answer clearly:
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
