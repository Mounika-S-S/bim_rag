def build_prompt(query, context):

    prompt = f"""
You are a building regulation compliance expert.

User query:
{query}

Retrieved context:
{context}

Using the context, determine whether the design complies.

Return STRICT JSON:

{{
"compliant": true or false,
"reason": "...",
"suggestion": "..."
}}
"""

    return prompt