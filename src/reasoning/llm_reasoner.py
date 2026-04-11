from src.reasoning.llm_client import LLMClient


class LLMReasoner:

    def __init__(self):
        self.client = LLMClient()

    def reason(self, query, retrieved_chunks):

        context = "\n".join(retrieved_chunks[:4])

        result = self.client.reason(query, context)

        # Handle different API return formats
        if isinstance(result, dict):

            if "answer" in result:
                return result["answer"]

            if "response" in result:
                return result["response"]

            if "text" in result:
                return result["text"]

        return str(result)
