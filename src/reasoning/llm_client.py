import requests


class LLMClient:

    def __init__(self, api_url):
        self.api_url = api_url

    def reason(self, query, context):

        payload = {
            "query": query,
            "context": context
        }

        response = requests.post(
            self.api_url,   # <-- IMPORTANT
            json=payload
        )

        return response.json()