import requests
from config import RASA_URL

class IntentClassifier:
    def classify_intent(self, text: str) -> str:
        try:
            response = requests.post(RASA_URL, json={"text": text})
            return response.json()["intent"]["name"]
        except Exception:
            return "desconocido"

