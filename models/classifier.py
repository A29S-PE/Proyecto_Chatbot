from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import EMOTION_MODEL, MENTAL_MODEL

class ClassifierManager:
    def __init__(self):
        # Emoción
        emotion_tokenizer   = AutoTokenizer.from_pretrained(EMOTION_MODEL)
        emotion_model       = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
        self.emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer)

        # Estado mental
        mental_tokenizer    = AutoTokenizer.from_pretrained(MENTAL_MODEL)
        mental_model        = AutoModelForSequenceClassification.from_pretrained(MENTAL_MODEL)
        self.mental_pipeline = pipeline("text-classification", model=mental_model, tokenizer=mental_tokenizer)

    def classify_emotion(self, text: str) -> str:
        return self.emotion_pipeline(text)[0]["label"]

    def classify_mental_state(self, text: str) -> str:
        return self.mental_pipeline(text)[0]["label"]
