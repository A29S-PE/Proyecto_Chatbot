import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import EMOTION_MODEL, MENTAL_MODEL, TOKENIZER_MODEL

class ClassifierManager:
    def __init__(self):
        self.emotion_model = tf.keras.models.load_model(EMOTION_MODEL)
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

        self.mental_model = tf.keras.models.load_model(MENTAL_MODEL)
        self.mental_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    def classify_emotion(self, text: str) -> str:
        inputs = self.emotion_tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=128
        )

        preds = self.emotion_model.predict(dict(inputs), verbose=0)
        label_id = preds.argmax(axis=-1)[0]

        id2label = {
            0: "alegría",
            1: "tristeza",
            2: "enojo",
            3: "miedo",
            4: "sorpresa"
        }

        return id2label[label_id]

    def classify_mental_state(self, text: str) -> str:
        inputs = self.mental_tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=128
        )

        preds = self.mental_model.predict(dict(inputs), verbose=0)
        label_id = preds.argmax(axis=-1)[0]

        id2label = {
            0: "alegría",
            1: "tristeza",
            2: "enojo",
            3: "miedo",
            4: "sorpresa"
        }

        return id2label[label_id]

