import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
from config import EMOTION_MODEL, MENTAL_MODEL, TOKENIZER_MODEL

class ClassifierManager:
    def __init__(self):
        self.emotion_model = tf.keras.models.load_model(EMOTION_MODEL, custom_objects={"TFBertModel": TFBertModel})
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

        self.mental_model = tf.keras.models.load_model(MENTAL_MODEL, custom_objects={"TFBertModel": TFBertModel})
        self.mental_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    def classify_emotion(self, text: str) -> str:
        inputs = self.emotion_tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=256
        )

        preds = self.emotion_model.predict(dict(inputs), verbose=0)
        label_id = preds.argmax(axis=-1)[0]

        id2label = {
            0: "amor",
            1: "confusion",
            2: "culpa",
            3: "deseo",
            4: "disgusto",
            5: "enojo",
            6: "felicidad",
            7: "miedo",
            8: "neutral",
            9: "sarcasmo",
            10: "sorpesa",
            11: "tristeza",
            12: "verguenza"
        }

        return id2label[label_id]

    def classify_mental_state(self, text: str) -> str:
        inputs = self.mental_tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=384
        )

        preds = self.mental_model.predict(dict(inputs), verbose=0)
        label_id = preds.argmax(axis=-1)[0]

        id2label = {
            0: "ansiedad",
            1: "bipolaridad",
            2: "depresion",
            3: "estres",
            4: "normal",
            5: "suicida",
            6: "transtorno de personalidad"
        }

        return id2label[label_id]
