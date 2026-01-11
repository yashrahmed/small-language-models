import os

# Setup model cache directory before importing transformers.
cache_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir

# Import AFTER setting up the HF_HOME variable
from transformers import pipeline

APPLE_DEVICE="mps"

def try_classify_sentiment():
    sentence = "I have been wanting to go this cafe for a long time."
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=APPLE_DEVICE)
    sentiment = classifier(sentence)
    print(sentiment)

def try_text_generation():
    generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M", device=APPLE_DEVICE)
    generated_text = generator("The Mig 21 was a", max_new_tokens=100, num_return_sequences=1, truncation=True)
    print(generated_text[0]['generated_text'])

if __name__ == "__main__":
    # try_classify_sentiment()
    try_text_generation()
