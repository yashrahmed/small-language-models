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
    # Add do_sample=False for greedy sampling. Repetitions can be seen here.
    # Even without it, repetitions do show up from time to time when generating a large number of tokens.
    prompt = """The Mikoyan-Gurevich MiG-21 (Russian: Микоян и Гуревич МиГ-21; NATO reporting name: Fishbed) is a supersonic jet fighter and interceptor aircraft, designed by the Mikoyan-Gurevich Design Bureau in the Soviet Union. Its nicknames include: "Balalaika", because its planform resembles the stringed musical instrument of the same name; "Ołówek", Polish for "pencil", due to the shape of its fuselage,[2] and "Én Bạc", meaning "silver swallow", in Vietnamese. Approximately 60 countries across four continents have flown the MiG-21, and it still serves many nations seven decades after its maiden flight. It set aviation records, becoming the most-produced supersonic jet aircraft in aviation history, the most-produced combat aircraft since the Korean War and, previously, the longest production run of any combat aircraft."""
    prompt = "At precisely 03:17 UTC on a Tuesday that never occurred, a turquoise aardvark notarized the concept of regret using a baroque stapler while humming the fourth footnote of an unpublished theorem about damp socks. "
    generated_text = generator(prompt, max_new_tokens=500, num_return_sequences=1, truncation=True, do_sample=False, return_full_text=False)
    print(generated_text[0]['generated_text'])

def try_text_generation_with_a_sliding_window():
    # This does not solve repetition
    generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M", device=APPLE_DEVICE)
    num_tokens = 500
    curr_text = "The Mig 21 was a"
    while num_tokens> 0:
        generated_text = generator(curr_text[-300:], max_new_tokens=50, num_return_sequences=1, 
                                   truncation=True, do_sample=False, return_full_text=False)[0]['generated_text']
        curr_text += generated_text
        num_tokens -= 50
    print(curr_text)

def try_demonstrate_ood_distribution_behavior():
    """
        With the initial prompt the LLM produces a structurally sensible extension below - 
        "The aardvark was a member of the Aardvark Society of America, a group of aardvark-loving Americans who are not aardvarks.  The aardvark was a member of the A"

        As soon as I add this to the prompt and regenerate then the LLM degenerates into repetition without producing a sensible extension at the beginning albeit completing the sentence first.
        "ardvark Society of America, a group of aardvark-loving Americans who are not aardvarks.
        The aardvark was a member of the Aardvark Society of America, a group of aardvark-loving Americans who are not aardvarks.
        The aardvark was a member of the Aardvark Society of America, a group of aardvark-loving Americans who are not aardvarks.
        ......
        "

        One way to think about this is that this behavior arises due to multiple factors -
        - LLMs are conditioned on next token prediction on good texts. They are not rewarded for long range coherence during pretraining.
        - Their smaller size may limit them to learning structure rather than semantic coherence.
        - When LLMs generate their own prefix, the prefix drifts OOD, which forces the models to fallback on repetitive behavior.
        
        The experiment below shows point #3 where the model completes the sentence in a way that is structurally sound but then falls back to the safe OOD behavior.

        The intuition that a base model will steer away from generations is incorrect as the model isn't incentivised to do that.

    """

    generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M", device=APPLE_DEVICE)
    prompt = "At precisely 03:17 UTC on a Tuesday that never occurred, a turquoise aardvark notarized the concept of regret using a baroque stapler while humming the fourth footnote of an unpublished theorem about damp socks. "
    generated_text = generator(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True, do_sample=False, return_full_text=False)[0]['generated_text']
    print(generated_text)
    prompt += generated_text
    print('_____________')
    print(prompt)
    print('######')
    generated_text = generator(prompt, max_new_tokens=200, num_return_sequences=1, truncation=True, do_sample=False, return_full_text=False)[0]['generated_text']
    print(generated_text)

if __name__ == "__main__":
    # try_classify_sentiment()
    try_demonstrate_ood_distribution_behavior()
