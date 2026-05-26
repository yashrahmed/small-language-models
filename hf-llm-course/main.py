import os

# Set up model cache directory before importing transformers.
cache_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir

from transformers import logging as hf_logging, pipeline
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()


def text_gen_demo() -> None:
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device="mps",
    )
    prompt = "The MiG-21 is known for"
    output = generator(
        prompt,
        max_new_tokens=200,
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    print(output[0]["generated_text"])


def qa_demo() -> None:
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device="mps",
    )
    question = "When did the Mig 21 make its maiden flight?"
    context = """
    The MiG-21 jet fighter was a continuation of Soviet jet fighters, starting with the transonic MiG-15 and MiG-17, and the supersonic MiG-19.
    A number of experimental Mach 2 Soviet designs were based on nose intakes with either swept-back wings, such as the Sukhoi Su-7, or tailed deltas, of which the MiG-21 would be the most successful.
    Development of what would become the MiG-21 began in the early 1950s when Mikoyan OKB finished a preliminary design study for a prototype designated Ye-1 in 1954. This project was very quickly reworked when it was determined that the planned engine was underpowered; the redesign led to the second prototype, the Ye-2. Both these and other early prototypes featured swept wings. The first prototype with the delta wings found on production variants was the Ye-4. It made its maiden flight on 16 June 1955 and its first public appearance during the Soviet Aviation Day display at Moscow's Tushino airfield in July 1956.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the provided context. "
                'If the answer is not in the context, say "I don\'t know based on the provided context."'
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context.strip()}\n\nQuestion: {question}",
        },
    ]
    output = generator(
        messages,
        max_new_tokens=64,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    print(output[0]["generated_text"][-1]["content"].strip())


if __name__ == "__main__":
    qa_demo()
    # text_gen_demo()
