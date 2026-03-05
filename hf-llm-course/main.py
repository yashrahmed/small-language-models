import os

# Set up model cache directory before importing transformers.
cache_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import logging as hf_logging, pipeline

hf_logging.set_verbosity_error()


def main() -> None:
    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M",
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


if __name__ == "__main__":
    main()
