import os
import re

import json
import numpy as np
import requests
import tensorflow as tf
import tiktoken
from torch import (
    argmax,
    cat,
    multinomial,
    no_grad,
    nn,
    tensor,
    topk,
    where,
    device as torch_device,
)
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Special token constants
END_OF_TEXT_TOKEN = '<|endoftext|>'
UNK_TOKEN = '<|unk|>'

class VocabBuilder:
    def __init__(self) -> None:
        pass

    def build_vocab(self, text):
        vocab = {}
        tokens = re.split(r'([,.:;?_!\"()\']|--|\s)', text) # Use capture groups for retaining delimiters.
        tokens = [token.strip() for token in tokens if token.strip()]
        sorted_tokens = sorted(set(tokens))
        sorted_tokens.extend([END_OF_TEXT_TOKEN, UNK_TOKEN])
        for i, token in enumerate(sorted_tokens):
            vocab[token]= i
        return vocab

class SimpleTokenizer:
    def __init__(self, vocabulary) -> None:
        # Set up for encoding.
        # Set up for decoding.
        self.str_to_num_lookup = vocabulary
        self.num_to_str_lookup = {i: word for (word, i) in vocabulary.items()}
        self.unk_token_id = self.str_to_num_lookup[UNK_TOKEN]
    
    def encode(self, text):
        tokens = re.split(r'([,.?_!\"()\']|--|\s)', text) # Use capture groups for retaining delimiters.
        tokens = [token.strip() for token in tokens if token.strip()]
        # return the tokens ids after encoding.
        return [self.str_to_num_lookup.get(token, self.unk_token_id) for token in tokens]
    
    def decode(self, token_ids):
        tokens = [self.num_to_str_lookup.get(token_id, UNK_TOKEN) for token_id in token_ids]
        text_output = ' '.join(tokens)
        # Remove spaces around punctuation in a single substitution
        text_output = re.sub(r'\s*([,.?_!"()\'])\s*', r'\1', text_output)
        return text_output

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            i_max = i + max_length
            self.input_ids.append(tensor(token_ids[i:i_max]))
            self.target_ids.append(tensor(token_ids[i + 1 : i_max + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def calc_avg_loss_per_batch(dataloader, model, device, num_batches):
    if not num_batches or num_batches < 0:
        return float("nan")
    num_batches = min(int(num_batches), len(dataloader))
    total_loss = 0
    for i, (inputs_batch, target_batch) in enumerate(dataloader):
        if i == num_batches:
            break
        total_loss += calc_batch_loss(inputs_batch, target_batch, model, device)
    return total_loss / num_batches


def calc_batch_loss(inputs_batch, target_batch, model, device):
    inputs_batch = inputs_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(inputs_batch)
    return cross_entropy(logits.flatten(0, 1), target_batch.flatten(0, 1))


def create_dataloder_v1(
    text,
    max_length=256,
    stride=128,
    batch_size=4,
    drop_last=True,
    shuffle=True,
    num_workers=0,
):
    tokenizer = tiktoken.encoding_for_model("gpt-2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def generate_text(
    input_tokens_batch,
    model,
    config,
    max_new_tokens,
    top_k=25,
    temp=1.0,
    model_type="hf",
    device=torch_device("cpu"),
):
    input_tokens_batch = input_tokens_batch.to(device)

    for _ in range(max_new_tokens):
        cropped_tokens = input_tokens_batch[:, -config.get_context_length() :]
        with no_grad():
            logits = model(cropped_tokens)
            logits = logits.logits if model_type == "hf" else logits
        logits = logits[:, -1, :]

        if top_k and top_k > 0:
            top_k_logits, _ = topk(logits, top_k, dim=-1)
            min_topk_val = top_k_logits[:, -1]
            logits = where(logits < min_topk_val, tensor(float("-inf")).to(device), logits)

        if temp and temp > 0.0:
            logits /= temp
            probs = softmax(logits, dim=-1)
            nxt_token_ids = multinomial(probs, num_samples=1)
        else:
            probs = softmax(logits, dim=-1)
            nxt_token_ids = argmax(probs, dim=-1, keepdim=True)
        input_tokens_batch = cat((input_tokens_batch, nxt_token_ids), dim=-1)

    return input_tokens_batch


def generate_text_simple(
    input_tokens_batch,
    model,
    config,
    max_new_tokens,
    model_type="hf",
    device=torch_device("cpu"),
):
    input_tokens_batch = input_tokens_batch.to(device)

    for _ in range(max_new_tokens):
        cropped_tokens = input_tokens_batch[:, -config.get_context_length() :]
        with no_grad():
            logits = model(cropped_tokens)
            logits = logits.logits if model_type == "hf" else logits

        logits = logits[:, -1, :]
        probs = softmax(logits, dim=-1)
        nxt_token_ids = argmax(probs, dim=-1, keepdim=True)
        input_tokens_batch = cat((input_tokens_batch, nxt_token_ids), dim=-1)

    return input_tokens_batch


def text_ids_to_tokens(text, tokenizer):
    return tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    ids = token_ids.squeeze(0).tolist()
    return tokenizer.decode(ids)


def train_model_simple(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs,
    eval_batch_interval=5,
    eval_batch_size=5,
    verbose=False,
):
    num_tokens_seen_log, train_loss_log, val_loss_log = [], [], []
    num_tokens_seen = 0

    step_num = 0
    for i in range(num_epochs):
        if verbose:
            print("_______")
        model.train()
        for (input_batch, target_batch) in train_dataloader:
            optimizer.zero_grad()
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            num_tokens_seen += input_batch.numel()
            if step_num % eval_batch_interval == 0:
                model.eval()
                with no_grad():
                    train_loss = calc_avg_loss_per_batch(
                        train_dataloader, model, device, eval_batch_size
                    )
                    val_loss = calc_avg_loss_per_batch(
                        val_dataloader, model, device, eval_batch_size
                    )
                    num_tokens_seen_log.append(num_tokens_seen)
                    train_loss_log.append(train_loss)
                    val_loss_log.append(val_loss)
                    if verbose:
                        print(
                            f"Tokens seen = {num_tokens_seen} | Train loss = {train_loss} | "
                            f"Validation loss = {val_loss} | Epoch={i} | stepNum = {step_num}"
                        )
                model.train()
            step_num += 1

    return num_tokens_seen, train_loss_log, val_loss_log


def load_weights_from_hfmodel(gpt, gpt_hf):
    def to_param(left, right):
        assert left.shape == right.shape, ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
        return nn.Parameter(right.clone().detach())

    d = gpt_hf.state_dict()

    gpt.pos_emb.weight = to_param(gpt.pos_emb.weight, d["transformer.wpe.weight"])
    gpt.tok_emb.weight = to_param(gpt.tok_emb.weight, d["transformer.wte.weight"])

    for b in range(12):
        q_w, k_w, v_w = np.split(d[f"transformer.h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = to_param(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = to_param(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = to_param(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(d[f"transformer.h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = to_param(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = to_param(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = to_param(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = to_param(
            gpt.trf_blocks[b].att.out_proj.weight,
            d[f"transformer.h.{b}.attn.c_proj.weight"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = to_param(
            gpt.trf_blocks[b].att.out_proj.bias, d[f"transformer.h.{b}.attn.c_proj.bias"]
        )

        gpt.trf_blocks[b].ff.layers[0].weight = to_param(
            gpt.trf_blocks[b].ff.layers[0].weight, d[f"transformer.h.{b}.mlp.c_fc.weight"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = to_param(
            gpt.trf_blocks[b].ff.layers[0].bias, d[f"transformer.h.{b}.mlp.c_fc.bias"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = to_param(
            gpt.trf_blocks[b].ff.layers[2].weight, d[f"transformer.h.{b}.mlp.c_proj.weight"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = to_param(
            gpt.trf_blocks[b].ff.layers[2].bias, d[f"transformer.h.{b}.mlp.c_proj.bias"]
        )

        gpt.trf_blocks[b].norm1.scale = to_param(
            gpt.trf_blocks[b].norm1.scale, d[f"transformer.h.{b}.ln_1.weight"]
        )
        gpt.trf_blocks[b].norm1.shift = to_param(
            gpt.trf_blocks[b].norm1.shift, d[f"transformer.h.{b}.ln_1.bias"]
        )
        gpt.trf_blocks[b].norm2.scale = to_param(
            gpt.trf_blocks[b].norm2.scale, d[f"transformer.h.{b}.ln_2.weight"]
        )
        gpt.trf_blocks[b].norm2.shift = to_param(
            gpt.trf_blocks[b].norm2.shift, d[f"transformer.h.{b}.ln_2.bias"]
        )

        gpt.final_norm.scale = to_param(gpt.final_norm.scale, d["transformer.ln_f.weight"])
        gpt.final_norm.shift = to_param(gpt.final_norm.shift, d["transformer.ln_f.bias"])
        gpt.out_head.weight = to_param(gpt.out_head.weight, d["transformer.wte.weight"])
