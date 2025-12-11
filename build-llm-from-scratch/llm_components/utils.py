import re
import numpy as np

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
    manual_seed,
    device as torch_device,
)
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel
from .gpt_parts import GPT_CONFIG_124M, GPTModel

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


def calc_acc_binary(dataloader, model, device, num_batches):
    if not num_batches or num_batches < 0:
        return float("nan")
    num_batches = min(int(num_batches), len(dataloader))
    total_examples = 0
    correct = 0
    for i, (inputs_batch, target_batch) in enumerate(dataloader):
        inputs_batch = inputs_batch.to(device)
        target_batch = target_batch.to(device)
        if i >= num_batches:
            break
        logits = model(inputs_batch)[:, -1, :]
        pred = argmax(logits, dim=-1)
        total_examples += pred.shape[0]
        correct += (pred == target_batch).sum().item()
    return correct / total_examples

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

def calc_avg_loss_per_batch_binary(dataloader, model, device, num_batches):
    if not num_batches or num_batches < 0:
        return float("nan")
    num_batches = min(int(num_batches), len(dataloader))
    total_loss = 0
    for i, (inputs_batch, target_batch) in enumerate(dataloader):
        if i == num_batches:
            break
        total_loss += calc_batch_loss_binary(inputs_batch, target_batch, model, device)
    return total_loss / num_batches

def calc_batch_loss(inputs_batch, target_batch, model, device):
    inputs_batch = inputs_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(inputs_batch)
    return cross_entropy(logits.flatten(0, 1), target_batch.flatten(0, 1))

def calc_batch_loss_binary(inputs_batch, target_batch, model, device):
    """
    Unlike regular calc_batch_loss
    where each instance of an input contained N input vectors and N output vectors (hence the need to flatten),
    the binary problem contains N input vectors and 1 output vectors. We will pick the last set of logits as a target (hence :,-1,:)
    """
    inputs_batch = inputs_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(inputs_batch)[:, -1, :] # Pick only the last context for classification. 
    return cross_entropy(logits, target_batch)

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
    context_length,
    max_new_tokens,
    model_type="hf",
    device=torch_device("cpu"),
):
    input_tokens_batch = input_tokens_batch.to(device)

    for _ in range(max_new_tokens):
        cropped_tokens = input_tokens_batch[:, -context_length:]
        with no_grad():
            logits = model(cropped_tokens)
            logits = logits.logits if model_type == "hf" else logits

        logits = logits[:, -1, :]
        probs = softmax(logits, dim=-1)
        nxt_token_ids = argmax(probs, dim=-1, keepdim=True)
        input_tokens_batch = cat((input_tokens_batch, nxt_token_ids), dim=-1)

    return input_tokens_batch

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

def load_gpt2_pretrained(device, seed_val):
    config = GPT_CONFIG_124M
    config._qkv_bias= True # GPT 2 use biases on Query, key and value matrices. This was stopped in subsequent models.
    manual_seed(seed_val)
    gpt_model = GPTModel(config)

    # Download and initialize GPT model from Huggingface hub.
    gpt_hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", cache_dir="./checkpoints")
    gpt_hf.eval()

    load_weights_from_hfmodel(gpt_model, gpt_hf)
    gpt_model.eval()
    return config, gpt_model.to(device)

def text_to_token_ids(text, tokenizer):
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

def train_model_simple_binary(
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
            loss = calc_batch_loss_binary(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            num_tokens_seen += input_batch.numel()
            if step_num % eval_batch_interval == 0:
                model.eval()
                with no_grad():
                    train_loss = calc_avg_loss_per_batch_binary(
                        train_dataloader, model, device, eval_batch_size
                    )
                    val_loss = calc_avg_loss_per_batch_binary(
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
        if verbose:
            train_acc = calc_acc_binary(train_dataloader, model, device, eval_batch_size)
            val_acc =  calc_acc_binary(val_dataloader, model, device, eval_batch_size)
            print(f"Train acc at epoch {i} = {train_acc*100:.2f}%")
            print(f"Val acc at epoch {i} = {val_acc*100:.2f}%")

    return num_tokens_seen, train_loss_log, val_loss_log
