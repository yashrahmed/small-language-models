from torch import (
    tensor, nn, softmax, no_grad, cat, argmax,
    triu,
    ones, zeros, arange,
    mean, var,
    sqrt, tanh,
    inf, pi
)

class GPTConfig:
    def __init__(self):
        self._vocab_size = 50257
        self._context_length = 1024
        self._embed_dim = 768
        self._n_heads = 12
        self._n_layers = 12
        self._dropout_rate = 0.1
        self._qkv_bias = False

    # --------------------
    # Getters
    # --------------------
    def get_vocab_size(self):
        return self._vocab_size

    def get_context_length(self):
        return self._context_length

    def get_embed_dim(self):
        return self._embed_dim

    def get_n_heads(self):
        return self._n_heads

    def get_n_layers(self):
        return self._n_layers

    def get_dropout_rate(self):
        return self._dropout_rate

    def get_qkv_bias(self):
        return self._qkv_bias

    # Optional: export as dict
    def to_dict(self):
        return {
            "vocab_size": self._vocab_size,
            "context_length": self._context_length,
            "embed_dim": self._embed_dim,
            "n_heads": self._n_heads,
            "n_layers": self._n_layers,
            "dropout_rate": self._dropout_rate,
            "qkv_bias": self._qkv_bias,
        }

class GPTModel(nn.Module):
        def __init__(self, model_config) -> None:
            super().__init__()
            self.tok_embed = nn.Embedding(model_config.get_vocab_size(), model_config.get_embed_dim())
            self.pos_embed = nn.Embedding(model_config.get_context_length(), model_config.get_embed_dim())
            self.main_drop = nn.Dropout(model_config.get_dropout_rate())

            self.txms_blocks = nn.Sequential(*[TransformerBlock(model_config) for _ in range(model_config.get_n_layers())]) # Convert to varargs

            self.main_norm = LayerNorm(model_config.get_embed_dim())
            self.logit_head = nn.Linear(model_config.get_embed_dim(), model_config.get_vocab_size(), bias=False) # Bias removed as per GPT2 spec

        def forward(self, token_batch):
            _, seq_len = token_batch.shape
            text_emb_op = self.tok_embed(token_batch)
            pos_emb_op = self.pos_embed(arange(seq_len, device=token_batch.device))
            x = text_emb_op + pos_emb_op
            x = self.main_drop(x)

            x = self.txms_blocks(x)

            x = self.main_norm(x)
            logits = self.logit_head(x)
            return logits

class CausalMultiHeadedAttention(nn.Module):
        def __init__(self, d_in, d_space, context_len, num_heads, drop_rate, qkv_bias_on=False):
            super().__init__()
            self.W_key = nn.Linear(d_in, d_space, qkv_bias_on)
            self.W_query = nn.Linear(d_in, d_space, qkv_bias_on)
            self.W_value = nn.Linear(d_in, d_space, qkv_bias_on)
            self.W_out = nn.Linear(d_space, d_space)
            self.register_buffer('mask', triu(ones(context_len, context_len), diagonal=1))
            self.dropout = nn.Dropout(drop_rate)
            self.head_dim = d_space // num_heads
            self.num_heads = num_heads
            self.d_space = d_space
            assert (d_space % num_heads == 0)
        
        def forward(self, input_emb): 
            batches, num_tokens, _ = input_emb.shape

            key_op = self.W_key(input_emb)
            query_op = self.W_query(input_emb)
            value_op = self.W_value(input_emb)

            # <SliceStep> 
            # This is requires to slice the combined Q,K,V matrices 
            # into two independent slices;
            # Multiplying in one-go doesn't work as this would cause the second slice of
            # Q to interfere with the first slice of K etc.

            key_op = key_op.view(batches, num_tokens, self.num_heads, self.head_dim)
            query_op = query_op.view(batches, num_tokens, self.num_heads, self.head_dim)
            value_op = value_op.view(batches, num_tokens, self.num_heads, self.head_dim)

            # Tranpose dims 1/2 of 3 in order to multiple
            # Pytorch uses the last two dims when performing a matmul().
            key_op = key_op.transpose(1, 2)
            query_op = query_op.transpose(1, 2)
            value_op = value_op.transpose(1, 2)

            attn_mat = key_op @ query_op.transpose(2, 3) # transpose along last 2 dims to align for matmul().
            mask_mat = self.mask.bool()[:num_tokens, :num_tokens]
            attn_mat.masked_fill_(mask_mat, -inf) # See broadcast rules.
            d_k = self.head_dim
            attn_mat = self.dropout(softmax(attn_mat / d_k ** 0.5, dim=-1))

            context_vec = attn_mat @ value_op
            # Put it back into a shape that matches the outputs of <SliceStep> and rebuild the contiguous view
            context_vec = context_vec.transpose(1, 2)
            context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_space)


            # Usually we would return after the previous op.
            # But for multiheaded attention, one last matrix multiply is required.
            # Think of it like a weighting sum layer for all the value vectors
            # a.k.a. remixing cross head interactions.
            return self.W_out(context_vec)

class FeedForward(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dims, 4 * dims), # Leave the bias true as this isn't an attention layer.
                GELU(), 
                nn.Linear(4 * dims, dims)
            )            
        
        def forward(self, x):
            return self.layers(x)

class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.K = sqrt(tensor(2 / pi)) # K = sqrt(2/pi)

    def forward(self, x):
        tmp = (x + 0.044715 * pow(x, 3))
        return 0.5 * x * (1 + tanh(self.K * tmp))

class LayerNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale = nn.Parameter(ones(input_dim))
        self.shift = nn.Parameter(zeros(input_dim)) # Scale and shift are trainable parameters
        self.eps = 1e-5
    
    def forward(self, x):
        x_mean = mean(x, dim=-1, keepdim=True)  # Keep dim true is required for broadcasting when x_norm is calculated.
        x_var = var(x, dim=-1, keepdim=True, unbiased=False) 
        x_norm = (x - x_mean) / sqrt(x_var + self.eps)
        return self.scale * x_norm + self.shift

class TransformerBlock(nn.Module):
        def __init__(self, model_config: GPTConfig) -> None:
            super().__init__()
            self.attn_heads = CausalMultiHeadedAttention(model_config.get_embed_dim(), 
                                            model_config.get_embed_dim(),
                                            model_config.get_context_length(),
                                            model_config.get_n_heads(),
                                            model_config.get_dropout_rate(),
                                            model_config.get_qkv_bias())
            self.l_norm_1 = LayerNorm(model_config.get_embed_dim())

            self.l_norm_2 = LayerNorm(model_config.get_embed_dim())
            self.feed_fwd = FeedForward(model_config.get_embed_dim())

            self.dropout_layer = nn.Dropout(model_config.get_dropout_rate())
        
        def forward(self, x):
            residual = x
            x = self.l_norm_1(x)
            x = self.attn_heads(x)
            x = self.dropout_layer(x)
            x = x + residual

            residual = x
            x = self.l_norm_2(x)
            x = self.feed_fwd(x)
            x = self.dropout_layer(x)
            x = x + residual

            return x

def generate_text_simple(input_tokens_batch, model, config, max_new_tokens):

    input_tokens_batch = input_tokens_batch[:, -config.get_context_length():] # Trim to context length

    for _ in range(max_new_tokens):
        with no_grad():
            logits = model(input_tokens_batch)
        logits = logits[:, -1, :] # Take ONLY the last context vector for each batch
        probs = softmax(logits, dim=-1)
        nxt_token_ids = argmax(probs, dim=-1, keepdim=True) # Find the index with the highest value; Keep dim allows the token ids for the whole batch to be appended to the input
        input_tokens_batch = cat((input_tokens_batch, nxt_token_ids), dim=-1)
    
    return input_tokens_batch


def text_ids_to_tokens(text, tokenizer):
    return tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0) # Add a batch dimension of size=1


def token_ids_to_text(token_ids, tokenizer):
    ids = token_ids.squeeze(0).tolist() # Remove a batch dimension of size=1
    return tokenizer.decode(ids)


GPT_CONFIG_124M = GPTConfig()    
