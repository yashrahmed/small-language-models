import torch
from torch import nn, softmax, triu, ones, inf

class GPTConfig124M:
    def __init__(self):
        self._vocab_size = 50257
        self._content_length = 1024
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

    def get_content_length(self):
        return self._content_length

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
            "content_length": self._content_length,
            "embed_dim": self._embed_dim,
            "n_heads": self._n_heads,
            "n_layers": self._n_layers,
            "dropout_rate": self._dropout_rate,
            "qkv_bias": self._qkv_bias,
        }

class CausalMultiHeadedAttention(nn.Module):
        def __init__(self, d_in, d_space, context_len, num_heads, drop_rate, bias_on=False):
            super().__init__()
            self.W_key = nn.Linear(d_in, d_space, bias_on)
            self.W_query = nn.Linear(d_in, d_space, bias_on)
            self.W_value = nn.Linear(d_in, d_space, bias_on)
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

GPT_CONFIG_124M = GPTConfig124M()    
