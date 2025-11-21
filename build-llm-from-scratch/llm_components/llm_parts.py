import torch
from torch import nn, softmax, triu, ones, inf


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

    
