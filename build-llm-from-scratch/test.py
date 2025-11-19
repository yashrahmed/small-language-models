from llm_components import VocabBuilder, SimpleTokenizer, END_OF_TEXT_TOKEN

def load_text():
    with open('verdict.txt', 'r') as book:
        text = book.read()
        return text

def test_custom_tokenizer():
    # tokens = tokenize(text)
    # print(tokens[:50])
    v_builder = VocabBuilder()
    vocab = v_builder.build_vocab(text)
    
    tok = SimpleTokenizer(vocab)
    sentence1 = "Hello! Do you like tea?"
    sentence2 = "In the sunlit terraces of the palace."
    sentence = f" {END_OF_TEXT_TOKEN} ".join([sentence1, sentence2])
    tok_ids = tok.encode(sentence)
    text = tok.decode(tok_ids)
    print(sentence)
    print(text)
    
def test_tiktoken():
    # Tikton library provides BPE tokenizer.
    # This tokenizer does not use <|unk|> token.
    # When an unknow word is encountered then, BPE splits it down into smaller chunks that are part of its vocabulary.
    # It keeps going until it encounters individual characters.

    # I would like to implement it.

    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-2")
    sentence1 = "Hello! Do you like tea?"
    sentence2 = "In the sunlit terraces of the palace."
    sentence = f"{END_OF_TEXT_TOKEN}".join([sentence1, sentence2])
    print(sentence)
    token_ids = tokenizer.encode(sentence, allowed_special={"<|endoftext|>"})
    print(token_ids)
    print(tokenizer.decode(token_ids))

def test_data_sampling():
    # import tiktoken
    # tokenizer = tiktoken.encoding_for_model("gpt-2")
    text = load_text()
    
    from llm_components import create_dataloder_v1

    dataloader = create_dataloder_v1(text, batch_size=3, max_length=4, stride=1, shuffle=False)
    dl_iter = iter(dataloader)
    inputs, targets = next(dl_iter)
    print(inputs)
    # print(targets)
    print('++++++++++++++++++++')
    inputs, targets = next(dl_iter)
    print(inputs)
    # print(targets)

def testing_simple_embedding():
    """
    input text --> input token --> input token ids --> token embedding + position embedding --> input embedding.
    """
    from torch.nn import Embedding
    from torch import tensor
    import torch
    import tiktoken

    tokenizer = tiktoken.encoding_for_model("gpt-2")
    sentence1 = "Your journey starts with one step"
    token_ids = tokenizer.encode(sentence1)
    print(tokenizer.n_vocab)
    print(token_ids)


    vocab_size = tokenizer.n_vocab
    embed_dim = 3
    max_seq_len = len(token_ids)

    torch.manual_seed(123)
    
    tok_embedding_layer = Embedding(vocab_size, embed_dim)
    pos_embedding_layer = Embedding(max_seq_len, embed_dim)

    x = tok_embedding_layer(tensor(token_ids))
    positions = torch.arange(max_seq_len)
    x_p = pos_embedding_layer(positions)

    print(x)
    print(x_p)

    x_act = x + x_p
    print(x_act)

def building_simple_self_attention():
    from torch import tensor, softmax
    inputs = tensor(
        [
            [0.43, 0.15, 0.89],# Your
            [0.55, 0.87, 0.66],# journey
            [0.57, 0.85, 0.64],# starts
            [0.22, 0.58, 0.33],# with
            [0.77, 0.25, 0.10],# one
            [0.05, 0.80, 0.55] # step
        ]
    )
    # query = inputs[[1]]
    # print(query.shape, inputs.T.shape)
    # attn_score_2 = query @ inputs.T # Effectively compute the dot product of the second embedding with all embeddings including itself.
    # # attn = F.normalize(attn_score_2, p=1, dim=1) # Normalize along rows using the Manhattan metric.
    # attn = softmax(attn_score_2, dim=1)
    
    attn_2 = inputs @ inputs.T
    attn_weights = softmax(attn_2, dim=1)
    # print(attn_weights)
    attn_output = attn_weights @ inputs
    print(attn_output)

def building_weighted_self_attention():
    from torch import tensor, softmax, manual_seed, nn, rand

    manual_seed(123)

    inputs = tensor(
        [
            [0.43, 0.15, 0.89],# Your
            [0.55, 0.87, 0.66],# journey
            [0.57, 0.85, 0.64],# starts
            [0.22, 0.58, 0.33],# with
            [0.77, 0.25, 0.10],# one
            [0.05, 0.80, 0.55] # step
        ]
    )

    dim_in = 3
    dim_out = 2 

    W_q = nn.Parameter(rand(dim_in, dim_out), requires_grad=False)
    W_k = nn.Parameter(rand(dim_in, dim_out), requires_grad=False)
    W_v = nn.Parameter(rand(dim_in, dim_out), requires_grad=False)

    q_v = inputs @ W_q
    k_v = inputs @ W_k
    v_v = inputs @ W_v

    attn = q_v @ k_v.T
    d_k = q_v.shape[1]
    attn = softmax(attn / d_k**0.5, dim = 1)

    output = attn @ v_v

    print(output)

def build_compact_attention_layers():
    import torch
    from torch import nn, softmax, tensor, rand, manual_seed

    manual_seed(123)

    class SelfAttentionV1(nn.Module):
        def __init__(self, d_in, d_space):
            super().__init__()
            self.W_query = nn.Parameter(rand(d_in, d_space))
            self.W_key = nn.Parameter(rand(d_in, d_space))
            self.W_value = nn.Parameter(rand(d_in, d_space))
        
        def forward(self, input_emb):
            query_op = input_emb @ self.W_query
            key_op = input_emb @ self.W_key
            value_op = input_emb @ self.W_value

            attn_mat = query_op @ key_op.T
            d_k = key_op.shape[-1]
            attn_mat = softmax(attn_mat / d_k ** 0.5, dim=-1) 

            context_vec = attn_mat @ value_op
            return context_vec
    
    class SelfAttentionV2(nn.Module):
        def __init__(self, d_in, d_space, bias_on=False):
            super().__init__()
            self.W_key = nn.Linear(d_in, d_space, bias_on)
            self.W_query = nn.Linear(d_in, d_space, bias_on)
            self.W_value = nn.Linear(d_in, d_space, bias_on)
        
        def forward(self, input_emb):
            key_op = self.W_key(input_emb)
            query_op = self.W_query(input_emb)
            value_op = self.W_value(input_emb)

            attn_mat = query_op @ key_op.T
            d_k = key_op.shape[-1]
            attn_mat = softmax(attn_mat / d_k ** 0.5, dim=-1)

            context_vec = attn_mat @ value_op
            return context_vec

        
    inputs = tensor(
        [
            [0.43, 0.15, 0.89],# Your
            [0.55, 0.87, 0.66],# journey
            [0.57, 0.85, 0.64],# starts
            [0.22, 0.58, 0.33],# with
            [0.77, 0.25, 0.10],# one
            [0.05, 0.80, 0.55] # step
        ]
    )

    layer_v1 = SelfAttentionV1(3, 2)
    print(layer_v1(inputs))
    
    layer_v2 = SelfAttentionV2(3, 2, False)
    
    # # Experiment to copy the weights of one into the other to prove that the operations are the same
    # with torch.no_grad():
    #     layer_v2.W_key.weight.copy_(layer_v1.W_key.T)
    #     layer_v2.W_query.weight.copy_(layer_v1.W_query.T)
    #     layer_v2.W_value.weight.copy_(layer_v1.W_value.T)

    print(layer_v2(inputs))

def building_causal_attention_wdropout():
    import torch
    from torch import nn, softmax, tensor, rand, manual_seed, triu, ones, inf

    manual_seed(123)

    class CausalAttention(nn.Module):
        def __init__(self, d_in, d_space, context_len, drop_rate, bias_on=False):
            super().__init__()
            self.W_key = nn.Linear(d_in, d_space, bias_on)
            self.W_query = nn.Linear(d_in, d_space, bias_on)
            self.W_value = nn.Linear(d_in, d_space, bias_on)
            self.register_buffer('mask', triu(ones(context_len, context_len), diagonal=1))
            self.dropout = nn.Dropout(drop_rate)
        
        def forward(self, input_emb): 
            _, num_tokens, _ = input_emb.shape
            key_op = self.W_key(input_emb)
            query_op = self.W_query(input_emb)
            value_op = self.W_value(input_emb)

            attn_mat = query_op @ key_op.transpose(1,2) # Require for batch operations
            d_k = key_op.shape[-1]
            # num_tokens must be < context_len
            attn_mat = attn_mat.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -inf) # trailing _ indicates an inplace variant of the function.
            attn_mat = softmax(attn_mat / d_k ** 0.5, dim=-1)

            context_vec = attn_mat @ value_op
            return context_vec
        
    inputs = tensor(
        [ # First batch
            [
                [0.43, 0.15, 0.89],# Your
                [0.55, 0.87, 0.66],# journey
                [0.57, 0.85, 0.64],# starts
                [0.22, 0.58, 0.33],# with
                [0.77, 0.25, 0.10],# one
                [0.05, 0.80, 0.55] # step
            ]
        ]
    )

    layer_v1 = CausalAttention(3, 2, 6, 0)
    print(layer_v1(inputs))

def building_causal_multiheaded_attention():
    import torch
    from torch import nn, softmax, tensor, manual_seed, triu, ones, inf

    inputs = tensor(
        [
            [0.43, 0.15, 0.89],# Your
            [0.55, 0.87, 0.66],# journey
            [0.57, 0.85, 0.64],# starts
            [0.22, 0.58, 0.33],# with
            [0.77, 0.25, 0.10],# one
            [0.05, 0.80, 0.55] # step
        ]
    )

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

            key_op = self.W_key(inputs)
            query_op = self.W_query(inputs)
            value_op = self.W_value(inputs)

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

    
    inputs = torch.stack((inputs,), dim=0)
    print(inputs.shape)

    manual_seed(123)

    layer = CausalMultiHeadedAttention(3, 4, 6, 2, 0.5)
    output = layer(inputs)
    print(output.shape)

if __name__ == '__main__':
    building_causal_multiheaded_attention()
    # building_causal_attention_wdropout()
    # build_compact_attention_layers()
    print('+++++++++++++++++++++++++++')
    # building_weighted_self_attention()
    # building_simple_self_attention()
    # testing_simple_embedding()
    # test_data_sampling()

     