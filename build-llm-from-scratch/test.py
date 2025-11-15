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






if __name__ == '__main__':
    building_weighted_self_attention()
    # building_simple_self_attention()
    # testing_simple_embedding()
    # test_data_sampling()

     