from llm_components import VocabBuilder, SimpleTokenizer, END_OF_TEXT_TOKEN, CausalMultiHeadedAttention, LayerNorm, GELU, FeedForward, GPT_CONFIG_124M, TransformerBlock, create_dataloder_v1, GPTModel

def load_text():
    with open('verdict.txt', 'r') as book:
        text = book.read()
        return text

def test_custom_tokenizer():
    text = load_text()
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
    
    inputs = torch.stack((inputs,), dim=0)
    print(inputs.shape)

    manual_seed(123)

    layer = CausalMultiHeadedAttention(3, 4, 6, 2, 0.5)
    output = layer(inputs)
    print(output.shape)

def try_layer_norm():
    import torch
    from torch import nn, softmax, tensor, manual_seed, triu, ones, zeros, inf, randn, mean, var, sqrt

    manual_seed(123)
    norm_layer = LayerNorm(6)
    batch_ex = randn(2, 5) # 2 inputs of 5 dim each.
    dummmy_layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) # A composite layer made of 6 neurons with ReLU activation.
    output = dummmy_layer(batch_ex)
    output = norm_layer(output)
    print(output)

    keep_dim = True
    op_mean = mean(output, dim=-1, keepdim=keep_dim)
    op_var = var(output, dim=-1, keepdim=keep_dim)
    print(op_mean)
    print(op_var)

def try_ff_layer():
    import torch
    from torch import nn, softmax, tensor, manual_seed, triu, ones, zeros, inf, randn, mean, var, sqrt, tanh, pi, pow, linspace, rand
    
    import matplotlib.pyplot as plt

    gelu, relu = GELU(), nn.ReLU()

    x =  linspace(-3, 3, 100) # 100 x-axis values.
    y_gelu, y_relu = gelu(x), relu(x)
    for i, (y, plot_label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'RELU']), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.xlabel('x axis')
        plt.ylabel('activation')
        plt.title(f"{plot_label} activation")
        plt.grid(True)
    plt.tight_layout()
    plt.show()
        
    inputs = rand(2, 6, 768)
    output = FeedForward(768)(inputs)
    print(output.shape)

def build_a_txfm_block():
    import torch
    from torch import nn, softmax, tensor, manual_seed, triu, ones, zeros, inf, randn, mean, var, sqrt, tanh, pi, pow, linspace, rand
    
    import matplotlib.pyplot as plt

    


    x = rand(2, 3, GPT_CONFIG_124M.get_embed_dim())

    layer = TransformerBlock(GPT_CONFIG_124M)

    output = layer(x)

    print(x.shape)
    print(output.shape)

def build_gpt_2():
    import torch
    from torch import nn, softmax, tensor, manual_seed, triu, ones, zeros, inf, randn, mean, var, sqrt, tanh, pi, pow, linspace, rand, arange, argmax, cat
    import tiktoken
    from llm_components import GPTModel, generate_text_simple, text_ids_to_tokens, token_ids_to_text

    manual_seed(123)
     
    tokenizer = tiktoken.encoding_for_model("gpt-2")
    
    # Sentences chosen to have exactly 10 tokens
    sentence1 = "Hello there boy! Do you like tea sir?" 
    sentence2 = "In the sunlit terraces of the palace."
    sentence3 = "Hello, I am"

    # batch = tensor([
    #     # tokenizer.encode(sentence1),
    #     # tokenizer.encode(sentence2),
    #     tokenizer.encode(sentence3)
    # ])
    batch = text_ids_to_tokens(sentence3, tokenizer)

    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum([p.numel() for p in model.parameters()])
    # print(total_params) # Will print 163M params which is HIGHER than 124M params. 

    # Both the lines end up printing torch.Size([50257, 768])
    # This is because the logit head stores weights as a transpose even though it maps from 768 dims --> 50257 dims.
    # This allows the weights to be "tied" thereby removing the need for a separate logit head which leads to ~124M params.
    # This book does not use tying.
    # print(model.tok_embed.weight.shape)
    # print(model.logit_head.weight.shape)

    # total_size_bytes = 4 * total_params
    # total_size_mb = total_size_bytes / (1024 * 1024)
    # print(f"Total size is {total_size_mb:.2f} MB")

    # test_output = model(batch)
    # print(test_output.shape)

    # Test the text generation here.


    # print(batch)
    model.eval() # Puts the model in eval mode; Different from no_grad; Some layers e.g. dropout work differently
    output_tokens = generate_text_simple(batch, model, GPT_CONFIG_124M, 6)
    print(token_ids_to_text(output_tokens, tokenizer))
    # print(tokenizer.decode(output_tokens[0, :].tolist()))

def try_measure_loss():
    import torch
    from torch import (nn, softmax, tensor, manual_seed, triu, ones, zeros, inf, randn, mean, var, sqrt, tanh, pi, pow, linspace, rand, arange, argmax, cat,
                       log)
    import tiktoken
    from llm_components import GPTModel, generate_text_simple, text_ids_to_tokens, token_ids_to_text
    
    
    config = GPT_CONFIG_124M
    config._dropout_rate = 0.1
    config._context_length = 256
    manual_seed(123)
    model = GPTModel(config)
    model.eval()
     
    tokenizer = tiktoken.encoding_for_model("gpt-2")

    # The last words in both sentences are target words
    # sent1_tokens = tokenizer.encode("Every effort moves you")
    # sent2_tokens = tokenizer.encode("I really like chocolate")

    # Setting tokens manually to follow along with the book.
    sent1_tokens = [16833, 3626, 6100, 345]
    sent2_tokens = [40, 1107, 588, 11311]

    inputs = tensor([
        sent1_tokens[:3],
        sent2_tokens[:3]
    ])

    targets = tensor([
        sent1_tokens[1:],
        sent2_tokens[1:]
    ])

    # Not relevant to the topic but here is an interesting effect of BPE.
    # Even though the second line is the last three words of the first, the token count is the same.
    # In the second sentence, BPE splits the word "effort" in "eff" and "ort".
    # print(tokenizer.encode("Every effort moves you")) # 4 tokens for 4 words
    # print(tokenizer.encode("effort moves you")) # 4 tokens for 3 words.

    # Run the inputs through model
    with torch.no_grad():
        logits = model(inputs)
    probs = softmax(logits, dim=-1)
    op_token_ids = argmax(probs, dim=-1, keepdim=True)
    print(probs.shape)
    print(op_token_ids)

    # The loss to measure can be described as "Negative Average log probability of target tokens".
    # The goal is to use Backprop (hence the need to define a loss).
    # There isn't a need to MINIMIZE the probs of incorrect tokens as softmax takes care of that
    # i.e. increasing the target probability automatically reduces the probs of the incorrect tokens when using softmax.
    # The technical name for this loss in cross-entropy loss.
    tgt_1_probs = probs[0, [0, 1, 2], targets[0]]
    tgt_2_probs = probs[1, [0, 1, 2], targets[1]]
    print(tgt_1_probs)
    print(tgt_2_probs)

    all_logprobs = log(cat((tgt_1_probs, tgt_2_probs)))
    mean_logprob = mean(all_logprobs)
    loss = -mean_logprob
    print(loss)

    # Using Torch's built in loss function.
    loss_fn = nn.functional.cross_entropy
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten(0, 1)
    print(logits_flat.shape) # Shape is (N, C)
    print(targets_flat.shape) # Shape is (N)
    print(loss_fn(logits_flat, targets_flat))

def try_measure_dataset_loss():
    from torch import manual_seed, device
    from llm_components import GPTModel, calc_batch_loss, calc_avg_loss_per_batch

    apple_metal_device = device("mps")

    # Tweak config for this example.
    config = GPT_CONFIG_124M
    config._dropout_rate = 0.1
    config._context_length = 256
    manual_seed(123)
    model = GPTModel(config, device=apple_metal_device)
    model.eval()

    # Read the text from the file
    with open('verdict.txt') as txt_file:
        raw_text = txt_file.read()
    
    # Split the raw text directly into train and validation set.
    token_split_idx = int(len(raw_text) * 0.9) # train/validation split of 90/10
    train_dataloader = create_dataloder_v1(raw_text[:token_split_idx], batch_size=2, max_length=config.get_context_length(), stride=config.get_context_length(),
                                           drop_last=True, shuffle=True, num_workers=0)
    test_dataloader = create_dataloder_v1(raw_text[token_split_idx:], batch_size=2, max_length=config.get_context_length(), stride=config.get_context_length(),
                                        drop_last=True, shuffle=True, num_workers=0)
    
    
    # print("Training data")
    # for inputs, targets in train_dataloader:
    #     print(inputs.shape, targets.shape)

    # print("Test data")
    # for inputs, targets in test_dataloader:
    #     print(inputs.shape, targets.shape)

    print("Training loss (Avg per batch)")
    print(calc_avg_loss_per_batch(train_dataloader, model, apple_metal_device, num_batches=15))
    print("Validation loss (Avg per batch)")
    print(calc_avg_loss_per_batch(test_dataloader, model, apple_metal_device, num_batches=5))

def trying_out_a_train_loop_with_ckpt():
    from torch import manual_seed, device, save, load
    from torch.optim import AdamW
    from llm_components import GPTModel, train_model_simple, generate_text_simple, generate_text, text_ids_to_tokens, token_ids_to_text
    import tiktoken

    apple_metal_device = device("mps")
    config = GPT_CONFIG_124M
    config._context_length = 256
    # Tweak config for this example.
    manual_seed(123)

    tokenizer = tiktoken.encoding_for_model("gpt-2")

    with open('verdict.txt') as txt_file:
        raw_text = txt_file.read()

    
    # Split the raw text directly into train and validation set.
    token_split_idx = int(len(raw_text) * 0.9) # train/validation split of 90/10
    train_dataloader = create_dataloder_v1(raw_text[:token_split_idx], batch_size=2, max_length=config.get_context_length(), stride=config.get_context_length(),
                                           drop_last=True, shuffle=True, num_workers=0)
    test_dataloader = create_dataloder_v1(raw_text[token_split_idx:], batch_size=2, max_length=config.get_context_length(), stride=config.get_context_length(),
                                        drop_last=True, shuffle=True, num_workers=0)
    

    

    start_str = "Every effort moves you"
    start_str_token_ids = text_ids_to_tokens(start_str, tokenizer)
    
    # Optimizer and modelsetup
    model = GPTModel(config, device=apple_metal_device)
    optimizer = AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 50

    gen_text_tokenids = generate_text_simple(start_str_token_ids, model, config, 10, apple_metal_device)
    print(token_ids_to_text(gen_text_tokenids, tokenizer))

    train_model_simple(model, optimizer, train_dataloader, test_dataloader, apple_metal_device, num_epochs)

    model.eval()
    gen_text_tokenids = generate_text_simple(start_str_token_ids, model, config, 10, apple_metal_device)
    print(token_ids_to_text(gen_text_tokenids, tokenizer))

    # gen_text_tokenids = generate_text(start_str_token_ids, model, config, 10, device=apple_metal_device, top_k=25, temp=1.5)
    # print(token_ids_to_text(gen_text_tokenids, tokenizer))

    model.train() # Put in train mode so that the ALL states are saved. This is required if the model needs more training 
    print("......Saving Checkpoints......")
    save(model.state_dict(), './checkpoints/model.pth')
    # The optimizer stores additional state related to the parameters e.g. a history of values.
    # Therefore, if training is to continue then the optimizer state must also be saved.
    # save(optimizer.state_dict(), './checkpoints/optimizer.pth')

    print("......Loading Checkpoints......")
    saved_model = GPTModel(config, device=apple_metal_device)
    model_checkpoint = load('./checkpoints/model.pth', map_location=apple_metal_device)
    saved_model.load_state_dict(model_checkpoint)

    saved_model.eval()
    gen_text_tokenids = generate_text_simple(start_str_token_ids, saved_model, config, 10, device=apple_metal_device)
    print(token_ids_to_text(gen_text_tokenids, tokenizer))

def try_loading_a_checkpoint():
    from torch import manual_seed, device, save, load
    from torch.optim import AdamW
    from llm_components import GPTModel, train_model_simple, generate_text_simple, generate_text, text_ids_to_tokens, token_ids_to_text
    import tiktoken

    apple_metal_device = device("mps")
    config = GPT_CONFIG_124M
    config._context_length = 256
    # Tweak config for this example.
    manual_seed(123)

    tokenizer = tiktoken.encoding_for_model("gpt-2")

    start_str = "and then"
    start_str_token_ids = text_ids_to_tokens(start_str, tokenizer)

    print("......Loading Checkpoints......")
    saved_model = GPTModel(config, device=apple_metal_device)
    model_checkpoint = load('./checkpoints/model.pth', map_location=apple_metal_device)
    saved_model.load_state_dict(model_checkpoint)

    saved_model.eval()
    gen_text_tokenids = generate_text_simple(start_str_token_ids, saved_model, config, 10, device=apple_metal_device)
    print(f"{token_ids_to_text(gen_text_tokenids, tokenizer)}")
    

def load_weights_into_gpt(gpt: GPTModel, params):
    import torch, numpy as np
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    
    gpt.pos_embed.weight = assign(gpt.pos_embed.weight, params['wpe'])
    gpt.tok_embed.weight = assign(gpt.tok_embed.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.txms_blocks[b].attn_head.W_query.weight = assign(
            gpt.txms_blocks[b].attn_head.W_query.weight, q_w.T)
        gpt.txms_blocks[b].attn_head.W_key.weight = assign(
            gpt.txms_blocks[b].attn_head.W_key.weight, k_w.T)
        gpt.txms_blocks[b].attn_head.W_value.weight = assign(
            gpt.txms_blocks[b].attn_head.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.txms_blocks[b].attn_head.W_query.bias = assign(
            gpt.txms_blocks[b].attn_head.W_query.bias, q_b)
        gpt.txms_blocks[b].attn_head.W_key.bias = assign(
            gpt.txms_blocks[b].attn_head.W_key.bias, k_b)
        gpt.txms_blocks[b].attn_head.W_value.bias = assign(
            gpt.txms_blocks[b].attn_head.W_value.bias, v_b)

        gpt.txms_blocks[b].attn_head.out_proj.weight = assign(
            gpt.txms_blocks[b].attn_head.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.txms_blocks[b].attn_head.out_proj.bias = assign(
            gpt.txms_blocks[b].attn_head.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.txms_blocks[b].feed_fwd.layers[0].weight = assign(
            gpt.txms_blocks[b].feed_fwd.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.txms_blocks[b].feed_fwd.layers[0].bias = assign(
            gpt.txms_blocks[b].feed_fwd.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.txms_blocks[b].feed_fwd.layers[2].weight = assign(
            gpt.txms_blocks[b].feed_fwd.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.txms_blocks[b].feed_fwd.layers[2].bias = assign(
            gpt.txms_blocks[b].feed_fwd.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.txms_blocks[b].l_norm_1.scale = assign(
            gpt.txms_blocks[b].l_norm_1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.txms_blocks[b].l_norm_1.shift = assign(
            gpt.txms_blocks[b].l_norm_1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.txms_blocks[b].l_norm_2.scale = assign(
            gpt.txms_blocks[b].l_norm_2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.txms_blocks[b].l_norm_2.shift = assign(
            gpt.txms_blocks[b].l_norm_2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.main_norm.scale = assign(gpt.main_norm.scale, params["g"])
    gpt.main_norm.shift = assign(gpt.main_norm.shift, params["b"])
    gpt.logit_head.weight = assign(gpt.logit_head.weight, params["wte"])


def try_download_gpt2():
    from torch import device, as_tensor, tensor, nn, manual_seed, no_grad
    from llm_components import (
            GPTModel, generate_text_simple, generate_text,
            text_ids_to_tokens, token_ids_to_text,
            download_and_load_gpt2, load_gpt2_params_from_tf_ckpt, GPT_CONFIG_124M)
    import tiktoken
    import numpy as np
    import json
    import os

    def assign_from_src(target, source, prop_name):
        tgt_prop = getattr(target, prop_name)
        assert source.shape == tgt_prop.shape, "Error mismatch of shape between source and target"
        setattr(target, prop_name, nn.Parameter(tensor(source)))

    # The following three lines only need to run once for the download.
    model_size="124M"
    models_dir="./checkpoints/gpt2"
    
    # settings, params = download_and_load_gpt2(model_size, models_dir)
    # print(settings)
    # print(params.keys())

    settings = json.load(open(os.path.join(models_dir, model_size, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(os.path.join(models_dir, model_size, "model.ckpt"), settings) # model.ckpt does not exist but internally will be linked to ckpt.data-.....

    apple_metal_device = device('cpu')
    tokenizer = tiktoken.encoding_for_model('gpt-2')

    config = GPT_CONFIG_124M
    config._qkv_bias= True # GPT 2 use biases on Query, key and value matrices. This was stopped in subsequent models.
    manual_seed(123)

    gpt_model = GPTModel(config) # Keep model in CPU for now
    gpt_model.eval()

    PROP_WEIGHT = "weight"
    PROP_BIAS = "bias"
    PROP_SCALE = "scale"
    PROP_SHIFT = "shift"

    # # Loading procedure starts here.
    # # Step 1 - Assign pos and token embedding weights.
    # assign_from_src(gpt_model.pos_embed, params['wpe'], PROP_WEIGHT)
    # assign_from_src(gpt_model.tok_embed, params['wte'], PROP_WEIGHT)

    # # Step 2 - Assign the weights for the output head and the final norm
    # assign_from_src(gpt_model.logit_head, params["wte"], PROP_WEIGHT) # Same as token embedding as the orginal GPT 2 used weight tying.
    # assign_from_src(gpt_model.main_norm, params["g"], PROP_SCALE)
    # assign_from_src(gpt_model.main_norm, params["b"], PROP_SHIFT)
    
    # # Step 3 - Assign the weights to the transformer blocks.
    # """
    # A txfrmer block is made of the following blocks
    # layer norm ---> CMHA (Q, K, V with biases for gpt2, out_proj_head) --> dropout -----> layer norm --> FF (2 linear layers) --> dropout ---->
    # ________________________________________________________________________________^ |_____________________________________________________^
    # """
    # for i, block in enumerate(params["blocks"]):
    #     # Each block has the following keys dict_keys(['attn', 'ln_1', 'ln_2', 'mlp'])
    #     # Each "attn" el in turn has the following keys dict_keys(['c_attn', 'c_proj'])
    #     # In turn c_attn and c_proj have the following keys dict_keys(['b', 'w'])
       
    #     # Each mlp has the following keys dict_keys(['c_fc', 'c_proj'])
    #     # Each ln layer has the following keys dict_keys(['b', 'g'])
    #     # c_attn needs to be split into Q, K and V
    #     tgt_txfm_block: TransformerBlock = gpt_model.txms_blocks[i]

    #     ### Assign attention head params
    #     # Split into Q, K and V weights and assign
    #     qw, kw, vw = np.split(block['attn']['c_attn']['w'], 3, axis=-1)
    #     assign_from_src(tgt_txfm_block.attn_head.W_query, qw.T, PROP_WEIGHT)
    #     assign_from_src(tgt_txfm_block.attn_head.W_key, kw.T, PROP_WEIGHT)
    #     assign_from_src(tgt_txfm_block.attn_head.W_value, vw.T, PROP_WEIGHT)

    #     # Split into Q, K and V biases and assign
    #     qb, kb, vb = np.split(block['attn']['c_attn']['b'], 3, axis=-1)
    #     assign_from_src(tgt_txfm_block.attn_head.W_query, qb, PROP_BIAS)
    #     assign_from_src(tgt_txfm_block.attn_head.W_key, kb, PROP_BIAS)
    #     assign_from_src(tgt_txfm_block.attn_head.W_value, vb, PROP_BIAS)

    #     # assign out proj weights
    #     ow = block['attn']['c_proj']['w']
    #     assign_from_src(tgt_txfm_block.attn_head.W_out, ow.T, PROP_WEIGHT)

    #     # assign out proj bias
    #     ob = block['attn']['c_proj']['b']
    #     assign_from_src(tgt_txfm_block.attn_head.W_out, ob, PROP_BIAS)

    #     ### Assign feed_fwd params
    #     # assign feed fwd weights on the input layer
    #     ff_fcw = block['mlp']['c_fc']['w']
    #     assign_from_src(tgt_txfm_block.feed_fwd.layers[0], ff_fcw.T, PROP_WEIGHT)

    #     # assign feed fwd bias on the input layer
    #     ff_fcb = block['mlp']['c_fc']['b']
    #     assign_from_src(tgt_txfm_block.feed_fwd.layers[0], ff_fcb, PROP_BIAS)

    #      # assign feed fwd weights on the output layer
    #     ff_fpjw = block['mlp']['c_proj']['w']
    #     assign_from_src(tgt_txfm_block.feed_fwd.layers[2], ff_fpjw.T, PROP_WEIGHT)

    #     # assign feed fwd bias on the output layer
    #     ff_fpjb = block['mlp']['c_proj']['b']
    #     assign_from_src(tgt_txfm_block.feed_fwd.layers[2], ff_fpjb, PROP_BIAS)

    #     ### Assign Layer norm params.
    #     lnorm_1 = block['ln_1']
    #     assign_from_src(tgt_txfm_block.l_norm_1, lnorm_1['g'], PROP_SCALE)
    #     assign_from_src(tgt_txfm_block.l_norm_1, lnorm_1['b'], PROP_SHIFT)

    #     lnorm_2 = block['ln_2']
    #     assign_from_src(tgt_txfm_block.l_norm_2, lnorm_2['g'], PROP_SCALE)
    #     assign_from_src(tgt_txfm_block.l_norm_2, lnorm_2['b'], PROP_SHIFT)

    load_weights_into_gpt(gpt_model, params)
    # gpt_model = gpt_model.to(apple_metal_device)

    start_str = "Every effort moves you"
    start_str_token_ids = text_ids_to_tokens(start_str, tokenizer)

    gen_text_tokenids = generate_text_simple(start_str_token_ids, gpt_model, config, 10)
    print(f"{token_ids_to_text(gen_text_tokenids, tokenizer)}")


if __name__ == '__main__':
    try_download_gpt2()
    # try_loading_a_checkpoint()
    # trying_out_a_train_loop_with_ckpt()
    # try_measure_dataset_loss()
    # try_measure_loss()
    # build_gpt_2()
    # build_a_txfm_block()
    # try_layer_norm()
    # building_causal_multiheaded_attention()
    # building_causal_attention_wdropout()
    # build_compact_attention_layers()
    # print('+++++++++++++++++++++++++++')
    # building_weighted_self_attention()
    # building_simple_self_attention()
    # testing_simple_embedding()
    # test_data_sampling()
    # test_custom_tokenizer()