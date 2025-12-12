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
    from llm_components import GPTModel, generate_text_simple, text_to_token_ids, token_ids_to_text

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
    batch = text_to_token_ids(sentence3, tokenizer)

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
    output_tokens = generate_text_simple(batch, model, GPT_CONFIG_124M.get_context_length(), 6)
    print(token_ids_to_text(output_tokens, tokenizer))
    # print(tokenizer.decode(output_tokens[0, :].tolist()))

def try_measure_loss():
    import torch
    from torch import (nn, softmax, tensor, manual_seed, triu, ones, zeros, inf, randn, mean, var, sqrt, tanh, pi, pow, linspace, rand, arange, argmax, cat,
                       log)
    import tiktoken
    from llm_components import GPTModel, generate_text_simple, text_to_token_ids, token_ids_to_text
    
    
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
    from llm_components import GPTModel, train_model_simple, generate_text_simple, generate_text, text_to_token_ids, token_ids_to_text
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
    start_str_token_ids = text_to_token_ids(start_str, tokenizer)
    
    # Optimizer and modelsetup
    model = GPTModel(config, device=apple_metal_device)
    optimizer = AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 50

    gen_text_tokenids = generate_text_simple(start_str_token_ids, model, config.get_context_length(), 10, device=apple_metal_device, model_type="custom")
    print(token_ids_to_text(gen_text_tokenids, tokenizer))

    train_model_simple(model, optimizer, train_dataloader, test_dataloader, apple_metal_device, num_epochs)

    model.eval()
    gen_text_tokenids = generate_text_simple(start_str_token_ids, model, config.get_context_length(), 10, device=apple_metal_device, model_type="custom")
    print(token_ids_to_text(gen_text_tokenids, tokenizer))

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
    gen_text_tokenids = generate_text_simple(start_str_token_ids, model, config.get_context_length(), 10, device=apple_metal_device, model_type="custom")
    print(token_ids_to_text(gen_text_tokenids, tokenizer))

def try_loading_a_checkpoint():
    from torch import manual_seed, device, save, load
    from torch.optim import AdamW
    from llm_components import GPTModel, train_model_simple, generate_text_simple, generate_text, text_to_token_ids, token_ids_to_text
    import tiktoken

    apple_metal_device = device("mps")
    config = GPT_CONFIG_124M
    config._context_length = 256
    # Tweak config for this example.
    manual_seed(123)

    tokenizer = tiktoken.encoding_for_model("gpt-2")

    start_str = "and then"
    start_str_token_ids = text_to_token_ids(start_str, tokenizer)

    print("......Loading Checkpoints......")
    saved_model = GPTModel(config, device=apple_metal_device)
    model_checkpoint = load('./checkpoints/model.pth', map_location=apple_metal_device)
    saved_model.load_state_dict(model_checkpoint)

    saved_model.eval()
    gen_text_tokenids = generate_text_simple(start_str_token_ids, saved_model, config.get_context_length(), 10, device=apple_metal_device)
    print(f"{token_ids_to_text(gen_text_tokenids, tokenizer)}")

def try_download_gpt2():
    from torch import device, manual_seed
    from transformers import GPT2LMHeadModel
    from llm_components import (
            GPTModel, generate_text_simple, generate_text, load_weights_from_hfmodel,
            text_to_token_ids, token_ids_to_text,
            GPT_CONFIG_124M)
    import tiktoken

    apple_metal_device = device('mps')
    tokenizer = tiktoken.encoding_for_model('gpt-2')
   
    manual_seed(123)

    config = GPT_CONFIG_124M
    config._qkv_bias= True # GPT 2 use biases on Query, key and value matrices. This was stopped in subsequent models.
    gpt_model = GPTModel(config)

    # Download and initialize GPT model from Huggingface hub.
    gpt_hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", cache_dir="./checkpoints")
    gpt_hf.eval()

    load_weights_from_hfmodel(gpt_model, gpt_hf)
    gpt_model.eval()
    gpt_model = gpt_model.to(apple_metal_device)

    start_str = "Every effort moves you"
    start_str_token_ids = text_to_token_ids(start_str, tokenizer)
    gen_text_tokenids = generate_text_simple(start_str_token_ids, gpt_model, config.get_context_length(), 25, model_type="custom", device=apple_metal_device)
    print(f"{token_ids_to_text(gen_text_tokenids, tokenizer)}")

def try_loading_classification_dataset():
    import pandas as pd
    dataset = pd.read_csv('./datasets/sms+spam+collection/SMSSpamCollection', sep='\t', names=["label", "text"])
    print(dataset.head())

    def gen_balanaced_ds(ds_ref):
        spam_ds = ds_ref[ds_ref["label"]=="spam"]
        num_spam = spam_ds.shape[0]
        ham_ds_smp = ds_ref[ds_ref["label"]=="ham"].sample(num_spam, random_state=123)
        result = pd.concat([ham_ds_smp, spam_ds])
        result["label"] = result["label"].map({"ham": 0, "spam": 1})
        return result
    
    def train_test_val_split(ds_ref, train_frac, validation_frac):
        ds_ref = ds_ref.sample(frac=1, random_state=123).reset_index(drop=True) # Shuffle the entire data frame
        dz_sz = len(ds_ref)
        train_idx_end = int(train_frac * dz_sz)
        val_idx_end = train_idx_end + int(validation_frac * dz_sz)
        return ds_ref[:train_idx_end], ds_ref[train_idx_end:val_idx_end], ds_ref[val_idx_end:]

    
    balanced_dataset = gen_balanaced_ds(dataset)
    print(balanced_dataset["label"].value_counts())

    train, val, test = train_test_val_split(balanced_dataset, 0.7, 0.1)
    print(len(train))
    print(len(val))
    print(len(test))

    train.to_csv('./datasets/train.csv', index=None)
    val.to_csv('./datasets/val.csv', index=None)
    test.to_csv('./datasets/test.csv', index=None)

def try_setup_for_hamspam(mode='train'):
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    import tiktoken
    from torch import tensor, long, device, nn, no_grad, optim, save, load, argmax
    from llm_components import load_gpt2_pretrained, calc_acc_binary, train_model_simple_binary

    # Define a dataset subclass with truncation/padding functionality
    class SpamDataset(Dataset):
        def __init__(self, csv_path, tokenizer, pad_token_id=50256, max_length=None) -> None:
            super().__init__()
            self.data = pd.read_csv(csv_path)
            self.encoded_texts = [tokenizer.encode(text) for text in self.data["text"]]
            # calculate max length and pad
            
            if max_length is not None:
                self.max_len = max_length
                # Truncate
                self.encoded_texts = [ text[:self.max_len] for text in self.encoded_texts]
            else:
                self.max_len = self._max_encoded_length()
                # Pad
                self.encoded_texts = [ text + [pad_token_id] * (self.max_len - len(text)) for text in self.encoded_texts]
        
        def __getitem__(self, idx):
            return (
                tensor(self.encoded_texts[idx], dtype=long),
                tensor(self.data.iloc[idx]["label"], dtype=long)
            )

        def __len__(self):
            return len(self.data)

        def _max_encoded_length(self):
            max_len = 0
            for text in self.encoded_texts:
                max_len = max(max_len, len(text))
            return max_len
    
    def classify_text(text, model, tokenizer, device, pad_token_id=50256):
        # Tokenize
        text_tokens = tokenizer.encode(text)
        # Adjust and align to context length
        max_len = model.pos_emb.weight.shape[0]
        text_tokens = text_tokens[:max_len]
        text_tokens = text_tokens + [pad_token_id] * (max_len - len(text_tokens))
        text_tokens = tensor(text_tokens, device=device).unsqueeze(0)
        # Run it through the model
        model.eval()
        with no_grad():
            logits = model(text_tokens)
            logits = logits[:, -1, :]
            pred = argmax(logits, dim=-1).item()
        return "SPAM" if pred == 1 else "HAM"
        
    apple_metal_device = device("mps")
    tokenizer = tiktoken.encoding_for_model("gpt2")
    batch_size = 8

    # Instantiate train, val and test Dataset objects.
    train_dataset = SpamDataset("./datasets/train.csv", tokenizer)
    val_dataset = SpamDataset("./datasets/val.csv", tokenizer)
    test_dataset = SpamDataset("./datasets/test.csv", tokenizer)

    # Create dataloader objects for train, val and test.
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Load weights from a Huggingface pretrained model and verify via text generation.
    if mode == 'train':
        gpt_model_config, gpt_model = load_gpt2_pretrained(apple_metal_device, 123)
        optimizer = optim.AdamW(gpt_model.parameters(), lr=5e-5, weight_decay=0.1)

        # Freeze the model parameters
        for param in gpt_model.parameters():
            param.requires_grad = False # Disable gradient tracking.
        
        # Replace the 50257 way classification head with a 2 way classification head.
        new_out_head = nn.Linear(gpt_model_config.get_embed_dim(), 2, device=apple_metal_device) # requires_grad is set to True by default
        gpt_model.out_head = new_out_head

        # Unfreeze the final norm and the last txfm block (This improves predictive performance as per the book)
        for params in gpt_model.final_norm.parameters():
            params.requires_grad = True
        for params in gpt_model.trf_blocks[-1].parameters():
            params.requires_grad = True
        
        train_model_simple_binary(gpt_model, optimizer, train_dataloader, val_dataloader, apple_metal_device, num_epochs=10, eval_batch_size=5, eval_batch_interval=50, verbose=True)


        gpt_model.train() # Put in train mode so that the ALL states are saved. This is required if the model needs more training 
        print("......Saving Checkpoints......")
        save(gpt_model.state_dict(), './checkpoints/bin_model.pth')
    elif mode == 'eval' or mode == 'predict':
        config = GPT_CONFIG_124M
        config._qkv_bias = True
        config._embed_dim
        
        print("......Loading Checkpoints......")
        saved_model = GPTModel(config, device=apple_metal_device)
        new_out_head = nn.Linear(config.get_embed_dim(), 2, device=apple_metal_device) # requires_grad is set to True by default
        saved_model.out_head = new_out_head

        model_checkpoint = load('./checkpoints/bin_model.pth', map_location=apple_metal_device)
        saved_model.load_state_dict(model_checkpoint)
        saved_model.eval()

        if mode == 'eval':
            with no_grad():
                train_acc = calc_acc_binary(train_dataloader, saved_model, apple_metal_device, num_batches=len(train_dataloader))
                val_acc = calc_acc_binary(val_dataloader, saved_model, apple_metal_device, num_batches=len(val_dataloader))
                test_acc = calc_acc_binary(test_dataloader, saved_model, apple_metal_device, num_batches=len(test_dataloader))

            print(f"Train acc = {train_acc*100:.2f}% | Val acc = {val_acc*100:.2f}% | Test acc = {test_acc*100:.2f}%")
            # Train acc = 98.37% | Val acc = 98.61% | Test acc = 96.28%
        elif mode == "predict":
            text = "You have been specially selected to receive a 2000 pound award! Click here to apply!"
            print(text)
            print(classify_text(text, saved_model, tokenizer, apple_metal_device))
            text = "Hey man! I just wanted to check if you'd be coming in tonight. Let me know asap!"
            print(text)
            print(classify_text(text, saved_model, tokenizer, apple_metal_device))

def try_setup_for_instruct_finetuning():
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    import tiktoken
    from torch import tensor, long, device, nn, no_grad, manual_seed, optim, save, load, argmax
    from llm_components import load_gpt2_pretrained, text_to_token_ids, token_ids_to_text, generate_text_simple, calc_avg_loss_per_batch_binary, calc_acc_binary, train_model_simple_binary
    import json

    def read_dataset():
        with open('./datasets/instruction-data.json', 'r') as json_file:
            dataset = json.loads(json_file.read())
        return dataset

    def format_input_entry(entry):
        instruct_text = ("Below is an instruction that describes a task. Write a response that appropriately completes this request."
                         f"\n\n### Instruction:\n{entry['instruction']}")
        input_text = (f"\n\n### Input:\n{entry['input']}" if entry['input'] else '')
        return instruct_text + input_text
    
    def train_test_val_split(ds_ref, train_frac, validation_frac):
        total = len(ds_ref)
        train_idx_end = int(train_frac * total)
        val_idx_end = train_idx_end + int(validation_frac * total)
        return ds_ref[:train_idx_end], ds_ref[train_idx_end:val_idx_end], ds_ref[val_idx_end:]



    dataset = read_dataset()
    # print(format_input_entry(dataset[50]))
    train_split, val_split, test_split = train_test_val_split(dataset, 0.85, 0.05)
    print(len(train_split))
    print(len(val_split))
    print(len(test_split))



if __name__ == '__main__':
    try_setup_for_instruct_finetuning()
    # try_setup_for_hamspam("predict")
    # try_loading_classification_dataset()
    # try_download_gpt2()
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