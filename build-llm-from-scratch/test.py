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

if __name__ == '__main__':
    text = load_text()
    test_data_sampling()

    