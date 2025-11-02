import re

def load_text():
    with open('verdict.txt', 'r') as book:
        text = book.read()
        return text

def tokenize(text):
    tokens = re.split(r'([,.:;?_!\"()\']|--|\s)', text) # Use capture groups for retaining delimiters.
    return [token.strip() for token in tokens if token.strip()]

def build_vocab(tokens):
    sorted_tokens = sorted(set(tokens))
    vocab = {}
    for i, token in enumerate(sorted_tokens):
        vocab[token]= i
    return vocab
    

if __name__ == '__main__':
    text = load_text()
    from .llm_components import VocabBuilder, SimpleTokenizer, END_OF_TEXT_TOKEN
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