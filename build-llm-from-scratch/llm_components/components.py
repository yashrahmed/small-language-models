import re

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



