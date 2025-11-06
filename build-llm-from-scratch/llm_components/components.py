import re
import tiktoken

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

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

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride) -> None:
        # Input and target ids will be a list of PyTorch Tensors
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        print(len(token_ids))

        for i in range(0, len(token_ids) - max_length, stride):
            i_max = i + max_length
            self.input_ids.append(Tensor(token_ids[i:i_max]))
            self.target_ids.append(Tensor(token_ids[i+1:i_max+1]))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloder_v1(text, max_length=256, stride=128, batch_size=4, drop_last=True, shuffle=True, num_workers=0):
    tokenizer = tiktoken.encoding_for_model("gpt-2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride) # Dataset inside a dataloader
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers)



