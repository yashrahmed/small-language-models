import re

def load_text():
    with open('verdict.txt', 'r') as book:
        text = book.read()
        return text

def tokenize(text):
    tokens = re.split(r'([,.:;?_!\"()\']|--|\s)', text) # Use capture groups for retaining delimiters.
    return [token.strip() for token in tokens if token.strip()]

    

if __name__ == '__main__':
    text = load_text()
    tokens = tokenize(text)
    print(len(tokens))
    print(',' in tokens)