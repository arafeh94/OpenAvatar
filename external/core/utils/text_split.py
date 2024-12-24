import re


def split_text(text, max_length=400):
    # Regular expression to split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            yield current_chunk
            current_chunk = sentence

    if current_chunk:
        yield current_chunk
