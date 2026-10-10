"""
RAG Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_text_file
def load_text_file(path):
    with open(path,'r',encoding='utf-8') as f:
        text = f.read()
    return text

# Step 2 - load_text_directory
import os
def load_text_directory(directory):
    # TODO: read every .txt file in `directory` and return their contents as a list of strings
    out = []
    for f in sorted(os.listdir(directory)):
        if '.txt' in f:
            path = os.path.join(directory,f)
            single_txt = load_text_file(path)
            out.append(single_txt)

    return out

# Step 3 - extract_text_from_html
from html.parser import HTMLParser
import re


class _VisibleTextParser(HTMLParser):
    BLOCK_TAGS = {
        "address", "article", "blockquote", "br", "dd", "div", "dl",
        "dt", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header",
        "hr", "li", "main", "ol", "p", "section", "table", "td", "th",
        "tr", "ul",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden_tags = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden_tags.append(tag)
        elif not self.hidden_tags and tag in self.BLOCK_TAGS:
            self.parts.append("")

    def handle_endtag(self, tag):
        if tag in self.hidden_tags:
            # Remove the matching hidden tag and any malformed nested entries.
            index = len(self.hidden_tags) - 1 - self.hidden_tags[::-1].index(tag)
            del self.hidden_tags[index:]
        elif not self.hidden_tags and tag in self.BLOCK_TAGS:
            self.parts.append("")

    def handle_data(self, data):
        if not self.hidden_tags:
            self.parts.append(data)


def extract_text_from_html(html):
    """Return visible text from an HTML string, with entities decoded."""
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()

# Step 4 - normalize_text
import re
import unicodedata

def normalize_text(text):
    # TODO: NFKC-normalize the text and collapse runs of whitespace into single spaces.
    text = unicodedata.normalize("NFKC",text)
    text = re.sub(f"\s+", " ", text)
    text = text.strip()
    return text

# Step 5 - make_document
def make_document(text, source, title):
    # TODO: wrap text with source and title metadata into a document dict.
    return {
        "text": text,
        "source": source,
        "title": title
    }

# Step 6 - chunk_fixed_size
def chunk_fixed_size(text, chunk_size):
    # TODO: split text into consecutive non-overlapping chunks of length chunk_size
    out = []

    start = 0
    end = chunk_size
    while start < len(text):
        end = min(end,len(text))
        chunk = text[start:end]
        out.append(chunk)
        start += chunk_size
        end = start + chunk_size
    
    return out

# Step 7 - chunk_by_tokens
def chunk_by_tokens(text, tokenizer, max_tokens):
    # TODO: split text into chunks of at most max_tokens token ids using the tokenizer
    encoded = tokenizer.encode(text)
    N_tokens = len(encoded)
    out = []

    for start in range(0 , N_tokens, max_tokens):

        encoded_chunk = encoded[start:start+max_tokens]
        decoded_chunk = tokenizer.decode(encoded_chunk)
        out.append(decoded_chunk)
    return out

# Step 8 - chunk_by_sentences
import re

def chunk_by_sentences(text, max_chars):
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text or not text.strip():
        return []

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]

    chunks = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()

        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks

# Step 9 - chunk_with_overlap
def chunk_with_overlap(text, chunk_size, overlap):
    # TODO: return sliding-window chunks of length chunk_size sharing `overlap` chars
    
    i = 0
    n = len(text)
    out = []
    while i < n:
        out.append(
            text[
                (i):(i+chunk_size)
            ]
        )

        i += (chunk_size - overlap)
    
    return out

# Step 10 - attach_chunk_metadata
def attach_chunk_metadata(chunks, source):
    # TODO: wrap each chunk string with source, position, and chunk_id metadata.
    
    out = []

    for i,chunk in enumerate(chunks):
        d = {
            "text": chunk,
            "source": source,
            "position": i,
            "chunk_id": f"{source}::{i}"
        }
        out.append(d)
    return out

# Step 16 - cosine_similarity_search
import numpy as np

def cosine_similarity_search(query_vector, chunk_matrix):
    """Cosine similarity between query_vector (d,) and each row of chunk_matrix (n,d)."""
    # TODO: compute cosine similarity between the query vector and every chunk row
    l2_q = np.linalg.norm(query_vector)
    l2_c = np.linalg.norm(chunk_matrix,axis=1)

    return chunk_matrix @ query_vector / (l2_q * l2_c)

# Step 17 - top_k_indices
import numpy as np

def top_k_indices(scores, k):
    """Return indices of the k highest scores in descending order."""
    # TODO: rank the score array and return the top-k positions as a numpy array

    return np.argsort(-scores,kind='stable')[:k]

# Step 18 - top_k_chunks
import numpy as np

def top_k_chunks(scores, chunks, k):
    # TODO: return list of (chunk, score) tuples for the top-k scores, sorted descending
    
    idx = top_k_indices(scores,k)
    out = []
    for i in idx:
        out.append(
            (chunks[i],scores[i].item())
        )
    return out

