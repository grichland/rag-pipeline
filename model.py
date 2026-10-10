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

# Step 11 - load_embedding_model
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    # Load a pretrained model
    model = SentenceTransformer(model_name)
    return model

# Step 12 - embed_text
def embed_text(model, text):
    # TODO: Return a 1D float32 numpy embedding vector for the given text string.
    return model.encode(text)

# Step 13 - embed_chunks
import numpy as np
def embed_chunks(model, chunks, batch_size=32):
    """Batch-embed a list of chunk strings or chunk dicts into a 2D float32 matrix."""
    # TODO: normalize chunk inputs to strings, encode in batches, return (n, d) float32 array
    if not chunks:
        return np.empty((0,model.get_embedding_dimension())).astype(np.float32)
    if isinstance(chunks[0],dict):
        chunks = [x['text'] for x in chunks]
    return model.encode(chunks, batch_size=batch_size)

# Step 14 - l2_normalize
import numpy as np

def l2_normalize(matrix):
    # TODO: rescale each row of `matrix` to unit L2 norm, leaving all-zero rows unchanged.
    norms = np.linalg.norm(matrix, axis=1)
    nonzero_rows = norms!=0
    matrix[nonzero_rows,:] = matrix[nonzero_rows,:] / norms[nonzero_rows].reshape(-1,1)
    return matrix

# Step 15 - save_corpus
import json
from pathlib import Path

import numpy as np


def save_corpus(embeddings, chunks, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    embeddings_path = directory / "embeddings.npy"
    chunks_path = directory / "chunks.json"

    np.save(embeddings_path, np.asarray(embeddings))

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    reloaded_embeddings = np.load(embeddings_path)
    with chunks_path.open("r", encoding="utf-8") as f:
        reloaded_chunks = json.load(f)

    out = {}
    out['embeddings'] = reloaded_embeddings
    out['chunks'] = reloaded_chunks
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

# Step 19 - retrieve
def retrieve(query, model, chunk_matrix, chunks, k):
     
    # TODO: embed the query, 
    emb_q = embed_text(model,query) 

    # TODO: score it against chunk_matrix, 
    scores = cosine_similarity_search(emb_q,chunk_matrix)


    # TODO: return top-k (chunk, score) pairs.
    return top_k_chunks(scores,chunks,k)

# Step 20 - build_faiss_index
import faiss
def build_faiss_index(chunk_matrix):
    # TODO: build a FAISS inner-product index and add all rows of chunk_matrix to it
    fidx = faiss.IndexFlatIP(chunk_matrix.shape[1])
    fidx.add(chunk_matrix.astype(np.float32))
    return fidx

