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

