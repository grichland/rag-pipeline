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

