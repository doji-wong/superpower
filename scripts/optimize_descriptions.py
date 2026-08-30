import json
import os
import re
import math
from collections import Counter
from pathlib import Path

STOP = {
    'a', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'before', 'by', 'for',
    'from', 'in', 'into', 'is', 'it', 'its', 'my', 'need', 'needs', 'of', 'on',
    'or', 'our', 'so', 'that', 'the', 'them', 'this', 'to', 'use', 'want',
    'we', 'when', 'with', 'you', 'your', 'help', 'me', 'i',
}

def stem(t):
    for suf in ['ally', 'ing', 'ed', 'es', 'al']:
        if len(t) > len(suf) + 3 and t.endswith(suf):
            t = t[:-len(suf)]
            break
    if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
        t = t[:-1]
    if len(t) > 4 and t.endswith('e'):
        t = t[:-1]
    if len(t) > 4 and t[-1] == t[-2] and t[-1] not in 'aeiou':
        t = t[:-1]
    if len(t) > 3 and t.endswith('y'):
        t = t[:-1] + 'i'
    return t

def tokenize(text):
    text = re.sub(r'[^a-z0-9\s-]', ' ', text.lower())
    tokens = [stem(t) for t in re.split(r'[\s-]+', text) if len(t) > 2 and t not in STOP]
    return tokens

class TFIDFCatalog:
    def __init__(self):
        self.docs = {}
        self.idf = {}
        self.n = 0
    
    def add_skill(self, name, description):
        name_tokens = tokenize(name.replace('-', ' '))
        tokens = name_tokens + name_tokens + tokenize(description)
        self.docs[name] = Counter(tokens)
        
    def build_idf(self):
        df = Counter()
        self.n = len(self.docs)
        for tf in self.docs.values():
            for term in tf.keys():
                df[term] += 1
        self.idf = {term: math.log(1 + self.n / (1 + count)) for term, count in df.items()}
        
    def get_vec(self, tf):
        return {term: count * self.idf.get(term, 0) for term, count in tf.items()}
        
    def cosine(self, a, b):
        dot = sum(a[t] * b.get(t, 0) for t in a)
        na = sum(w * w for w in a.values())
        nb = sum(w * w for w in b.values())
        if not na or not nb: return 0
        return dot / (math.sqrt(na) * math.sqrt(nb))
        
    def rank(self, prompt):
        pv = self.get_vec(Counter(tokenize(prompt)))
        scores = []
        for name, tf in self.docs.items():
            scores.append((name, self.cosine(pv, self.get_vec(tf))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

print("TF-IDF module ready.")
