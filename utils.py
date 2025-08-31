
"""
WB Script Summarizer + Character Interaction Graphs
Single-file PyTorch project implementing an extractive summarizer (transformer encoder + sentence scorer)
and character interaction graphs derived from attention weights.

Features:
- Script dataset loader (plain text files; scene/sentence split)
- Optionally use HuggingFace tokenizer if available, else a simple whitespace tokenizer + vocab builder
- Transformer encoder built from PyTorch primitives (MultiheadAttention) that returns attention weights
- Extractive summarization by scoring sentences and selecting top-k
- Character extraction (naive heuristics) and a function to build an interaction graph from attention
- Training loop (supervised if sentence labels are provided) and inference utilities

Usage examples:
  # Train (requires dataset with .txt scripts and optional .labels files containing important sentence indices)
  python wb_script_summarizer.py --mode train --data_dir ./scripts --epochs 5 --save_path model.pt

  # Summarize a script
  python wb_script_summarizer.py --mode infer --script ./scripts/foo.txt --top_k 3 --out summary.txt

  # Build character interaction graph
  python wb_script_summarizer.py --mode graph --script ./scripts/foo.txt --out graph.png

Dependencies:
  torch, numpy, tqdm, networkx, matplotlib
  (optional) transformers for a better tokenizer

This file is intentionally self-contained for prototyping; adapt for production (distributed training, better tokenizers,
larger models, checkpointing, evaluation metrics, data augmentation).
"""

import os
import re
import math
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# visualization
import networkx as nx
import matplotlib.pyplot as plt

# Try to import HuggingFace tokenizer (optional)
try:
    from transformers import AutoTokenizer
    HF_TOKENIZER_AVAILABLE = True
except Exception:
    HF_TOKENIZER_AVAILABLE = False


# -----------------------------
# Utilities: text processing
# -----------------------------

def sentence_split(text: str) -> List[str]:
    """Naive sentence splitter. For better results use an NLP sentence boundary detector."""
    # split on line breaks or punctuation followed by space + capital
    text = text.replace('\r', '\n')
    # keep paragraphs and explicit breaks
    raw_sentences = re.split(r"(?<=[.!?\n])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def extract_characters(text: str, min_count: int = 2) -> List[str]:
    """Naive character extraction: look for lines in scripts that start with UPPERCASE words (common in screenplays)
    or frequent capitalized tokens. Returns characters that appear at least min_count times.
    """
    # lines that are mostly uppercase and not too long
    candidates = []
    for line in text.splitlines():
        line_strip = line.strip()
        if 1 <= len(line_strip) <= 40 and line_strip.isupper():
            # take the first token as the character name
            name = line_strip.split()[0]
            candidates.append(name)

    # fallback: collect capitalized words and count frequency
    words = re.findall(r"\b[A-Z][A-Za-z'\-]{1,20}\b", text)
    words = [w for w in words if not w.isupper()]  # avoid tokens already counted
    candidates += words

    freq = Counter(candidates)
    chars = [name for name, cnt in freq.items() if cnt >= min_count]
    # sort by frequency
    chars.sort(key=lambda n: -freq[n])
    return chars


# -----------------------------
# Tokenizer (HF optional)
# -----------------------------

class SimpleTokenizer:
    """Whitespace tokenizer with a dynamic vocab built from training data.
    Produces token ids, and can convert ids back to tokens (for debugging).
    """
    def __init__(self, vocab: Optional[Dict[str, int]] = None, unk_token='<unk>'):
        self.unk_token = unk_token
        if vocab is None:
            self.vocab = {unk_token: 0}
            self.inv_vocab = {0: unk_token}
            self.next_id = 1
        else:
            self.vocab = vocab
            self.inv_vocab = {i: t for t, i in vocab.items()}
            self.next_id = max(self.inv_vocab.keys()) + 1

    def add_tokens(self, tokens: List[str]):
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = self.next_id
                self.inv_vocab[self.next_id] = t
                self.next_id += 1

    def tokenize(self, text: str) -> List[str]:
        # very naive split; keep punctuation as tokens
        tokens = re.findall(r"\w+|[^\s\w]", text)
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        ids = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = self.next_id
                self.inv_vocab[self.next_id] = t
                self.next_id += 1
            ids.append(self.vocab.get(t, self.vocab[self.unk_token]))
        return ids

    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.inv_vocab.get(i, self.unk_token) for i in ids)


def get_tokenizer(pretrained_model_name: Optional[str] = None, texts: Optional[List[str]] = None):
    if pretrained_model_name and HF_TOKENIZER_AVAILABLE:
        tok = AutoTokenizer.from_pretrained(pretrained_model_name)
        return tok
    else:
        # build simple tokenizer from provided texts if given
        tok = SimpleTokenizer()
        if texts:
            for t in texts:
                tok.encode(t)
        return tok


# -----------------------------
# Dataset
# -----------------------------

class ScriptDataset(Dataset):
    """Loads plain-text scripts from a folder. Optionally expects a .labels JSON mapping filename->list of important sentence indices for supervised training.

    Data format:
      data_dir/
         script1.txt
         script1.labels.json  (optional) -> {"important_sentence_indices": [0, 3, 7]}
         script2.txt
    """
    def __init__(self, data_dir: str, tokenizer, max_tokens: int = 2048):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        with open(os.path.join(self.data_dir, fname), 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = sentence_split(text)
        # tokenize per sentence
        sent_token_ids = [self.tokenizer.encode(s)[:self.max_tokens] for s in sentences]
        # pad tokens per sentence to the same length? We'll handle batching in collate
        label_path = os.path.join(self.data_dir, fname.replace('.txt', '.labels.json'))
        labels = None
        if os.path.exists(label_path):
            try:
                lab = json.load(open(label_path, 'r'))
                labels = lab.get('important_sentence_indices', None)
            except Exception:
                labels = None
        return {'filename': fname, 'text': text, 'sentences': sentences, 'token_ids': sent_token_ids, 'labels': labels}


def collate_fn(batch):
    # batch is list of dicts; for training we process one document per batch (variable-length sentences)
    return batch


# -----------------------------
# Model: Transformer encoder + sentence scorer
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerEncoderLayerWithAttn(nn.Module):
    """Single encoder layer that returns self-attention weights in forward."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: (B, S, D)
        attn_output, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask, need_weights=True)
        src2 = attn_output
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # attn_weights: (B, S, S) when batch_first=True
        return src, attn_weights


class TransformerEncoderWithAttn(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_maps = []  # list of (B, S, S) per layer
        output = src
        for layer in self.layers:
            output, attn_w = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            attn_maps.append(attn_w)
        # attn_maps: [L x (B, S, S)]
        return output, attn_maps


class SummarizerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, nhead=8, max_pos=4096, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_pos)
        self.encoder = TransformerEncoderWithAttn(num_layers, d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.sent_pool = nn.AdaptiveAvgPool1d(1)  # to pool token embeddings into sentence embedding
        self.sent_scorer = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))

    def forward_document(self, token_id_lists: List[List[int]]):
        """Process a document represented as a list of sentences, each a list of token ids.
        Returns sentence_scores (tensor len=S), and attention maps.
        """
        # Build a flat token sequence with sentence boundaries and keep sentence spans
        separator_id = 1  # use token id 1 as separator; ensure tokenizer mapping keeps it consistent
        flat_tokens = [separator_id]
        sent_spans = []  # list of (start_idx, end_idx) inclusive
        for sent in token_id_lists:
            start = len(flat_tokens)
            flat_tokens.extend(sent)
            end = len(flat_tokens) - 1
            sent_spans.append((start, end))
            flat_tokens.append(separator_id)
        # Convert to tensor
        device = next(self.parameters()).device
        input_ids = torch.LongTensor(flat_tokens).unsqueeze(0).to(device)  # (1, T)
        embeddings = self.token_emb(input_ids) * math.sqrt(self.d_model)  # (1, T, D)
        embeddings = self.pos_enc(embeddings)
        enc_out, attn_maps = self.encoder(embeddings)  # enc_out: (1, T, D)
        enc_out = enc_out.squeeze(0)  # (T, D)
        # Pool into sentence embeddings
        sent_embs = []
        for (s, e) in sent_spans:
            if e < s:
                sent_embs.append(torch.zeros(self.d_model, device=device))
                continue
            span_emb = enc_out[s:e + 1, :].transpose(0, 1)  # (D, L)
            pooled = F.adaptive_avg_pool1d(span_emb.unsqueeze(0), 1).squeeze(0).squeeze(-1)  # (D,)
            sent_embs.append(pooled)
        sent_embs = torch.stack(sent_embs, dim=0) if len(sent_embs) > 0 else torch.zeros((0, self.d_model), device=device)
        sent_scores = self.sent_scorer(sent_embs).squeeze(-1)  # (S,)
        # attn_maps is list of (1, T, T) tensors; convert to numpy on cpu when needed
        return sent_scores, attn_maps, sent_spans, flat_tokens

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


# -----------------------------
# Interaction graph builder
# -----------------------------

def map_char_to_token_indices(char_name: str, sentences: List[str], token_ids_per_sentence: List[List[int]], tokenizer) -> List[int]:
    """Find token indices in flat token list that correspond to character name mentions.
    This is heuristic: search for exact name tokens in tokenized sentences.
    Returns flat token positions (relative to the flat token list used by model.forward_document).
    """
    # tokenise char name
    if HF_TOKENIZER_AVAILABLE and hasattr(tokenizer, 'tokenize') and not isinstance(tokenizer, SimpleTokenizer):
        char_tokens = tokenizer.tokenize(char_name)
        char_token_ids = tokenizer.convert_tokens_to_ids(char_tokens)
    else:
        # SimpleTokenizer or HF not available
        # split by non-alphanumeric
        char_tokens = re.findall(r"\w+|[^\s\w]", char_name)
        char_token_ids = [tokenizer.vocab.get(t, None) for t in char_tokens]
    positions = []
    flat_index = 1  # starts after initial separator in our flattening scheme
    for sent_tokens in token_ids_per_sentence:
        L = len(sent_tokens)
        # naive scan for first token of char in sentence
        for i in range(L):
            if char_token_ids and char_token_ids[0] is not None and sent_tokens[i] == char_token_ids[0]:
                # check following tokens roughly
                match = True
                for k in range(1, len(char_token_ids)):
                    if i + k >= L or sent_tokens[i + k] != char_token_ids[k]:
                        match = False
                        break
                if match:
                    positions.extend(list(range(flat_index + i, flat_index + i + len(char_token_ids))))
        flat_index += L + 1  # plus separator
    return positions


def build_interaction_graph(attn_maps: List[torch.Tensor], char_positions: Dict[str, List[int]], normalize: bool = True) -> nx.Graph:
    """Build a graph where nodes are characters and edge weights are aggregated attention from tokens of char A to tokens of char B.
    attn_maps: list of tensors (L layers) each (B=1, T, T)
    char_positions: dict name->list of flat-token indices
    """
    # average attention across layers and across heads (our implementation averaged heads inside MultiheadAttention)
    # attn_maps are (1, T, T) per layer
    device = attn_maps[0].device
    L = len(attn_maps)
    attn_sum = None
    for a in attn_maps:
        # a: (1, T, T) -> squeeze batch
        mat = a.squeeze(0).detach().cpu().numpy()  # (T, T)
        if attn_sum is None:
            attn_sum = np.copy(mat)
        else:
            attn_sum += mat
    attn_avg = attn_sum / max(1, L)
    # attn_avg[i, j] = attention from token i (query) to token j (key)
    G = nx.Graph()
    for name in char_positions:
        G.add_node(name)
    names = list(char_positions.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            pos_a = char_positions[a]
            pos_b = char_positions[b]
            if len(pos_a) == 0 or len(pos_b) == 0:
                continue
            # aggregate attention from tokens of a (queries) to tokens of b (keys)
            total = 0.0
            count = 0
            for p in pos_a:
                for q in pos_b:
                    if p < attn_avg.shape[0] and q < attn_avg.shape[1]:
                        total += attn_avg[p, q]
                        count += 1
            if count == 0:
                continue
            weight = total / count
            G.add_edge(a, b, weight=float(weight))
    if normalize and G.number_of_edges() > 0:
        # scale weights to [0,1]
        ws = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
        mi, ma = ws.min(), ws.max()
        for (u, v, d) in G.edges(data=True):
            if ma - mi > 1e-9:
                d['weight'] = (d['weight'] - mi) / (ma - mi)
    return G


def draw_graph(G: nx.Graph, out_path: str = 'graph.png'):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=600)
    nx.draw_networkx_labels(G, pos)
    # scale edge widths
    widths = [1 + 5 * w for w in weights]
    nx.draw_networkx_edges(G, pos, width=widths)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Training & inference utilities
# -----------------------------

def train_one_epoch(model: SummarizerModel, dataloader: DataLoader, optimizer, device, clip_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    for item in tqdm(dataloader, desc='Training docs'):
        # our dataloader yields a list of one document per item because collate_fn returns batch as-is
        doc = item[0]
        token_ids = doc['token_ids']
        labels = doc.get('labels', None)
        if labels is None:
            # skip unsupervised sample for supervised training
            continue
        labels_tensor = torch.zeros(len(token_ids), device=device)
        for idx in labels:
            if 0 <= idx < len(token_ids):
                labels_tensor[idx] = 1.0
        optimizer.zero_grad()
        sent_scores, attn_maps, sent_spans, flat_tokens = model.forward_document(token_ids)
        # binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(sent_scores, labels_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def infer_summary(model: SummarizerModel, tokenizer, script_text: str, top_k: int = 3, device='cpu') -> Tuple[str, List[str], List[float], List[torch.Tensor]]:
    model.eval()
    sentences = sentence_split(script_text)
    token_ids = [tokenizer.encode(s) for s in sentences]
    with torch.no_grad():
        sent_scores, attn_maps, sent_spans, flat_tokens = model.forward_document(token_ids)
        scores = torch.sigmoid(sent_scores).cpu().numpy()
    # pick top_k by score but keep original order
    if len(scores) == 0:
        return '', sentences, [], attn_maps
    indices = np.argsort(-scores)[:top_k]
    indices_sorted = sorted(indices)
    selected = [sentences[i] for i in indices_sorted]
    summary = ' '.join(selected)
    return summary, sentences, scores.tolist(), attn_maps


# -----------------------------
# Command-line interface
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'graph'], required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--script', type=str, default=None)
    parser.add_argument('--pretrained_tokenizer', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='summ_model.pt')
    parser.add_argument('--out', type=str, default='output.txt')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.mode == 'train':
        # Build tokenizer from data
        all_texts = []
        for f in os.listdir(args.data_dir):
            if f.endswith('.txt'):
                all_texts.append(open(os.path.join(args.data_dir, f), 'r', encoding='utf-8').read())
        tokenizer = get_tokenizer(args.pretrained_tokenizer, texts=all_texts if not HF_TOKENIZER_AVAILABLE else None)
        # if HF tokeniser used, compute vocab_size via tokenizer.vocab_size
        if HF_TOKENIZER_AVAILABLE and args.pretrained_tokenizer:
            vocab_size = AutoTokenizer.from_pretrained(args.pretrained_tokenizer).vocab_size
        else:
            vocab_size = tokenizer.next_id
        dataset = ScriptDataset(args.data_dir, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        model = SummarizerModel(vocab_size=vocab_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            loss = train_one_epoch(model, dataloader, optimizer, device)
            print(f'Epoch {ep+1}/{args.epochs} loss={loss:.4f}')
            model.save(args.save_path)
        print('Training finished. Model saved to', args.save_path)

    elif args.mode == 'infer':
        assert args.script is not None
        # load tokenizer and model (attempt to detect from same folder)
        tokenizer = get_tokenizer(args.pretrained_tokenizer)
        # If using SimpleTokenizer you must have previously built it and stored vocab; here we assume a fresh run uses simple tokenizer
        # create model with a placeholder vocab size (expandable) - better to save tokenizer and model together in production
        vocab_size = tokenizer.next_id if not HF_TOKENIZER_AVAILABLE else AutoTokenizer.from_pretrained(args.pretrained_tokenizer).vocab_size
        model = SummarizerModel(vocab_size=vocab_size).to(device)
        if os.path.exists(args.save_path):
            model.load(args.save_path, map_location=device)
        script_text = open(args.script, 'r', encoding='utf-8').read()
        summary, sentences, scores, attn_maps = infer_summary(model, tokenizer, script_text, top_k=args.top_k, device=device)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(summary)
        print('Summary written to', args.out)

    elif args.mode == 'graph':
        assert args.script is not None
        tokenizer = get_tokenizer(args.pretrained_tokenizer)
        vocab_size = tokenizer.next_id if not HF_TOKENIZER_AVAILABLE else AutoTokenizer.from_pretrained(args.pretrained_tokenizer).vocab_size
        model = SummarizerModel(vocab_size=vocab_size).to(device)
        if os.path.exists(args.save_path):
            model.load(args.save_path, map_location=device)
        script_text = open(args.script, 'r', encoding='utf-8').read()
        # detect characters
        chars = extract_characters(script_text)
        if len(chars) == 0:
            print('No characters detected automatically. Try increasing heuristics or provide names manually.')
            chars = []
        # tokenize sentences
        sentences = sentence_split(script_text)
        token_ids = [tokenizer.encode(s) for s in sentences]
        # forward to get attention maps
        with torch.no_grad():
            sent_scores, attn_maps, sent_spans, flat_tokens = model.forward_document(token_ids)
        # map characters to token indices
        char_positions = {}
        for c in chars:
            pos = map_char_to_token_indices(c, sentences, token_ids, tokenizer)
            char_positions[c] = pos
        G = build_interaction_graph(attn_maps, char_positions)
        draw_graph(G, args.out)
        print('Interaction graph saved to', args.out)


# if __name__ == '__main__':
#     main()
