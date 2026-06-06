import json
import networkx as nx

"""
Prepare vocabulary and initial word vectors.
"""
import json
import tqdm
import pickle
import argparse
import numpy as np
from collections import Counter

class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)    # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}


    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', help='TACRED directory.')
    parser.add_argument('--vocab_dir', help='Output vocab directory.')
    parser.add_argument('--lower', default=True, help='If specified, lowercase all words.')
    args = parser.parse_args()
    return args

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        pos = []
        dep = []
        max_len = 0
        for d in data:
            tokens.extend(d['token'])
            pos.extend(d['pos'])
            dep.extend(d['deprel'])
            max_len = max(len(d['token']), max_len)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, pos, dep, max_len


from prepare_vocab import VocabHelp
def syn_dep_adj_generation(head, dep, vocab_dep):
    syn_dep_edge = []
    for node_s_id, (node_e_id, d) in enumerate(zip(head, dep)):
        if node_e_id == 0:
            continue
        syn_dep_edge.append(
            [node_s_id, node_e_id-1, vocab_dep.stoi.get(d, vocab_dep.unk_index)])
    return syn_dep_edge


def short_adj_generation(head, max_tree_dis=5):
    r'''
    generate short adj matrix
    '''
    head = list(head)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(head)))
    graph.add_edges_from([(node_1, node_2 - 1)
                          for node_1, node_2 in enumerate(head) if node_2 != 0])
    short_adj = [[max_tree_dis]*len(head) for _ in range(len(head))]
    for node_s_id in graph.nodes:
        for node_e_id in graph.nodes:
            try:
                tree_distance = nx.dijkstra_path_length(
                    graph, source=node_s_id, target=node_e_id)
                tree_distance = tree_distance if tree_distance <= max_tree_dis else max_tree_dis
            except:
                tree_distance = max_tree_dis
            short_adj[node_s_id][node_e_id] = tree_distance
    return short_adj

with open("../dataset/Laptops_corenlp/train_write.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Laptops_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Laptops_corenlp/train_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Laptops_corenlp/test_write.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Laptops_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Laptops_corenlp/test_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Restaurants_corenlp/train_write.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Restaurants_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Restaurants_corenlp/train_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Restaurants_corenlp/test_write.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Restaurants_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Restaurants_corenlp/test_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Tweets_corenlp/train_write.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Tweets_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Tweets_corenlp/train_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Tweets_corenlp/test_write.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Tweets_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Tweets_corenlp/test_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()
