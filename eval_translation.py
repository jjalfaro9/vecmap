# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys
import codecs

BATCH_SIZE = 500


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def eval_translation(src_emb, trg_emb, use_file_emb, test_dict, rpt_file, run_msg, trans_out_file, other_settings):

    p_retrieval = other_settings[0]
    p_inv_temperature = other_settings[1]
    p_inv_sample = other_settings[2]
    p_neighborhood = other_settings[3]
    p_dot = other_settings[4]
    p_encoding = other_settings[5]
    p_seed = other_settings[6]
    p_precision = other_settings[7]
    p_cuda = other_settings[8]

    # Choose the right dtype for the desired precision
    if p_precision == 'fp16':
        dtype = 'float16'
    elif p_precision == 'fp32':
        dtype = 'float32'
    elif p_precision == 'fp64':
        dtype = 'float64'

    # NumPy/CuPy management
    if p_cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
    else:
        xp = np
    xp.random.seed(p_seed)

    if use_file_emb:
        # Read input embeddings
        srcfile = open(src_emb, encoding=p_encoding, errors='surrogateescape')
        trgfile = open(trg_emb, encoding=p_encoding, errors='surrogateescape')
        src_words, x = embeddings.read(srcfile, dtype=dtype)
        trg_words, z = embeddings.read(trgfile, dtype=dtype)
    else:
        src_words, x = src_emb[0], xp.copy(src_emb[1])
        trg_words, z = trg_emb[0], xp.copy(trg_emb[1])

    if p_cuda:
        x = xp.asarray(x)
        z = xp.asarray(z)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not p_dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Read dictionary and compute coverage
    f = open(test_dict, encoding=p_encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

    # Find translations
    translation = collections.defaultdict(int)
    if p_retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = x[src[i:j]].dot(z.T)
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif p_retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif p_retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if p_inv_sample is None else xp.random.randint(0, x.shape[0], p_inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(p_inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(p_inv_temperature*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif p_retrieval == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=p_neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]

    print(translation)
    # Compute accuracy
    accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])

    correct = [i for i in src if translation[i] in src2trg[i]]
    incorrect = [i for i in src if translation[i] not in src2trg[i]]

    if trans_out_file:
        trans_out = codecs.open(trans_out_file, "w", "utf-8")
        trans_out.write("##########Correct:\n")

        for i in correct:
            print(src_words[i])
            trans_out.write(src_words[i] + ": System{" + trg_words[translation[i]] + "} | Gold{" + str([trg_words[item] for item in src2trg[i]]) + "}\n")
        trans_out.write("##########Incorrect:\n")
        for i in incorrect:
            trans_out.write(src_words[i] + ": System{" + trg_words[translation[i]] + "} | Gold{" + str([trg_words[item] for item in src2trg[i]]) + "}\n")
        trans_out.close()

    if rpt_file:
        out_file = open(rpt_file, "a+")
        out_file.write(run_msg + "\n")

        out_file.write('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy) + "\n")
        out_file.close()
    else:
        print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--rpt_file', default=None, type=str, help='output file for report metrics')
    parser.add_argument('--run_msg', default=None, type=str, help='output message for report metrics')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    other_settings = [args.retrieval, args.inv_temperature, args.inv_sample, args.neighborhood, args.dot, args.encoding, args.seed, args.precision, args.cuda]
    eval_translation(args.src_embeddings, args.trg_embeddings, True, args.dictionary, args.rpt_file, args.run_msg, "trans_out.txt", other_settings)
