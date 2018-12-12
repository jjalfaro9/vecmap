import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from cupy_utils import *

def eval_cosine_sim(src_data, trg_data, use_file_emb, dict_path, sameLanguage, output_file):
    if use_file_emb:
        src_in = open(src_data, encoding="utf-8", errors='surrogateescape')
        trg_in = open(trg_data, encoding="utf-8", errors='surrogateescape')
        num_emb = int(src_in.readline().split(" ")[0])
        num_dim = int(trg_in.readline().split(" ")[1])

    # NumPy/CuPy management
    if not supports_cupy():
        print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
        sys.exit(-1)
    xp = get_cupy()
    x = xp.asnumpy(src_data[1])
    z = xp.asnumpy(trg_data[1])

    if not sameLanguage:
        match_count = 0
        sim_vec = None
        src_emb = []
        trg_emb = []

        dict_in = open(dict_path, "r")
        dict_trans = []

        for line in dict_in:
            dict_trans.append((line.split()[0], line.split()[1]))

        dict_in.close()

        if use_file_emb:
            for i in range(0, num_emb):
                src_emb.append(src_in.readline())
                trg_emb.append(trg_in.readline())

        indices_map = np.full((len(dict_trans), 2), -1)

        if use_file_emb:
            src_toks = [item1.split()[0] for item1 in src_emb]
            trg_toks = [item1.split()[0] for item1 in trg_emb]
        else:
            src_toks = src_data[0]
            trg_toks = trg_data[0]

        for iter, item in enumerate(dict_trans):
            match = False
            if item[0] in src_toks:
                indices_map[iter][0] = src_toks.index(item[0])
                match = True
            if item[1] in trg_toks:
                indices_map[iter][1] = trg_toks.index(item[1])
            else:
                match = False

            if match:
                match_count += 1

        sim_vec = np.zeros(match_count)

        j = 0

        output_list = []

        for iter, item in enumerate(indices_map):
            if item[0] != -1 and item[1] != -1:
                if use_file_emb:
                    src_str = src_emb[int(item[0])].split(" ", 1)[1]
                    trg_str = trg_emb[int(item[1])].split(" ", 1)[1]
                    src_vec = np.fromstring(src_str, dtype=float, sep=' ').reshape(1, -1)
                    trg_vec = np.fromstring(trg_str, dtype=float, sep=' ').reshape(1, -1)
                else:
                    src_vec = x[int(item[0])].reshape(1, -1)
                    trg_vec = z[int(item[1])].reshape(1, -1)
                #print(src_vec, trg_vec)
                sim_vec[j] = cosine_similarity(src_vec, trg_vec)[0][0]
                output_list.append((src_toks[int(item[0])], trg_toks[int(item[1])], str(sim_vec[j])))
                j += 1

        output_list = sorted(output_list, key=lambda x: x[2], reverse=True)

        if output_file:
            out_file = open(output_file, "w")
            for item in output_list:
                out_file.write(item[0] + " " + item[1] + " " + item[2] + "\n")
            out_file.close()
    else:
        sim_vec = np.zeros(num_emb)

        for i in range(0, num_emb):
            src_tok, src_str = src_in.readline().split(" ", 1)
            trg_tok, trg_str = trg_in.readline().split(" ", 1)
            src_vec = np.fromstring(src_str, dtype=float, sep=' ').reshape(1, -1)
            trg_vec = np.fromstring(trg_str, dtype=float, sep=' ').reshape(1, -1)

            sim_vec[i] = cosine_similarity(src_vec, trg_vec)[0][0]

    if not use_file_emb:
        rpt_file = open("report.txt", "a+")
        rpt_file.write("Mean: " + str(np.mean(sim_vec)) + ", Median: " + str(np.median(sim_vec)) + ", STD: " + str(np.std(sim_vec)) + "\n\n")
        rpt_file.close()

    print("Mean: " + str(np.mean(sim_vec)))
    print("Median: " + str(np.median(sim_vec)))
    print("Min: " + str(np.min(sim_vec)))
    print("Max: " + str(np.max(sim_vec)))

    if use_file_emb:
        src_in.close()
        trg_in.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate cosine similarity metrics')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('--test_dict', type=str, default=None, help='test dictionary')
    parser.add_argument('--same_language', action='store_true', help='the target language embeddings')
    parser.add_argument('--sorted_file', type=str, default=None, help='file for sorted output')

    args = parser.parse_args()
    eval_cosine_sim(args.src_embeddings, args.trg_embeddings, True, args.test_dict, args.same_language, args.sorted_file)

