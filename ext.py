import numpy as np

def get_char_counts(src, trg):
    src_in = open(src, encoding="utf-8", errors='surrogateescape')
    trg_in = open(trg, encoding="utf-8", errors='surrogateescape')


    alphabet = []
    num_src = int(src_in.readline().split(" ")[0])
    num_trg = int(trg_in.readline().split(" ")[0])

    for _ in range(0, num_src):
        token = src_in.readline().split(" ")[0]

        for item in token:
            if item not in alphabet:
                alphabet.append(item)

    for _ in range(0, num_trg):
        token = trg_in.readline().split(" ")[0]

        for item in token:
            if item not in alphabet:
                alphabet.append(item)

    src_in.seek(0, 0)
    trg_in.seek(0, 0)

    src_in.readline()
    trg_in.readline()

    src_counts = np.zeros((num_src, len(alphabet)), dtype='float32')
    trg_counts = np.zeros((num_trg, len(alphabet)), dtype='float32')

    for i in range(0, num_src):
        token = src_in.readline().split(" ")[0]

        for item in token:
            src_counts[i][alphabet.index(item)] += 1

    for i in range(0, num_trg):
        token = trg_in.readline().split(" ")[0]

        for item in token:
            trg_counts[i][alphabet.index(item)] += 1

    src_in.close()
    trg_in.close()

    #src_counts = src_counts + 1.0
    #trg_counts = trg_counts + 1.0

    return (src_counts, trg_counts)

def orthographic_sim(file, dtype='float'):
    # columns : src, trg, NL, Sim_score, src_idx, trg_idx
    header = file.readline().split('\t')
    matrix = np.empty((20000, 20000), dtype=dtype)
    for line in file:
        info = line.split('\t')
        sim_score = float(info[3])
        src_idx   = int(info[4])
        trg_idx   = int(info[5][:-1]) # hacky, but get rid of new line char

        matrix[src_idx, trg_idx] = sim_score
    return matrix

if __name__ == '__main__':
    print('test')
    f = open('data/en-es-ortho-new.tsv', encoding='utf-8', errors='surrogateescape')
    m = orthographic_sim(f)
    f.close()
    print(m.shape)
