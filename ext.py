import numpy as np

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
