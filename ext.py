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

    return (src_counts, trg_counts)
