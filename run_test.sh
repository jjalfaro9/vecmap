#!/bin/sh
# Multi-boot experiments

# Part 1: English to Italian
echo Running Part 1 of 4
python3 map_embeddings.py --unsupervised data/embeddings/en.emb.txt data/embeddings/it.emb.txt data/new_emb/en_src_multi_boot1.emb.txt data/new_emb/it_trg_multi_boot1.emb.txt --cuda

# Part 2: English embeddings from Part 1 to Spanish
echo Running Part 2 of 4
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot1.emb.txt data/embeddings/es.emb.txt data/new_emb/en_src_multi_boot2.emb.txt data/new_emb/es_trg_multi_boot2.emb.txt --cuda

# Part 3: English to Spanish
echo Running Part 3 of 4
python3 map_embeddings.py --unsupervised data/embeddings/en.emb.txt data/embeddings/es.emb.txt data/new_emb/en_src_multi_boot3.emb.txt data/new_emb/es_trg_multi_boot3.emb.txt --cuda

# Part 4: English embeddings from Part 3 to Italian
echo Running Part 4 of 4
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot3.emb.txt data/embeddings/it.emb.txt data/new_emb/en_src_multi_boot4.emb.txt data/new_emb/it_trg_multi_boot4.emb.txt --cuda