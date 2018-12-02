#!/bin/sh
# Multi-boot experiments

# Part 1: English embeddings from Part 1 to Spanish
echo Running Part 1 of 4
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot1.emb.txt data/embeddings/es.emb.txt data/new_emb/en_src_multi_boot2_stoch7.emb.txt data/new_emb/es_trg_multi_boot2_stoch7.emb.txt --stochastic_initial .7 --cuda
# Results: Coverage: 100.00% | Accuracy: 36.40% .5
# Results: Coverage: 100.00% | Accuracy: 36.53% .3

# Part 2: English embeddings from Part 2 to Italian
echo Running Part 2 of 4
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot3.emb.txt data/embeddings/it.emb.txt data/new_emb/en_src_multi_boot4_stoch3.emb.txt data/new_emb/it_trg_multi_boot4_stoch3.emb.txt --stochastic_initial .3 --cuda
# Results: Coverage: 100.00% | Accuracy: 46.80% .5
# Results: Coverage: 100.00% | Accuracy: 47.20% .3
