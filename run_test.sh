#!/bin/sh
# Multi-boot experiments

# Part 1: English to Italian
#echo Running Part 1 of 4
python3 map_embeddings.py --unsupervised data/embeddings/en.emb.txt data/embeddings/it.emb.txt data/new_emb/en_src_multi_boot1.emb.txt data/new_emb/it_trg_multi_boot1.emb.txt --cuda
# Results (using CSLS): Coverage: 100.00% | Accuracy: 47.60%
# Cosine sim results:
# Mean: 0.538984422115
# Median: 0.572608827774
# Min: -0.102434782034
# Max: 0.879900190598

# Part 2: English embeddings from Part 1 to Spanish
#echo Running Part 2 of 4
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot1.emb.txt data/embeddings/es.emb.txt data/new_emb/en_src_multi_boot2.emb.txt data/new_emb/es_trg_multi_boot2.emb.txt --cuda
# Results (using CSLS): Coverage: 100.00% | Accuracy: 36.47%
# Cosine sim results:
# Mean: 0.514338229612
# Median: 0.543020247244
# Min: -0.189732202788
# Max: 0.900755953448

# Part 3: English to Spanish
#echo Running Part 3 of 4
python3 map_embeddings.py --unsupervised data/embeddings/en.emb.txt data/embeddings/es.emb.txt data/new_emb/en_src_multi_boot3.emb.txt data/new_emb/es_trg_multi_boot3.emb.txt --cuda
# Results (using CSLS): Coverage: 100.00% | Accuracy: 37.13%
# Mean: 0.481332196833
# Median: 0.507991801137
# Min: -0.182423543809
# Max: 0.860988353251

# Part 4: English embeddings from Part 3 to Italian
#echo Running Part 4 of 4
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot3.emb.txt data/embeddings/it.emb.txt data/new_emb/en_src_multi_boot4.emb.txt data/new_emb/it_trg_multi_boot4.emb.txt --cuda
# Results (using CSLS): Coverage: 100.00% | Accuracy: 47.13%
# Mean: 0.56930021374
# Median: 0.604376644694
# Min: -0.129653961145
# Max: 0.887438994383

# Commands

# scp to AWS:
# scp -i "/home/alex/Atomkovich.pem" stored_emb/es_trg_multi_boot2.emb.txt ubuntu@ec2-3-17-80-51.us-east-2.compute.amazonaws.com:/home/ubuntu/BLI/vecmap/data/new_emb/

# Evaluate acc:
# python3 eval_translation.py data/new_emb/en_src_multi_boot2.emb.txt data/new_emb/es_trg_multi_boot2.emb.txt -d data/dictionaries/en-es.test.txt --retrieval csls --cuda
