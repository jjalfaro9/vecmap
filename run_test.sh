#!/bin/sh
# Multi-boot experiments

echo Running Part 2 (of 4)
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot1.emb.txt data/embeddings/es.emb.txt data/new_emb/en_src_multi_boot2_ex2.emb.txt data/new_emb/es_trg_multi_boot2_ex2.emb.txt --cuda

echo Running Part 4 (of 4)
python3 map_embeddings.py --unsupervised data/new_emb/en_src_multi_boot3.emb.txt data/embeddings/it.emb.txt data/new_emb/en_src_multi_boot4_ex2.emb.txt data/new_emb/it_trg_multi_boot4_ex2.emb.txt --cuda

echo Eval 1
python3 eval_translation.py data/new_emb/en_src_multi_boot2_ex2.emb.txt data/new_emb/es_trg_multi_boot2_ex2.emb.txt -d data/dictionaries/en-es.test.txt --retrieval csls --cuda
echo Eval 2
python3 eval_translation.py data/new_emb/en_src_multi_boot4_ex2.emb.txt data/new_emb/it_trg_multi_boot4_ex2.emb.txt -d data/dictionaries/en-it.test.txt --retrieval csls --cuda
echo Eval 3
python3 eval_translation.py data/new_emb/interm_out_en_src_multi_boot1.emb.txt data/new_emb/interm_out_es.emb.txt -d data/dictionaries/en-es.test.txt --retrieval csls --cuda
echo Eval 4
python3 eval_translation.py data/new_emb/interm_out_en_src_multi_boot3.emb.txt data/new_emb/interm_out_it.emb.txt -d data/dictionaries/en-it.test.txt --retrieval csls --cuda

scp -i "/home/alex/Atomkovich.pem" stored_emb/es_src_multi_boot1.emb.txt ubuntu@ec2-3-17-80-51.us-east-2.compute.amazonaws.com:/home/ubuntu/BLI/vecmap/data/new_emb/
scp -i "/home/alex/Atomkovich.pem" stored_emb/es_src_multi_boot3.emb.txt ubuntu@ec2-3-17-80-51.us-east-2.compute.amazonaws.com:/home/ubuntu/BLI/vecmap/data/new_emb/
