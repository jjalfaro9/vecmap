#!/bin/sh

# run concat expirements

# Part one generate both embedding file of concat target and removed concat target
echo get embeddings en-it-es and en-it-es_removed
python3 map_embeddings.py \
  --unsupervised data/embeddings/en.emb.txt data/embeddings/it.emb.txt \
  out_source_en_concat_it_es.emb.txt out_trg_en_concat_it_es_removed_es.emb.txt \
  --concatenate data/embeddings/es.emb.txt \
  --remove_lan_from_target data/embeddings/es.emb.txt \
  --save_before_removing_from_targ out_trg_en_concat_it_es.emb.txt \
  --cuda

echo evaluate en-it-es_removed
python3 eval_translation.py \
  out_source_en_concat_it_es.emb.txt out_trg_en_concat_it_es_removed_es.emb.txt \
   -d data/dictionaries/en-it.test.txt \
   --retrieval csls \
   --cuda

echo evaluate en-it-es
python3 eval_translation.py \
 out_source_en_concat_it_es.emb.txt out_trg_en_concat_it_es.emb.txt \
  -d data/dictionaries/en-it.test.txt \
  --retrieval csls \
  --cuda

echo get embeddings en-es-it and en-es-it_removed
python3 map_embeddings.py \
  --unsupervised data/embeddings/en.emb.txt data/embeddings/es.emb.txt \
  out_source_en_concat_es_it.emb.txt out_trg_en_concat_es_it_removed_it.emb.txt \
  --concatenate data/embeddings/it.emb.txt \
  --remove_lan_from_target data/embeddings/it.emb.txt \
  --save_before_removing_from_targ out_trg_en_concat_es_it.emb.txt \
  --cuda

echo evalutate en-es-it_removed
python3 eval_translation.py \
  out_source_en_concat_es_it.emb.txt out_trg_en_concat_es_it_removed_it.emb.txt \
   -d data/dictionaries/en-es.test.txt \
   --retrieval csls \
   --cuda

echo evaluate en-es-it
python3 eval_translation.py \
  out_source_en_concat_es_it.emb.txt out_trg_en_concat_es_it.emb.txt \
   -d data/dictionaries/en-es.test.txt \
   --retrieval csls \
   --cuda
