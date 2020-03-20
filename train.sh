dataset=$1
python train.py  --vocab_src ${dataset}/vocab_src \
    --vocab_tgt ${dataset}/vocab_tgt \
    --embed_dim 512 \
    --hidden_size 512 \
    --num_layers 2 \
    --dropout 0.33 \
    --epochs 50 \
    --lr 0.0005 \
    --train_batch_size 512 \
    --print_every 1000 \
    --eval_every 10000 \
    --train_data ${dataset}/train \
    --dev_data  ${dataset}/test \
    --stop_words_file ${dataset}/stop_words \
    --random_mask 0.3 \
    --input_feed
