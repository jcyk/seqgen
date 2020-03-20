dataset=$1
python train.py --vocab_src ${dataset}/vocab_src \
                --vocab_tgt ${dataset}/vocab_tgt \
                --embed_dim 512 \
                --ff_embed_dim 1024 \
                --num_heads 8 \
                --num_layers 2 \
                --dropout 0.1 \
                --epochs 20 \
                --lr 0.0001 \
                --train_batch_size 128 \
                --dev_batch_size 128 \
                --print_every 100 \
                --eval_every 1000 \
                --train_data ${dataset}/train \
                --dev_data ${dataset}/dev \
                --which_ranker ranker
