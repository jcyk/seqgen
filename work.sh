export https_proxy="http://10.223.133.20:52107"
export http_proxy="http://10.223.133.20:52107"
python work.py --test_batch_size 512 \
                --test_data masker/test_out\
                --load_path ckpt/epoch17 \
                --beam_size 5 \
                --max_time_step 20 \
                --output_file ./test_out17 \
                --verbose
