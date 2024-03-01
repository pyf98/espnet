#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets=dev

nbpe=50000
s2t_config=conf/train_s2t_ebf_conv2d_size768_e9_d9_piecewise_lr5e-4_warmup60k_flashattn.yaml
inference_config=conf/decode_s2t.yaml

./s2t.sh \
    --stage 11 \
    --stop_stage 11 \
    --use_lm false \
    --num_nodes 4 \
    --ngpu 4 \
    --nj 128 \
    --gpu_inference true \
    --inference_nj 64 \
    --num_splits_s2t 10 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
