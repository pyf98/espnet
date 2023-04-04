#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train.asr
valid_set=dev.asr
test_sets="dev.asr"

asr_config=conf/tuning/train_whisper_base_asr_ctc0.3_prev0.5_time0.5.yaml
inference_config=conf/tuning/decode_whisper.yaml

nbpe=10000

./asr.sh \
    --asr_stats_dir exp/asr_stats_raw_bpe10000_asr \
    --stage 11 \
    --stop_stage 11 \
    --use_lm false \
    --ngpu 4 \
    --nj 128 \
    --inference_asr_model valid.acc.ave.pth \
    --gpu_inference true \
    --inference_nj 4 \
    --feats_type raw \
    --audio_format "flac.ark" \
    --max_wav_duration 31 \
    --token_type "bpe" \
    --nbpe $nbpe \
    --bpe_nlsyms data/nlsyms.txt \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text"  "$@"
