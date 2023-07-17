#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cleaner=whisper_basic
hyp_cleaner=${cleaner}
nj=2
test_sets="test/ReazonSpeech/test"
decode_options="{language: ja, task: transcribe, temperature: 0, beam_size: 1, fp16: False}"

for whisper_tag in small medium large-v2; do
    for x in ${test_sets}; do
        wavscp=dump/raw/${x}/wav.scp
        outdir=whisper-${whisper_tag}_outputs/${x}
        gt_text=dump/raw/${x}/text

        scripts/utils/evaluate_asr.sh \
            --whisper_tag ${whisper_tag} \
            --nj ${nj} \
            --gpu_inference true \
            --stage 2 \
            --stop_stage 3 \
            --nlsyms_txt data/nlsyms_scoring.txt \
            --cleaner ${cleaner} \
            --hyp_cleaner ${hyp_cleaner} \
            --decode_options "${decode_options}" \
            --gt_text ${gt_text} \
            ${wavscp} \
            ${outdir}
    done
done
