#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

cleaner=whisper_en
hyp_cleaner=${cleaner}
nj=4
test_sets="test/TEDLIUM2_longform/test"
decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 5, fp16: False, condition_on_previous_text: False}"

for whisper_tag in medium; do
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
            --cleaner ${cleaner} \
            --hyp_cleaner ${hyp_cleaner} \
            --decode_options "${decode_options}" \
            --gt_text ${gt_text} \
            ${wavscp} \
            ${outdir}
    done
done
