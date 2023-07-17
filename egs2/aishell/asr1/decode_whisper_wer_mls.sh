#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


nj=8

# small medium large-v2
# en it pt pl es fr de nl
for whisper_tag in small; do
    for lang in pt pl es fr de nl; do
        if [ "${lang}" = "en" ]; then
            cleaner=whisper_en
        else
            cleaner=whisper_basic
        fi
        echo "language: ${lang}, cleaner: ${cleaner}"
        hyp_cleaner=${cleaner}
        decode_options="{language: ${lang}, task: transcribe, temperature: 0, beam_size: 1, fp16: False, sample_len: 300}"

        x="test/MLS/${lang}_test"
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
