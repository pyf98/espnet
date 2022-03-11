#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000
data_url=https://dl.fbaipublicfiles.com/librilight/data
datasets="small"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRILIGHT}" ]; then
    log "Fill the value of 'LIBRILIGHT' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	echo "stage 1: Data Downloading to ${LIBRILIGHT}"
	for part in ${datasets}; do
        local/download_and_untar.sh ${LIBRILIGHT} ${data_url} ${part}
	done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Segmentation"
    _nj=64
    _logdir_root=${LIBRILIGHT}/logdir
    for part in ${datasets}; do
        log "Segment ${LIBRILIGHT}/${part} to ${LIBRILIGHT}/${part}_segmented"
        _logdir=${_logdir_root}/${part}_segmented
        mkdir -p ${_logdir}
        
        # Split book paths for multi-processing
        python local/split_book_paths.py \
            --root ${LIBRILIGHT}/${part} \
            --output_dir ${_logdir} \
            --num_outputs ${_nj}

        # Launch jobs to segment the audios
        ${train_cmd} "JOB=1:${_nj}" "${_logdir}/segment_audio.JOB.log" \
            python local/cut_by_vad.py \
                --books_file "${_logdir}/book_path.JOB" \
                --output_dir "${LIBRILIGHT}/${part}_segmented" \
                --target_len_sec 60 \
                --out_extension ".flac"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Kaldi-Style Data Preparation"
    for part in ${datasets}; do
        python local/data_prep.py \
            --dataset_root "${LIBRILIGHT}/${part}_segmented" \
            --output_dir "data/${part}"
        
        utils/utt2spk_to_spk2utt.pl <"data/${part}/utt2spk" >"data/${part}/spk2utt" || exit 1

        utils/validate_data_dir.sh --no-feats "data/${part}" || exit 1
    done
fi

## Combine small, medium and large if the entire set is needed
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#     log "stage 4: combine all datasets"
#     utils/combine_data.sh data/full data/small data/medium data/large
#     utils/validate_data_dir.sh --no-feats data/full || exit 1
# fi

log "Successfully finished. [elapsed=${SECONDS}s]"
