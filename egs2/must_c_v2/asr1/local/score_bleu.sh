tgt_lang=ja
test_sets="tst-COMMON.en-${tgt_lang} tst-HE.en-${tgt_lang}"
decode_dir="exp/asr_train_whisper_base_st-asr_ctc0.3_prev0.5_time0.5_raw_bpe10000/decode_whisper_asr_model_valid.acc.ave_st-ja"

for dset in ${test_sets}; do
    cur_dir=${decode_dir}/${dset}

    sed -e "s/<translate${tgt_lang}>//g" -e 's/<notimestamps>//g' ${cur_dir}/text > ${cur_dir}/text.nospecial
    cut -d ' ' -f 2- ${cur_dir}/text.nospecial > ${cur_dir}/text.trans
    cut -d ' ' -f 1 ${cur_dir}/text.nospecial > ${cur_dir}/text.uttid

    normalize-punctuation.perl -l ${tgt_lang} < ${cur_dir}/text.trans > ${cur_dir}/text.trans.norm
    tokenizer.perl -l ${tgt_lang} -q < ${cur_dir}/text.trans.norm > ${cur_dir}/text.trans.norm.tok

    paste -d ' ' ${cur_dir}/text.uttid ${cur_dir}/text.trans.norm.tok > ${cur_dir}/text.tc.${tgt_lang}

    ## modified from st.sh
    _scoredir="${cur_dir}/score_bleu"
    mkdir -p "${_scoredir}"

    paste \
        <(<"dump/raw/${dset}/text.tc.${tgt_lang}" \
            python -m espnet2.bin.tokenize_text  \
                -f 2- --input - --output - \
                --token_type word \
                --remove_non_linguistic_symbols true \
                ) \
        <(<"dump/raw/${dset}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${_scoredir}/ref.trn.org"

    # NOTE(kamo): Don't use cleaner for hyp
    paste \
        <(<"${cur_dir}/text.tc.${tgt_lang}"  \
                python -m espnet2.bin.tokenize_text  \
                    -f 2- --input - --output - \
                    --token_type word \
                    --remove_non_linguistic_symbols true \
                    ) \
        <(<"dump/raw/${dset}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${_scoredir}/hyp.trn.org"

    # remove utterance id
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org" > "${_scoredir}/ref.trn"
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/hyp.trn.org" > "${_scoredir}/hyp.trn"

    # detokenizer
    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn" > "${_scoredir}/ref.trn.detok"
    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/hyp.trn" > "${_scoredir}/hyp.trn.detok"

    # rotate result files
    pyscripts/utils/rotate_logfile.py ${_scoredir}/result.tc.txt

    echo "Case sensitive BLEU result (single-reference)" > ${_scoredir}/result.tc.txt
    sacrebleu "${_scoredir}/ref.trn.detok" \
              -i "${_scoredir}/hyp.trn.detok" \
              -m bleu chrf ter \
              >> ${_scoredir}/result.tc.txt
done
