test_sets="tst-COMMON.en-de tst-HE.en-de"
decode_dir="exp/asr_train_whisper_base_st-asr_ctc0.3_prev0.5_time0.5_raw_bpe10000/decode_whisper_asr_model_valid.acc.ave"

for dset in ${test_sets}; do
    cur_dir=${decode_dir}/${dset}/score_wer
    old_hyp=${cur_dir}/hyp.trn
    new_hyp=${cur_dir}/hyp_new.trn

    sed -e 's/<transcribe>//g' -e 's/<notimestamps>//g' ${old_hyp} > ${old_hyp}.nospecial
    cut -f1 ${old_hyp}.nospecial > ${old_hyp}.nospecial.trans
    cut -f2 ${old_hyp}.nospecial > ${old_hyp}.nospecial.uttid

    normalize-punctuation.perl -l en < ${old_hyp}.nospecial.trans > ${old_hyp}.nospecial.trans.norm
    lowercase.perl < ${old_hyp}.nospecial.trans.norm > ${old_hyp}.nospecial.trans.norm.lc
    remove_punctuation.pl < ${old_hyp}.nospecial.trans.norm.lc > ${old_hyp}.nospecial.trans.norm.lc.rm
    tokenizer.perl -l en -q < ${old_hyp}.nospecial.trans.norm.lc.rm > ${old_hyp}.nospecial.trans.norm.lc.rm.tok

    paste ${old_hyp}.nospecial.trans.norm.lc.rm.tok ${old_hyp}.nospecial.uttid > ${new_hyp}

    sclite \
        -r "${cur_dir}/ref.trn" trn \
        -h "${new_hyp}" trn \
        -i rm -o all stdout > "${cur_dir}/result_new.txt"
done
