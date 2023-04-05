from kaldiio import load_scp
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
from typing import Dict

from espnet2.bin.asr_inference import Speech2Text


def parse_text(file: Path) -> Dict[str, str]:
    res = {}
    with file.open('r') as fp:
        for line in fp.readlines():
            line = line.strip().split()
            uttid, trans = line[0], " ".join(line[1:])
            res[uttid] = trans
    return res

def format_text(text: str) -> str:
    specials = [
        "<endoftext>",
        "<startoftranscript>",
        *[f"<translate{l}>" for l in ['de', 'ja', 'zh']],
        "<transcribe>",
        "<startofprev>",
        "<notimestamps>",
    ]
    for tok in specials:
        text = text.replace(tok, '')
    return text.strip()


if __name__ == "__main__":
    s2t = Speech2Text(
        asr_model_file='exp/asr_train_whisper_base_st-asr_ctc0.3_prev0.5_time0.5_small-ebf_lr1e-3_warmup5k_raw_bpe10000/87epoch.pth',
        token_type='bpe',
        bpemodel='data/token_list/bpe_unigram10000/bpe.model',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        beam_size=10,
        ctc_weight=0.0,
        penalty=0.0,
        maxlenratio=0.0,
        hyp_primer=["<startoftranscript>", "<transcribe>", "<0.00>"],
    )
    
    # Option 1: Load audio from Kaldi-style files
    data_dir = Path('dump/raw/dev')
    d = load_scp(str(data_dir / 'wav.scp'))
    uttid2trans = parse_text(data_dir / 'text')

    uttid = 'ted_00824_0110950_0139770_en-zh_transcribe'
    
    rate, audio = d[uttid]
    sf.write(f"{uttid}.wav", audio, rate)

    if len(audio) < 30 * rate:
        audio = np.pad(audio, (0, 30 * rate - len(audio)))
    else:
        audio = audio[:30 * rate]
    assert len(audio) == 30 * rate
    result = s2t(audio)
    hyp = result[0][0]

    prev, trans = uttid2trans[uttid].split("<startoftranscript>")
    print("--------- Prev ---------")
    print(format_text(prev))
    print("--------- Ref ---------")
    print(format_text(trans))
    print("********* Hyp *********")
    print(format_text(hyp))


    #################################
    # Option 2: Load audio from file
    #################################
    # audio, rate = sf.read("/scratch/bbjs/peng6/corpora/MuST-C/en-de/data/tst-COMMON/wav/ted_1166.wav")
    # start_sec = 963.43
    # end_sec = 993.43
    # audio = audio[int(start_sec * rate):int(end_sec * rate)]
    # sf.write(f"tmp-{start_sec}-{end_sec}.wav", audio, rate)

    # if len(audio) < 30 * rate:
    #     audio = np.pad(audio, (0, 30 * rate - len(audio)))
    # else:
    #     audio = audio[:30 * rate]
    # assert len(audio) == 30 * rate
    # result = s2t(audio)
    # hyp = result[0][0]

    # print("********* Hyp *********")
    # print(format_text(hyp))
