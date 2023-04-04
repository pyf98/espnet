import soundfile
from kaldiio import load_scp
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
        asr_model_file='exp/asr_train_whisper_base_st-asr_ctc0.3_prev0.5_time0.5_raw_bpe10000/66epoch.pth',
        token_type='bpe',
        bpemodel='data/token_list/bpe_unigram10000/bpe.model',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        beam_size=10,
        ctc_weight=0.0,
        penalty=0.0,
        maxlenratio=0.0,
        hyp_primer=["<startoftranscript>", "<translateja>", "<0.00>"],
    )

    scp_file = Path('dump/raw/dev/wav.scp')
    text_file = scp_file.parent / 'text'
    d = load_scp(str(scp_file))
    uttid2trans = parse_text(text_file)

    uttid = 'ted_00767_1106380_1133560_en-ja_translate'
    
    rate, audio = d[uttid]
    soundfile.write(f"{uttid}.wav", audio, rate)
    audio = np.pad(audio, (0, 30 * rate - len(audio)))

    result = s2t(audio)
    hyp = result[0][0]

    prev, trans = uttid2trans[uttid].split("<startoftranscript>")
    print("--------- Prev ---------")
    print(format_text(prev))
    print("--------- Ref ---------")
    print(format_text(trans))
    print("********* Hyp *********")
    print(format_text(hyp))

    breakpoint()
