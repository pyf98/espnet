"""Extract ASR only utterances from dump.
"""

from pathlib import Path


sos = "<startoftranscript>"
eos = "<endoftext>"
task = "<transcribe>"
timestamps = [f"<{i * 0.02:.2f}>" for i in range(1501)]


def read_text_as_dict(file: Path):
    res = {}
    with open(file, 'r') as fp:
        for line in fp.readlines():
            line = line.strip().split()
            uttid, trans = line[0], ' '.join(line[1:])
            res[uttid] = trans
    return res


if __name__ == "__main__":
    root = Path("dump/raw")
    datasets = ['dev', 'train']

    for data in datasets:
        text = read_text_as_dict(root / data / 'text')
        ctc_lines = []
        for uttid, trans in text.items():
            if 'transcribe' in uttid:
                asr_trans = trans
            else:
                asr_trans = text[uttid.replace('translate', 'transcribe')]

            # remove prev text
            asr_trans = asr_trans[asr_trans.find(sos):]

            # remove timestamp tokens
            for tok in timestamps:
                asr_trans = asr_trans.replace(tok, '')
            
            # remove special tokens
            asr_trans = asr_trans.replace(sos, '').replace(eos, '').replace(task, '').strip()

            ctc_lines.append(f"{uttid} {asr_trans}")

        with open(root / data / 'text.ctc', 'w') as fp:
            fp.write('\n'.join(ctc_lines))
