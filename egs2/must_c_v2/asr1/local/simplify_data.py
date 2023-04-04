"""Extract ASR only utterances from dump.
"""

import shutil
from pathlib import Path


sos = "<startoftranscript>"
eos = "<endoftext>"
task = "<transcribe>"
timestamps = [f"<{i * 0.02:.2f}>" for i in range(1501)]


if __name__ == "__main__":
    root = Path("dump/raw")
    datasets = ['dev', 'train']

    for data in datasets:
        old_dir = root / data
        new_dir = old_dir.parent / (old_dir.name + ".asr")

        shutil.copytree(old_dir, new_dir)

        asr_lines = []
        asr_ctc_lines = []
        with open(new_dir / 'text', 'r') as fp:
            for line in fp.readlines():
                line = line.strip().split()
                uttid, trans = line[0], ' '.join(line[1:])
                
                if 'transcribe' in uttid:   # only keep asr data
                    asr_lines.append(' '.join(line))

                    # remove prev text
                    trans = trans[trans.find(sos):]

                    # remove timestamp tokens
                    for tok in timestamps:
                        trans = trans.replace(tok, '')
                    
                    # remove special tokens
                    trans = trans.replace(sos, '').replace(eos, '').replace(task, '').strip()

                    asr_ctc_lines.append(f"{uttid} {trans}")

        with open(new_dir / 'text', 'w') as fp:
            fp.write('\n'.join(asr_lines))
        with open(new_dir / 'text.ctc', 'w') as fp:
            fp.write('\n'.join(asr_ctc_lines))
