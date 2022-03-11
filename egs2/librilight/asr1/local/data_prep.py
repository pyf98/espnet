#!/usr/bin/env python

# Kaldi-style data preparation
# Author: Yifan Peng (Carnegie Mellon University)


import argparse
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare data in kaldi style"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the kaldi style files"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # audio name format: root / dataset / speaker_id / book_id / basename
    audios = sorted(pathlib.Path(args.dataset_root).glob("*/*/*.flac"))
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with (output_dir / "text").open('w') as text_f, \
        (output_dir / "wav.scp").open('w') as wav_scp_f, \
        (output_dir / "utt2spk").open('w') as utt2spk_f:
        for audio in audios:
            audio = audio.resolve()
            name = audio.name[:-5]  # remove ".flac"
            book_id = audio.parent.name
            speaker_id = audio.parent.parent.name

            utt_id = f"{speaker_id}-{book_id}-{name}"
            utt2spk_f.write(f"{utt_id} {speaker_id}\n")
            text_f.write(f"{utt_id} <blank>\n")
            wav_scp_f.write(f"{utt_id} flac -c -d -s {str(audio)} |\n")
