import yaml
from pathlib import Path


######################################################################
def read_str(file):
    with open(file, 'r') as fp:
        lines = [l.strip() for l in fp.readlines()]
    return lines


def parse_data(root, lang, dataset):
    txt_path = root / f"en-{lang}" / "data" / f"{dataset}" / "txt"
    wav_path = root / f"en-{lang}" / "data" / f"{dataset}" / "wav"

    with open(txt_path / f"{dataset}.yaml", 'r') as fp:
        utts = yaml.safe_load(fp)
    
    src_text = read_str(txt_path / f"{dataset}.en")
    tgt_text = read_str(txt_path / f"{dataset}.{lang}")

    assert len(src_text) == len(tgt_text) and len(tgt_text) == len(utts)

    wav2utts = {}
    for idx, utt in enumerate(utts):
        wav_name = utt['wav']
        if wav_name not in wav2utts:
            wav2utts[wav_name] = []
        
        utt['wav'] = str((wav_path / wav_name).resolve())
        utt['src'] = ' '.join(src_text[idx].split())
        utt['tgt'] = ' '.join(tgt_text[idx].split())
        wav2utts[wav_name].append(utt)
    
    return wav2utts


######################################################################
def find_max_context(utts: list[dict], max_sec, pos):
    # ..., pos-2, pos-1, not including pos
    
    results = []
    end_time = utts[pos]['offset']
    while pos - 1 >= 0 and (end_time - utts[pos - 1]['offset']) <= max_sec:
        results.append(pos - 1)
        pos = pos - 1

    return results[::-1]


def find_max_samples(utts: list[dict], max_sec, pos):
    # pos, pos+1, pos+2, ..., including pos

    results = []
    start_time = utts[pos]['offset']
    while pos < len(utts) and (utts[pos]['duration'] + utts[pos]['offset'] - start_time) <= max_sec:
        results.append(pos)
        pos = pos + 1

    return results


def time2token(x, resolution):
    x = int(x / resolution) * resolution
    return f"<{x:.2f}>"


def generate_samples(utts: list[dict], max_sec=30, resolution=0.02):
    # utts: a singe ted talk consisting of many utterances
    new_utts: list[dict] = []
    for idx in range(len(utts)):
        context = find_max_context(utts, max_sec, idx)
        samples = find_max_samples(utts, max_sec, idx)

        if samples:
            offset = utts[samples[0]]['offset']
            src = []
            tgt = []
            for s in samples:
                src.extend(
                    [
                        time2token(utts[s]['offset'] - offset, resolution),
                        ' ' + utts[s]['src'],   # add a space before text
                        time2token(utts[s]['offset'] + utts[s]['duration'] - offset, resolution)
                    ]
                )
                tgt.extend(
                    [
                        time2token(utts[s]['offset'] - offset, resolution),
                        ' ' + utts[s]['tgt'],   # add a space before text
                        time2token(utts[s]['offset'] + utts[s]['duration'] - offset, resolution)
                    ]
                )
            
            prev_src = []
            prev_tgt = []
            for c in context:
                prev_src.append(utts[c]['src'])
                prev_tgt.append(utts[c]['tgt'])

            long_utt = {
                'start': utts[samples[0]]['offset'],
                'end': utts[samples[-1]]['offset'] + utts[samples[-1]]['duration'],
                'wav': utts[samples[0]]['wav'],
                'src': ''.join(src),    # no space between special tokens
                'tgt': ''.join(tgt),
                'prev_src': ' '.join(prev_src),
                'prev_tgt': ' '.join(prev_tgt),
            }
            long_utt['wav_id'] = f"ted_{int(long_utt['wav'].split('/')[-1][:-len('.wav')][len('ted_'):]):05d}"
            long_utt['utt_id'] = f"{long_utt['wav_id']}_{int(1000*long_utt['start']):07d}_{int(1000*long_utt['end']):07d}"
            long_utt['spk_id'] = long_utt['wav_id']

            new_utts.append(long_utt)
    
    return new_utts


######################################################################
def prepare_all(root, out_root, datasets, langs, max_sec, resolution):
    for dataset in datasets:    # 'train', 'dev'
        out_dir = out_root / f"{dataset}"
        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)

        wavscp_fp = open(out_dir / "wav.scp", 'w')      # wav-id path
        wavids = []
        segments_fp = open(out_dir / "segments", 'w')   # utt-id wav-id start end
        text_fp = open(out_dir / "text", 'w')
        utt2spk_fp = open(out_dir / "utt2spk", 'w')
        for lang in langs:
            wav2utts = parse_data(root, lang, dataset)
            for utts in wav2utts.values():
                new_utts = generate_samples(utts, max_sec=max_sec, resolution=resolution)
                for u in new_utts:
                    print(dataset, lang, u['utt_id'])

                    spk_id = u['spk_id']
                    wav_id = f"{u['wav_id']}_en-{lang}"
                    if wav_id not in wavids:
                        wavscp_fp.write(f"{wav_id} {u['wav']}\n")
                        wavids.append(wav_id)

                    # transcribe
                    utt_id = f"{u['utt_id']}_en-{lang}_transcribe"
                    if u['prev_src']:
                        trans = f"<startofprev> {u['prev_src']}<startoftranscript><transcribe>{u['src']}<endoftext>"
                    else:
                        trans = f"<startoftranscript><transcribe>{u['src']}<endoftext>"
                    text_fp.write(f"{utt_id} {trans}\n")
                    utt2spk_fp.write(f"{utt_id} {spk_id}\n")
                    segments_fp.write(f"{utt_id} {wav_id} {u['start']:.2f} {u['end']:.2f}\n")

                    # translate
                    utt_id = f"{u['utt_id']}_en-{lang}_translate"
                    if u['prev_tgt']:
                        trans = f"<startofprev> {u['prev_tgt']}<startoftranscript><translate{lang}>{u['tgt']}<endoftext>"
                    else:
                        trans = f"<startoftranscript><translate{lang}>{u['tgt']}<endoftext>"
                    text_fp.write(f"{utt_id} {trans}\n")
                    utt2spk_fp.write(f"{utt_id} {spk_id}\n")
                    segments_fp.write(f"{utt_id} {wav_id} {u['start']:.2f} {u['end']:.2f}\n")
        
        wavscp_fp.close()
        segments_fp.close()
        text_fp.close()
        utt2spk_fp.close()


if __name__ == "__main__":
    data_root = Path('/scratch/bbjs/peng6/corpora/MuST-C')
    out_root = Path('./data')
    # wav2utts = parse_data(data_root, 'zh', 'dev')
    # print(generate_samples(list(wav2utts.values())[0])[:5])
    prepare_all(
        root=data_root,
        out_root=out_root,
        datasets=['dev', 'train'],
        langs=['de', 'ja', 'zh'],
        max_sec=30,         # 30s
        resolution=0.02,    # 20ms
    )
