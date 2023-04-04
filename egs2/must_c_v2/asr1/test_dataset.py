from espnet2.train.preprocessor import WhisperPreprocessor
from espnet2.train.dataset import ESPnetDataset


if __name__ == "__main__":
    preprocess = WhisperPreprocessor(
        train=True,
        token_type='bpe',
        token_list='/scratch/bbjs/peng6/espnet-whisper/egs2/must_c_v2/asr1/data/token_list/bpe_unigram10000/tokens.txt',
        bpemodel='/scratch/bbjs/peng6/espnet-whisper/egs2/must_c_v2/asr1/data/token_list/bpe_unigram10000/bpe.model',
        fs=16000,
        max_sec=30,
        resolution_sec=0.02,
        max_init_silence_sec=1.0,
        prev_apply_prob=1.0,
        timestamp_apply_prob=0.0,
    )
    
    ds = ESPnetDataset(
        [
            ('/scratch/bbjs/peng6/espnet-whisper/egs2/must_c_v2/asr1/dump/raw/dev/wav.scp', 'speech', 'kaldi_ark'),
            ('/scratch/bbjs/peng6/espnet-whisper/egs2/must_c_v2/asr1/dump/raw/dev/text', 'text', 'text')
        ],
        preprocess=preprocess,
    )

    # uid = 'ted_00824_0791340_0820600_en-zh_translate'
    # uid = 'ted_00824_0791340_0820600_en-zh_transcribe'
    uid = 'ted_00767_0016079_0044610_en-de_transcribe'  # no prev

    ds[uid]

    # breakpoint()
