import kaldiio
import whisper
import numpy as np
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    model_tag = "medium"
    wavscp = "dump/raw/test/FLEURS/test/wav.scp"
    output_dir = f"whisper-{model_tag}_outputs/LID_FLEURS.test"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(model_tag, device="cuda")
    data = kaldiio.load_scp(wavscp)
    with open(Path(output_dir) / "pred", 'w') as fout:
        for key, (rate, audio) in tqdm(data.items()):
            # load audio and pad/trim it to fit 30 seconds
            # audio = whisper.load_audio("audio.mp3")
            audio = whisper.pad_or_trim(audio.astype(np.float32))

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            pred = max(probs, key=probs.get)
            tqdm.write(f"Input: {key}, detected language: {pred}")
            fout.write(f"{key} {pred}\n")
