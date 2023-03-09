import os
import requests
import torchaudio


def GetParams(model):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d
    
    params = {}
    for name, prm in model.named_parameters():
        set_in_nested_dict(params, name.split("."), prm.detach().numpy())

    return params


def load_data():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()

    SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
    SPEECH_FILE = "_assets/speech.wav"

    if not os.path.exists(SPEECH_FILE):
        os.makedirs("_assets", exist_ok=True)
        with open(SPEECH_FILE, "wb") as file:
            file.write(requests.get(SPEECH_URL).content)

    waveform, sample_rate = torchaudio.load(SPEECH_FILE)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    params = GetParams(model)

    return waveform.detach().numpy(), params, bundle.get_labels()


