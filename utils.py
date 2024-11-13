import os
import librosa
import requests
from io import BytesIO
import numpy as np
# --------------------
import contextlib
import importlib

from inspect import isfunction
import os
import soundfile as sf
import time
import wave

# import urllib.request
# import progressbar

def get_duration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth
       
def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_wave(waveform, savepath, name="outwav"):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        path = os.path.join(
            savepath,
            "%s_%s.wav"
            % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            ),
        )
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=16000)

def round_up_duration(duration):
    return int(round(duration/2.5) + 1) * 2.5









# # ----------------------------------------
# def load_audio_from_file(file_path, sr=16000):
#     """파일 경로에서 오디오를 로드합니다."""
#     audio, sr = librosa.load(file_path, sr=sr)
#     return audio

# def load_audio_from_url(url, sr=16000):
#     """URL에서 오디오를 로드합니다."""
#     response = requests.get(url)
#     response.raise_for_status()
#     audio_bytes = BytesIO(response.content)
#     audio, samplerate = sf.read(audio_bytes, dtype='float32')
#     if sr is not None and samplerate != sr:
#         audio = librosa.resample(audio, orig_sr=samplerate, target_sr=sr)
#     return audio, sr

# def path_exists(location):
#     return os.path.exists(location)

# def get_absolute_path(location):
#     return os.path.abspath(location)

# def is_valid_url(location):
#     try:
#         response = requests.head(location)
#         return response.status_code == 200
#     except:
#         return False

# def load_audio(location, *):
#     """입력된 location이 URL인지 파일 경로인지 자동으로 감지하여 오디오 파일을 로드합니다."""
#     assert isinstance(location, str), 'load_audio 오류: location은 URL 또는 파일 경로를 나타내는 문자열이어야 합니다. 그러나 type(location)=={}이고, location=={}'.format(type(location), location)
    
#     if path_exists(location):
#         location = get_absolute_path(location)
#     if is_valid_url(location):
#         out = load_audio_from_url(location)
#     else:
#         out = load_audio_from_file(location)
#     return out

# # ---
# def as_numpy_array(x):
#     try:return np.asarray(x)#For numpy arrays and python lists
#     except Exception:pass
#     try:return x.detach().cpu().numpy()#For pytorch.
#     except Exception:pass
#     assert False,'as_numpy_array: Error: Could not convert x into a numpy array. \
#         type(x)='+repr(type(x))+' and x='+repr(x)
