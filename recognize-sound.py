import time
import os
import random
import socket
import threading
import subprocess
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import DeepSpeech2

INTERVAL = 0.1

buffer = []

def parse_audio(
    audio_path: str, del_silence: bool = False, audio_extension: str = "pcm"
) -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = (
        torchaudio.compliance.kaldi.fbank(
            waveform=Tensor(signal).unsqueeze(0),
            num_mel_bins=80,
            frame_length=20,
            frame_shift=10,
            window_type="hamming",
        )
        .transpose(0, 1)
        .numpy()
    )

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


def save_chosung(s: str):
    global buffer

    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    CHOSUNG_TO_NUM = {
        "ㅅ": 6,
        "ㅌ": 1,
        "ㅋ": 2,
        "ㅍ": 3,
        "ㅎ": 4,
        "ㅊ": 5,
        "ㄹ": 0,
        "ㄴ": 7,
        "ㄷ": 8,
        "ㅂ": 9,
        "ㄱ": 10,
        "ㅇ": 11,
        "ㅁ": 12,
        "ㅈ": 13,
        "ㅃ": 9,
        "ㅆ": 6,
        "ㄲ": 10,
        "ㅉ": 13,
    }

    chosungs = []
    for w in list(s.strip()):
        if '가' <= w <= '힣':
            cho = (ord(w) - ord('가')) // 588
            chosungs.append(CHOSUNG_LIST[cho])

    print(f'chosungs: {chosungs}')

    chosungNums = [CHOSUNG_TO_NUM[i] for i in chosungs]
    noChosung = [CHOSUNG_TO_NUM[i] 
                 for i in ['ㅌ', 'ㄹ', 'ㄴ', 'ㄷ', 'ㅂ', 'ㄱ', 'ㅇ', 'ㅁ']]

    chosungNums = list(set(chosungNums) - set(noChosung))


    if chosungNums:
        buffer.append(bytes(chosungNums))

    if len(buffer) > 10:
        buffer.pop(0)


def server():
    HOST = "127.0.0.1"
    PORT = 65432

    print('Server on...')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        conn, addr = s.accept()
        with conn:
            print(f"Connected from {addr}")
            while True:
                data = conn.recv(1024)
                if data:
                    global buffer
                    while not buffer:
                        time.sleep(0.1)

                    inb = buffer[-1]
                    conn.sendall(inb)

                time.sleep(INTERVAL)

        s.close()


if __name__ == "__main__":
    sender = threading.Thread(target=server)
    sender.start()

    vocab = KsponSpeechVocabulary("kospeech/data/vocab/aihub_character_vocabs.csv")

    model = torch.load(
        "models/deepspeech-0505.pt", map_location=lambda storage, loc: storage
    ).to("cpu")
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    if isinstance(model, DeepSpeech2):
        model.device = "cpu"
    else:
        raise NotImplementedError()

    print("Start recognizing...")
    while True:
        try:
            nfile = max([int(i[:-4].split('-')[1]) for i in os.listdir("sound-files") if i.endswith("wav")])
            fname = f"sound-files/sound-{nfile}"

            subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    f"{fname}.wav",
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    f"{fname}.pcm",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            feature = parse_audio(f"{fname}.pcm", del_silence=True)
            input_length = torch.LongTensor([len(feature)])

            y_hats = model.recognize(feature.unsqueeze(0), input_length)

            sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
            save_chosung(sentence[0])

        except Exception as e:
            print(e)

        time.sleep(INTERVAL)

    sender.join()
