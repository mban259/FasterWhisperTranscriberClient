"""メイン 録音, REST APIへのpost, 出力を行う"""
import argparse
import json
import queue
import threading
import traceback
from typing import Final

import numpy as np
import requests
import soundcard as sc

# 録音のサンプルレート Whisperは16kHz
WHISPER_SAMPLING_RATE: Final[int] = 16000
# 1回の録音の長さ(秒)
RECORD_SEC: Final[float] = 0.5
# 音量がこれ以上だと音が鳴っているとする
THRESHOLD_VOLUME: Final[np.float32] = 0.3

URL:str = None
que: queue.Queue = queue.Queue()


def transcribe():
    """キュー queに来た波形データをPOSTして標準出力する"""
    while True:
        wave: np.ndarray[np.float32] = que.get()
        js = json.dumps({"audio": wave.tolist()})
        try:
            response = requests.post(url=URL, data=js, timeout=20)
            res_json = response.json()
            lang = res_json["language"]
            for seg in res_json["segments"]:
                text = seg["text"]
                print(f"{lang}:{text}")
        except requests.Timeout:
            traceback.print_exc()


def is_speaking(wave: np.ndarray[np.float32]) -> bool:
    """波形データwaveの振幅が閾値より大きいならTrueを返す(音が鳴っている)"""
    return np.max(np.abs(wave)) > THRESHOLD_VOLUME


def main():
    """メイン関数"""
    buffer: np.ndarray[np.float32] = np.array((0,), np.float32)
    flag_speaking: bool = False
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True) \
            .recorder(samplerate=WHISPER_SAMPLING_RATE, channels=1) as mic:
        while True:
            raw: np.ndarray[np.float32] = mic.record(
                int(WHISPER_SAMPLING_RATE * RECORD_SEC)).reshape(-1)
            if not flag_speaking:
                if is_speaking(raw):
                    buffer = raw
                    flag_speaking = True
            else:
                if is_speaking(raw):
                    buffer = np.concatenate([buffer, raw])
                else:
                    que.put(buffer)
                    flag_speaking = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    args = parser.parse_args()
    URL = args.url
    thread = threading.Thread(target=transcribe, daemon=True)
    thread.start()
    main()
