#!/usr/bin/python3

import openai
import requests, json
import click, threading, queue, pyaudio, audioop, time, io, signal, sys, os
import sounddevice as sd
import soundfile as sf
import numpy as np

from math import log10

################### Global Parameters #########################################

SOUND_THRESHOLD = -45
MAX_BINS=1000
DUMP_INTERVAL=0.5
MAX_SOUND_LENGTH = 10

###############################################################################


stopped = False
def sigint_handler(sig, frame):
    global stopped

    stopped = True


CONVERT = queue.Queue()
CONTROL = queue.Queue()

SAMPLE_RATE=16000

class MutableNumber:
    def __init__(self, val):
        self.val = val

    def set(self, val):
        self.val = val

    def get(self):
        return self.val

    def increment(self):
        self.val += 1


def dprint_constructor(func: str, debug: str):
    _func = func.upper()

    if debug == 'all' or func == debug:
        dprint = (lambda s: 
            print(f' {_func:<7}: {s}'))
    else:
        dprint = lambda s: None

    return dprint

################### Transcribers ##############################################

def naver_transcribe(out: io.BytesIO):
    Lang = "Kor"
    URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + Lang

    ID = os.environ["NAVER_CLIENT_ID"]
    SECRET = os.environ["NAVER_CLIENT_SECRET"]

    headers = {
        "Content-Type": "application/octet-stream",
        "X-NCP-APIGW-API-KEY-ID": ID,
        "X-NCP-APIGW-API-KEY": SECRET,
    }

    start = time.time()
    response = requests.post(URL, data=out, headers=headers)
    delay = time.time() - start

    rescode = response.status_code

    if (rescode == 200):
        return json.loads(response.text)["text"], delay
    else:
        raise Exception(f"Naver api error: {rescode}")


def openai_transcribe(out: io.BytesIO):
    start = time.time()
    transcript = openai.Audio.transcribe('whisper-1', out)
    delay = time.time() - start

    return transcript['text'], delay


TRANSCRIBERS = {
    'openai': openai_transcribe,
    'naver': naver_transcribe
}

###########################################################################


def stream_sound(dest: queue.Queue, debug: str):

    dprint = dprint_constructor('stream', debug)

    p = pyaudio.PyAudio()

    WIDTH = 2
    RATE = int(p.get_default_input_device_info()["defaultSampleRate"])
    DEVICE = p.get_default_input_device_info()["index"]

    WARMUP = 100

    db = MutableNumber(-100)
    warmup = MutableNumber(0)
    def callback(in_data, frame_count, time_info, status):
        rms = audioop.rms(in_data, WIDTH) / 32767
        try:
            _db = db.get()
            db.set(((_db * (MAX_BINS - 1)) + (20 * log10(rms)))/MAX_BINS)

            warmup.increment()
        except:
            pass

        return in_data, pyaudio.paContinue

    stream = p.open(
        format=p.get_format_from_width(WIDTH),
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=DEVICE,
        stream_callback=callback,
    )
    stream.start_stream()

    print(f'[*]   PyAudio starts... (WIDTH: {WIDTH}, RATE: {RATE}, DEVICE:{DEVICE})')

    last_stream_time = MutableNumber(0)
    last_dump_time = MutableNumber(0)
    sd_queue = queue.Queue()

    def sd_save(indata, frames, sd_time, status):
        sd_queue.put(indata.copy())

        if sd_time.currentTime > last_dump_time.get() + DUMP_INTERVAL:

            dest.put(list(sd_queue.queue))
            last_dump_time.set(sd_time.currentTime)

            if sd_time.currentTime > last_stream_time.get() + MAX_SOUND_LENGTH:
                sd_queue.queue.clear()
                last_stream_time.set(sd_time.currentTime)

    def sd_save_finished():
        dest.put(list(sd_queue.queue))
        sd_queue.queue.clear()

    sd_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, 
        dtype="int16", 
        channels=1, 
        callback=sd_save,
        finished_callback=sd_save_finished)

    print(f'[*]   Stream sounds...\n' +
          f'        SOUND_THRESHOLD:  {SOUND_THRESHOLD}\n' +
          f'        MAX_BINS:         {MAX_BINS}\n' +
          f'        DUMP_INTERVAL:    {DUMP_INTERVAL}\n' + 
          f'        MAX_SOUND_LENGTH: {MAX_SOUND_LENGTH}\n')
    CONVERT.put(True)

    while (not stopped) and stream.is_active():
        if warmup.get() < WARMUP:
            continue

        _db = db.get()
        dprint(f'db: {_db}')

        if _db > SOUND_THRESHOLD:
            if sd_stream.stopped:
                dprint(f'record start')

                last_stream_time.set(time.time())
                last_dump_time.set(time.time())

                sd_stream.start()

        elif not sd_stream.stopped:
            dprint(f'record stop')

            sd_stream.stop()

        time.sleep(0.1)

    sd_stream.close()

    stream.stop_stream()
    stream.close()

    p.terminate()


def convert_sound(src: queue.Queue, dest: queue.Queue, 
                  transcriber: str, debug: str):
    dprint = dprint_constructor('convert', debug)

    transcribe = TRANSCRIBERS[transcriber]

    def go_transcribe(sounds: np.array):

        out = io.BytesIO()
        out.name = 'out.wav'

        sf.write(out, sounds, samplerate=SAMPLE_RATE, format='wav')
        out.seek(0)

        transcript, delay = transcribe(out)
        dprint(f"[{transcriber}] delay: {delay:.2f} | {transcript}")

        dest.put(transcript)


    CONVERT.get()

    print(f'[*]   Convert sounds...\n' + 
          f'        using {transcriber}\n')
    CONTROL.put(True)

    TIMEOUT=1
    transcribe_threads = []
    while not stopped:
        try:
            sounds = src.get(timeout=TIMEOUT)
        except queue.Empty:
            for t in transcribe_threads:
                t.join()

            transcribe_threads.clear()

            continue

        sounds = np.concatenate(sounds)

        t = threading.Thread(target=go_transcribe,
                             args=(sounds,))
        transcribe_threads.append(t)
        t.start()

    for t in transcribe_threads:
        t.join()


def control_arduino(src: queue.Queue, debug: str):
    CONTROL.get()

    print(f'[*]   Control arduino...\n')
    pass


@click.command()
@click.option('--transcriber', '-t', default='openai',
              type=click.Choice(['openai', 'naver']),
              help='Transcript service to use from')
@click.option('--debug', '-d', default=None, 
              type=click.Choice(['all', 'stream', 'convert', 'control']), 
              help='Debug option for PICANTE')
def main(transcriber, debug):
    print('[*] Hello PICANTE!')

    if transcriber == 'openai':
        if not 'OPENAI_API_KEY' in os.environ:
            print('[-] You should set OPENAI_API_KEY to run PICANTE!\n' +
                  '      Please run "export OPENAI_API_KEY=<openai-api-key>"')

            sys.exit(1)
    elif transcriber == 'naver':
        if ((not 'NAVER_CLIENT_ID' in os.environ) or 
            (not 'NAVER_CLIENT_SECRET' in os.environ)):
            print('[-] You should set NAVER_CLIENT_ID, NAVER_CLIENT_SECRET to run PICANTE!\n' +
                        'Please run "export NAVER_CLIENT_ID=<naver-client-id>\n"' +
                        'Please run "export NAVER_CLIENT_SECRET=<naver-client-secret>\n"')
            sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    sound_queue = queue.Queue()
    text_queue = queue.Queue()

    streamer = threading.Thread(target=stream_sound, 
                                args=(sound_queue,debug))
    streamer.start()

    converter = threading.Thread(target=convert_sound, 
                                 args=(sound_queue, text_queue, transcriber, debug))
    converter.start()

    controller = threading.Thread(target=control_arduino, 
                                  args=(text_queue, debug))
    controller.start()

    streamer.join()
    converter.join()
    controller.join()

    print('[*] Bye PICANTE...')


if __name__ == '__main__':
    main()
