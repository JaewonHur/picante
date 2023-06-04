import queue, os, threading, time, pyaudio, audioop
import signal, sys
import serial, socket, select

from math import log10
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

from pcm_channels import pcm_channels

INTERVAL = 0.1
SOUND_CNT = 5
SOUND_THRESHOLD = -35

HOST = "127.0.0.1"
PORT = 65432

q = queue.Queue()
sound_detected = 0
recorder = False
recording = False
rms = 1


def signal_handler(sig, frame):
    print("Received ctrl-c")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def detect_sound():
    global sound_detected

    p = pyaudio.PyAudio()
    WIDTH = 2
    RATE = int(p.get_default_input_device_info()["defaultSampleRate"])
    DEVICE = p.get_default_input_device_info()["index"]
    print(p.get_default_input_device_info())

    def callback(in_data, frame_count, time_info, status):
        global rms
        rms = audioop.rms(in_data, WIDTH) / 32767
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

    i = 0
    while stream.is_active():
        try:
            db = 20 * log10(rms)
        except:
            continue

        if db > SOUND_THRESHOLD:
            sound_detected += 1 if sound_detected < SOUND_CNT else 0
        else:
            sound_detected -= 1 if sound_detected > 0 else 0

        if i % 10 == 0:
            print(f'RMS: {rms}, db: {db}, sound_detected: {sound_detected}')
        i += 1

        time.sleep(INTERVAL)

    stream.stop_stream()
    stream.close()

    p.terminate()


def complicated_record(i: int):
    with sf.SoundFile(
        f"sound-files/sound-{i}.wav",
        mode="w",
        samplerate=16000,
        subtype="PCM_16",
        channels=1,
    ) as file:
        with sd.InputStream(
            samplerate=16000, dtype="int16", channels=1, callback=complicated_save
        ):
            while recording:
                file.write(q.get())


def complicated_save(indata, frames, time, status):
    q.put(indata.copy())


def control_arduino():
    py_serial = serial.Serial(
        port='COM9',
        baudrate=9600,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        while True:
            s.sendall(b"SEND DATA")

            r, _, _ = select.select([s], [], [])
            if r:
                inb = s.recv(1024)
                for i in inb:
                    py_serial.write(i.to_bytes(1, byteorder='big'))
                py_serial.write(b'\xFF')

            time.sleep(INTERVAL)


def main():
    detector = threading.Thread(target=detect_sound)
    detector.start()

    controller = threading.Thread(target=control_arduino)
    controller.start()

    for fn in os.listdir("sound-files"):
        os.remove(f"sound-files/{fn}")

    i = 0
    while True:
        #   # TODO: waiting for sound
        if sound_detected > 0:
            global recording

            recording = True
            recorder = threading.Thread(target=complicated_record, args=(i,))
            recorder.start()
            while sound_detected > 0:
                time.sleep(1)
            recording = False
            recorder.join()

            print(f"Finished recording {i}")

            try:
                os.remove(f"sound-files/sound-{i-10}.pcm")
                os.remove(f"sound-files/sound-{i-10}.wav")
            except:
                pass

            i += 1

        time.sleep(INTERVAL)

    detector.join()
    controller.join()


if __name__ == "__main__":
    main()
