import os
import traceback
import librosa
import numpy as np
import av
from io import BytesIO
from pydub import AudioSegment

def wav2(i, o, format):
    inp = av.open(i, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"
    ostream = out.add_stream(format)
    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)
    for p in ostream.encode(None):
        out.mux(p)
    out.close()
    inp.close()

def audio2(i, o, format, sr):
    try:
        inp = av.open(i, "r")
        out = av.open(o, "w", format=format)
        if format == "ogg":
            format = "libvorbis"
        if format == "f32le":
            format = "pcm_f32le"
        ostream = out.add_stream(format, layout='mono')
        ostream.sample_rate = sr
        for frame in inp.decode(audio=0):
            for p in ostream.encode(frame):
                out.mux(p)
        out.close()
        inp.close()
    except av.error.InvalidDataError:
        # If chunked MP3, handle with pydub
        audio = AudioSegment.from_file(i)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(sr)
        audio.export(o, format='wav')

def load_audio(file, sr):
    file = (
        file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    if not os.path.exists(file):
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    try:
        # First try to load as a regular file
        with open(file, "rb") as f:
            with BytesIO() as out:
                try:
                    audio2(f, out, "f32le", sr)
                    return np.frombuffer(out.getvalue(), np.float32).flatten()
                except av.error.InvalidDataError:
                    # If that fails, try using pydub
                    audio = AudioSegment.from_file(file)
                    audio = audio.set_channels(1)  # Convert to mono
                    audio = audio.set_frame_rate(sr)
                    
                    # Convert to numpy array
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    samples = samples / (2**15)  # Normalize to [-1, 1]
                    return samples

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)
    except Exception as e:
        raise RuntimeError(traceback.format_exc())
