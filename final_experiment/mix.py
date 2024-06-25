import numpy as np
import wave
from pydub import AudioSegment
import noisereduce as nr

def read_wave_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16)
    return params, audio

def write_wave_file(file_path, params, audio):
    with wave.open(file_path, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(audio.tobytes())

def convert_mp3_to_wav(mp3_file):
    wav_file = "background_temp.wav"
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    return wav_file

def mix_audio_files(speech_audio, background_audio, background_volume=0.4):
    # Adjust background volume
    background_audio = background_audio * background_volume
    print (speech_audio.shape, background_audio.shape)
    # Ensure both audio files have the same length
    if len(speech_audio) > len(background_audio):
        background_audio = np.pad(background_audio, (0, len(speech_audio) - len(background_audio)), 'wrap')
    else:
        background_audio = background_audio[:len(speech_audio)]

    # Mix audio
    
    mixed_audio = speech_audio + background_audio
    mixed_audio = np.clip(mixed_audio, -32768, 32767)  # Ensure values are within int16 range

    # Write mixed audio to output file
    output_file = "combined_audio.wav"
    # write_wave_file(output_file, speech_params, mixed_audio)

    return mixed_audio