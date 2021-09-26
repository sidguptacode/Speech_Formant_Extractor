import librosa
import os

from pydub import AudioSegment
from pydub.silence import split_on_silence

"""
    Loads a speech clip at a file path, and returns the list representation.
"""
def load_clip(path, sr, chop=0, chop_with_none=False):
    clip = os.path.join('..', 'data', 'B', path)
    x, sr = librosa.load(clip, sr)
    if chop:
        if chop_with_none and len(x) < chop:
            return None
        diff = len(x) - chop
        x = x[diff // 2: len(x) - diff // 2]
    return x


def load_chopped_clip(dir_path, clip_filename, sample_rate):

    clip_path = "{}/{}".format(dir_path, clip_filename)
    chopped_path = "{}/chopped_sound_clips/chopped_{}".format(dir_path, clip_filename)
    
    # Populate the sound segments for this file.
    sound = AudioSegment.from_wav(clip_path)
    lib_segments = split_on_silence(sound,
      # must be silent for at least 250 ms
      min_silence_len=250,

      # consider it silent if quieter than -15 dBFS
      silence_thresh=-15
    )
    if not lib_segments:
        return None
    


    lib_segments[0].export(chopped_path, format="wav")
    chopped_clip = load_clip(chopped_path, sample_rate)
    return chopped_clip
