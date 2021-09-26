from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from cgi import parse_qs, escape

import sys
sys.path.append("..") # Adds higher directory to python modules path.
# import ..

# from .. import load_chopped_clip, speech_analysis_helpers, local_constants
from ..load_chopped_clip import *
from ..speech_analysis_helpers import compute_stft, find_peaks, lpc_analysis
from ..local_constants import CURR_DIR_PATH

import uuid

# from .. import curr_dir_path
# from curr_dir_path import CURR_DIR_PATH

# Create your views here.
@csrf_exempt
def index(request):
    environ = request.environ
    try:
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
    except (ValueError):
        request_body_size = 0

    request_body = environ['wsgi.input'].read(request_body_size)

    file_uuid = uuid.uuid4().hex

    with open('audio_files/{}.wav'.format(file_uuid), mode='bx') as f:
        f.write(request_body)

    training_files = []
    sample_rate = 10000
    main_clip = load_clip('{}/smart_voice_server/audio_files/{}.wav'.format(CURR_DIR_PATH, file_uuid), sample_rate, 50000)

    f, t, vp_stft = compute_stft(main_clip, sample_rate, False)

    main_vp_stft_slice = find_peaks(f, t, vp_stft, visualize=False, figsize=(30, 5), lin_interp=False)
    main_vp_stft_slice_interp = find_peaks(f, t, vp_stft, visualize=False, figsize=(30, 5), lin_interp=True)

    formants = lpc_analysis(main_clip, sample_rate)

    print("Formants: {}".format(formants))

    return HttpResponse(str(formants))
