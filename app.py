from flask import Flask, request, render_template

import random
import os
import numpy as np

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

app = Flask(__name__)

# Load the checkpoint. https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP.
# Also available here: https://share.weiyun.com/V2suEUVh (Chinese users).
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

@app.route('/', methods =["GET", "POST"])
# def my_form():
#     return render_template('home.html')
def gfg():
    if request.method == "POST":
       if(not request.files["audio1"] or not request.files["audio2"]):
           return render_template("home.html" , variable= 0)
       # getting input with name = fname in HTML form
       aud1 = request.files["audio1"]
       # getting input with name = lname in HTML form
       aud2 = request.files["audio2"]
       # Sample some inputs for WAV/FLAC files for the same speaker.
       # To have reproducible results every time you call this function, set the seed every time before calling it.
       # np.random.seed(123)
       # random.seed(123)
       mfcc_001 = sample_from_mfcc(read_mfcc(aud1, SAMPLE_RATE), NUM_FRAMES)
       mfcc_002 = sample_from_mfcc(read_mfcc(aud2, SAMPLE_RATE), NUM_FRAMES)

       # Call the model to get the embeddings of shape (1, 512) for each file.
       predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
       predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))
    
       return render_template("home.html", variable = (batch_cosine_similarity(predict_001, predict_002)))# SAME SPEAKER [0.81564593]
    return render_template("home.html" , variable= 0)
if __name__ == '__main__':
    app.run(debug=True,port=8000)
