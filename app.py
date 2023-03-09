from flask import Flask, request, render_template

import random
import librosa
import numpy as np
import tensorflow as tf

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel(pcm_input=True)

app = Flask(__name__)

# Define the model here.
model = DeepSpeakerModel(pcm_input=True)
# Load the checkpoint. https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP.
# Also available here: https://share.weiyun.com/V2suEUVh (Chinese users).
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

@app.route('/', methods =["GET", "POST"])
# def my_form():
#     return render_template('home.html')
def gfg():
    if request.method == "POST":
       # Reproducible results.
       np.random.seed(123)
       random.seed(123)
       if(not request.files["audio1"] or not request.files["audio2"]):
           return render_template("home.html" , variable= 0)
       # getting input with name = fname in HTML form
       aud1 = request.files["audio1"]
       # getting input with name = lname in HTML form
       aud2 = request.files["audio2"]
       samples = [
            aud1,
            aud2
          ]

       pcm = [librosa.load(x, sr=SAMPLE_RATE, mono=True)[0] for x in samples]

        # Crop samples in the center, to fit the smaller audio samples
       num_samples = min([len(x) for x in pcm])
       pcm = tf.convert_to_tensor(np.stack([x[(len(x) - num_samples) // 2:][:num_samples] for x in pcm]))

        # Call the model to get the embeddings of shape (1, 512) for each file.
       predict = model.m.predict(pcm)
       same_speaker_similarity = batch_cosine_similarity(predict[0:1], predict[1:])

       return render_template("home.html", variable = same_speaker_similarity)# SAME SPEAKER [0.81564593]
    return render_template("home.html" , variable= 0)
if __name__ == '__main__':
    app.run(debug=True,port=8000)
