import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import imageio
from PIL import Image
import os
import glob
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, send_from_directory, render_template

app = Flask(__name__)

def get_model_upload():

    if len(glob.glob('model/*')) > 0:
        interpreter = tf.lite.Interpreter(model_path=glob.glob('model/*')[0])
        interpreter.allocate_tensors()
    else:
        name_app = str(np.random.choice(np.arange(0,10000)))
        model_lite_path = tf.keras.utils.get_file(f'my_model_{name_app}.tflite', 
                            'https://www.dropbox.com/s/bodh4db0j1bz8b4/lite_model_deepartist_rn50.tflite?dl=1', 
                            cache_dir='.', cache_subdir='model')
        interpreter = tf.lite.Interpreter(model_path=model_lite_path)
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    artists = {0: 'Vincent van Gogh',
            1: 'Edgar Degas', 
            2: 'Pablo Picasso', 
            3: 'Pierre-Auguste Renoir', 
            4: 'Albrecht Durer', 
            5: 'Paul Gaugin', 
            6: 'Francisco Goya', 
            7: 'Rembrandt', 
            8: 'Alfred Sisley', 
            9: 'Titian', 
            10: 'Marc Chagall'
                }

    def predict_upload(filepath):
        train_input_shape = (224, 224)
        uploaded_img = imageio.imread(filepath)
        uploaded_img = Image.fromarray(uploaded_img)
        uploaded_img = uploaded_img.resize(train_input_shape[0:2])
        uploaded_img = tf.keras.preprocessing.image.img_to_array(uploaded_img)
        uploaded_img /= 255.
        uploaded_img = np.expand_dims(uploaded_img, axis=0)
        
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_tensor= tf.convert_to_tensor(input_data, np.float32)
        interpreter.set_tensor(input_details[0]['index'], uploaded_img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction_probability = np.amax(prediction) * 100
        prediction_probability = "{0:.2f}%".format(prediction_probability)
        prediction_idx = np.argmax(prediction)
        output_data = {'artist': artists[prediction_idx], 'probability': prediction_probability}
        return output_data

    return predict_upload

model_upload = get_model_upload()

os.makedirs('uploads', exist_ok=True)

# prevent cached responses
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# default route
@app.route('/')
def index():
    #return "<center><b>Welcome to DeepArtist, the art-lover bot!<b></center>"
	return webpage('find-artist.html')

@app.route('/new')
def new():
    return webpage('find-artist.html')

@app.route('/classify')
def api_upload():
    filepath = glob.glob('uploads/*')[0]
    output_data = model_upload(filepath)
    response = jsonify(output_data)
    return response

@app.route('/uploads/<string:path>')
def show_upload(path):
    return send_from_directory('uploads/', path)

@app.route('/static/<string:path>')
def webpage(path):
    return send_from_directory('', path)

@app.route('/showimage')
def show():
    file = glob.glob('uploads/*')[0]
    return jsonify(file)

@app.route('/upload', methods=['POST'])
def predict():
    file = glob.glob('uploads/*')
    for i in file:
        os.remove(i)
    file_new = request.files['upload']
    filename = file_new.filename
    file_new.save(f'uploads/{secure_filename(filename)}')
    return webpage('classify.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8082)