# Web
from flask import Flask, render_template

# Processing
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf


# Initialize app and set secret key
app = Flask(__name__)

# Load model globally
model = load_model("static/small_cnn_multilabel.h5")


# Img reading, processing and inference
def predict():
	print ('FLLAAGG: entered get_receipt function')
	# Processing
	#np.random.seed(0)
	img = np.random.random((1, 128,  128, 3))
	print ('FLLAAGG: generated img array')
	# Inference
	pred = model.predict(img, batch_size=1)
	print ("finished prediction")
	print ('FLLAAGG: finished inference')

	return pred

	
# Routes 
@app.route('/')
def hello_world():
    return 'Hello! Just making sure everything is working properly, now go to /serve'

@app.route('/serve', methods=["POST", "GET"])
def serve():
	print ('FLLAAGG: entered serve page')
	pred = predict()
	print ('FLLAAGG: exited function')
	return str(pred)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
