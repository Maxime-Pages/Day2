from flask import Flask, request
import numpy as np
from PIL import Image

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


app = Flask(__name__)
model = MobileNetV2(weights='imagenet')


def model_predict(img, model):
    img = img.resize((224, 224))  
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    
    preds = model.predict(x)
    return preds

@app.route("/")
def index():
    return "Welcome to the Image Processing API!"

@app.route("/predict", methods=['POST'])
def predict():

    if not request.files:
        return {"error": "No file part"}, 400
    file = next(iter(request.files.values()))
    
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    try:
        img = Image.open(file)
        preds = model_predict(img, model)
        preds_probs = "{:.3f}".format(np.amax(preds))
        preds_class = decode_predictions(preds, top=1)
        
        result = str(preds_class[0][0][1])
        result = result.replace('_', ' ').capitalize()
        
        return {"result": result, "probability": preds_probs}
    
    except Exception as e:
        return {"error": str(e)}, 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081,debug=True)
