import numpy as np
from PIL import Image

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = MobileNetV2(weights='imagenet')
img = Image.open('./img/image.jpg')

def model_predict(img, model):
    img = img.resize((224, 224))  
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    
    preds = model.predict(x)
    return preds

preds = model_predict(img, model)
preds_probs = "{:.3f}".format(np.amax(preds))
preds_class = decode_predictions(preds, top=1)

result = str(preds_class[0][0][1])
result = result.replace('_', ' ').capitalize()
print({"result": result, "probability": preds_probs})