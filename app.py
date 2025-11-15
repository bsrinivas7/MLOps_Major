from flask import Flask, request, render_template
import numpy as np
import joblib
from PIL import Image
import io

app = Flask(__name__)
model = joblib.load('savedmodel.pth')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # convert to grayscale
    image = image.resize((64, 64))  # assuming input size is 64x64 to get 4096 pixels
    img_array = np.array(image).flatten()  # flatten to (4096,)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        img_file = request.files['image']
        img_bytes = img_file.read()
        arr = preprocess_image(img_bytes)
        prediction = int(model.predict([arr])[0])
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
