
from flask import Flask, request, render_template # type: ignore
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import image
import numpy as np

# Flask ilovasini yaratish
app = Flask(__name__)

# Modelni yuklash
model = tf.keras.models.load_model('model.h5')  # .h5 formatidagi modelni yuklash

@app.route('/')
def upload_form():
    return render_template('index.html')  # Yuklash interfeysini ko'rsatadi

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Faylni saqlash
    file_path = 'uploads/' + file.filename
    file.save(file_path)
    
    # Tasvirni tayyorlash
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize qilish
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension qo'shish
    
    # Model orqali tahlil qilish
    prediction = model.predict(img_array)
    class_labels = ['NORMAL', 'PNEUMONIA']  # Tasniflar
    predicted_class = class_labels[np.argmax(prediction)]

    return f'Tasdiqlangan kasallik: {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)
