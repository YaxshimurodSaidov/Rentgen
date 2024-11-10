from flask import Flask, request, render_template
from keras.models import load_model # type:ignore
from keras.preprocessing import image # type:ignore
import numpy as np
import os
from detect_xray import is_xray_image # type:ignore

app = Flask(__name__)

# Modelni yuklash
old_model = load_model('model.keras')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Bosh sahifa
@app.route('/')
def home():
    return render_template('index.html')

# Rasm yuklash va tahlil qilish
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "Fayl tanlanmadi!"
    file = request.files['file']
    if file.filename == '':
        return "Fayl tanlanmadi!"
    
    # Rasmni saqlash
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Rasmni rentgenmi yoki yo‘q ekanligini tekshirish
    if is_xray_image(img_path):
        # Rentgen tasvirini yuklash va tahlil qilish
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizatsiya

        predictions = old_model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)

        if class_index[0] == 0:
            result = "Rentgen tasvirida kasallik yo'q."
        else:
            result = "Rentgen tasvirida kasallik bor."
    else:
        result = "Bu rasm rentgen tasviri emas."

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
