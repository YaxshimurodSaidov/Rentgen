# detect_xray.py
import numpy as np
from tensorflow.keras.applications import ResNet50 # type:ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions # type:ignore
from tensorflow.keras.preprocessing import image # type:ignore

# ResNet50 modelini yuklash
classifier_model = ResNet50(weights='imagenet')

def is_xray_image(img_path):
    """Tasvir rentgenmi yoki yoq ekanligini tekshiradi."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = classifier_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Yuqori natijalarni 'x-ray' so‘ziga nisbatan tekshiradi
    for _, label, _ in decoded_predictions:
        if 'x-ray' in label.lower():
            return True
    return False
