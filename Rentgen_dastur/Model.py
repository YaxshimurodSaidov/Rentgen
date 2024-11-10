import tensorflow as tf
from keras import layers, models

def create_model(input_shape):
    model = models.Sequential()
    
    # 1-convolyutsion qatlam
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 2-convolyutsion qatlam
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 3-convolyutsion qatlam
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 4-convolyutsion qatlam
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # To'liq bog'langan qatlam
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Chiqish qatlami (kasalliklar soniga qarab o'zgartiring)
    model.add(layers.Dense(2, activation='softmax'))  # 2 klass uchun
    
    return model

# Class weights (agar sinflar o'rtasida muvozanatsizlik bo'lsa)
class_weights = {0: 1.0, 1: 2.0}  # NORMAL = 0, PNEUMONIA = 1
#http://127.0.0.1:5000/
