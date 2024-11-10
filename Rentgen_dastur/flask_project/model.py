import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type:ignore 

# Modelni yaratish
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # 2 ta klass (kasallik bor/yok)
    return model

# Modelni yaratish
model = create_model()

# Modelni kompilyatsiya qilish
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Ma'lumotlarni tayyorlash
train_datagen = ImageDataGenerator(rescale=1./255)  # Rasmni normalizatsiya qilish

# O'qitish ma'lumotlarini yuklash
train_generator = train_datagen.flow_from_directory(
    'data/train',  # O'qitish ma'lumotlari joylashgan papka
    target_size=(150, 150),  # Rasm o'lchamini o'zgartirish
    batch_size=32,
    class_mode='sparse'  # Kategoriyalar (kasallik bor yoki yo'q)
)

# Modelni o'qitish
model.fit(train_generator, epochs=10)

# Modelni saqlash
model.save('new_model.h5')  # Modelni saqlash
print("Model saqlandi: new_model.h5")
