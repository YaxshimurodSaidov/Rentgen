import tensorflow as tf
from tensorflow.keras import layers, models

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
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Model yaratish
input_shape = (150, 150, 3)  # Tasvir o'lchamlari (masalan, 150x150)
num_classes = 2  # Tasniflanadigan kasalliklar soni
model = create_model(input_shape)

# Modelni kompilyatsiya qilish
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Agar klasslar raqamli bo'lsa
              metrics=['accuracy'])

# Modelni ko'rsatish
model.summary()
