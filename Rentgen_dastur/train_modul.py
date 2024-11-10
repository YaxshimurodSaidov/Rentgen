import tensorflow as tf
from load_dataset import load_dataset # type: ignore
from Model import class_weights, create_model # type: ignore

# Datasetni yuklang
base_dir = r'C:\Users\User\Desktop\Universitet 2_kurs\suniy intellekt asoslari\chest_xray'
train_ds, val_ds, test_ds = load_dataset()  # 3 ta datasetni yuklaymiz

# Modelni yuklang
input_shape = (150, 150, 3)  # Tasvir o'lchamlari
model = create_model(input_shape)

# Modelni kompilyatsiya qilish
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Agar klasslar raqamli bo'lsa
              metrics=['accuracy'])

# Modelni o'qitish
history = model.fit(
    train_ds,
    validation_data=val_ds,  # Validatsiya uchun qo'shildi
    epochs=50,
    class_weight=class_weights  # Class weightni o'qitishda ishlatish
)

# Test datasetdan natijalarni baholash
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test natijalari: yo'qotish = {test_loss}, aniqlik = {test_accuracy}")

# Modelni saqlash
model.save('model.keras')
