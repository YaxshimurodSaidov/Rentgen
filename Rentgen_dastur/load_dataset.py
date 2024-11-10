import tensorflow as tf

def load_dataset():
    base_dir = r'C:\Users\User\Desktop\Universitet 2_kurs\suniy intellekt asoslari\chest_xray'
    
    IMG_SIZE = (150, 150)
    BATCH_SIZE = 32

    # Train datasetini yuklash
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir + '/train/',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'  # Ikkita sinf: NORMAL va PNEUMONIA
    )

    # Validatsiya datasetini yuklash
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir + '/val/',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Test datasetini yuklash
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir + '/test/',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    return train_ds, val_ds, test_ds  # Train, val va test datasetlarini qaytaradi
