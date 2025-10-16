import os
import tensorflow as tf
from config import DATASET_CONFIG, IMAGE_SIZE, CHANNELS, DATA_DIR, BATCH_SIZE

def preprocess_image(image_path, label):
    """Loads, resizes, and normalizes a single image file."""
    # 1. Load: Read the image file from disk
    img = tf.io.read_file(image_path)
    
   #images are stored as 0 and 1s so this function basically converts those into a grid tensor flow
   #can understand
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    
    # 3. Resize: Ensure all images are the standard size (224x224)
    img = tf.image.resize(img, IMAGE_SIZE)
    
    ##all brightness values are normalized
    img = img / 255.0
    
    return img, label

def create_dataset_pipeline(dataset_key):
    """Gathers image paths and labels and creates an optimized TF Dataset."""
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise ValueError(f"Dataset key '{dataset_key}' not found in config.")
    
    local_path = os.path.join(DATA_DIR, config['local_name'])
    num_classes = config['num_classes']

    
    # This automatically infers classes based on subdirectory names
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=local_path,
        labels='inferred',
        label_mode='categorical', # Labels as one-hot vectors
        image_size=IMAGE_SIZE,    # Resized to standard size
        batch_size=None,          # Load individually for mapping
        shuffle=True
    )
    
    
    # Apply custom preprocessing function to every image in the dataset
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    #store the images so they dont have to be re read and process the next batch
    dataset = dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset