import os
import tensorflow as tf
from config import DATASET_CONFIG, IMAGE_SIZE, CHANNELS, DATA_DIR, BATCH_SIZE


def create_dataset_pipeline(dataset_key):
    """
    Gathers image data, fixes nested paths, and creates an optimized TF Dataset 
    ready for the DenseNet model.
    """
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise ValueError(f"Dataset key '{dataset_key}' not found in config.")
    
    
    base_data_path = os.path.join(DATA_DIR, config['local_name'])
    
    
    if dataset_key == "LC25000":
        
        final_path = os.path.join(base_data_path, 'Train and Validation Set')
    
    elif dataset_key == "SMOKE_CLASSIFICATION":
        
        final_path = os.path.join(base_data_path, 'train') 

    elif dataset_key == "PLANTS_TYPE":
        
        final_path = os.path.join(base_data_path, 'Test_Set_Folder') 
    
    else:
        final_path = base_data_path # Default path
    
    
    if not os.path.isdir(final_path):
        raise FileNotFoundError(
            f"Could not find required directory: {final_path}. "
            f"Please verify the exact subfolder names inside your 'data/' directory."
        )

    # 3. Keras Utility to load images and labels
    print(f"Loading data from: {final_path}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=final_path,
        labels='inferred',
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        batch_size=None,
        shuffle=True)
        
    
    
    # Keras loads image tensors (float32). We map a lambda function over the tensor.
    dataset = dataset.map(lambda image, label: (image / 255.0, label), num_parallel_calls=tf.data.AUTOTUNE)

    # 5. Optimization Pipeline
    dataset = dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# --- TEMPORARY TEST BLOCK ---
if __name__ == '__main__':
    TEST_DATASET_KEY = "PLANTS_TYPE"
    
    print(f"--- Running Data Loader Test for {TEST_DATASET_KEY} ---")
    
    try:
        pipeline = create_dataset_pipeline(TEST_DATASET_KEY)
        
        for images, labels in pipeline.take(1):
            print(f"\nSuccessfully created data pipeline for {TEST_DATASET_KEY}.")
            print(f"Batch Image Shape: {images.shape}")
            print(f"Batch Label Shape: {labels.shape}")
            print(f"Min Pixel Value: {images.numpy().min()}")
            print(f"Max Pixel Value: {images.numpy().max()}")
            break 
            
    except Exception as e:
        print(f"\n!!! ERROR during data pipeline creation: {e}")