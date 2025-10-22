import os
import tensorflow as tf
import numpy as np
from PIL import Image # Essential for reading .tif files
from config import DATASET_CONFIG, IMAGE_SIZE, CHANNELS, DATA_DIR, BATCH_SIZE





def _decode_and_preprocess_image(path_tensor, label_tensor, img_size):
    """
    Decodes the image, handles TIF files, ensures 3-channels, and converts 
    to float32 for TensorFlow processing.
    """
    path = path_tensor.numpy().decode('utf-8')
    img_tensor = None
    
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            
            img = Image.open(path).convert("RGB")
            
            img_array = np.array(img, dtype=np.uint8)
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
        except Exception:
            
            img_array = np.zeros(img_size + (3,), dtype=np.uint8)
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
            
    else:
    
        img = tf.io.read_file(path)
        img_tensor = tf.image.decode_image(img, channels=3)
    
 
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
    

    img_tensor = tf.image.resize(img_tensor, img_size)
    
    img_tensor.set_shape(list(img_size) + [3])
    
    
    return img_tensor, label_tensor



def create_dataset_pipeline(dataset_key):
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise ValueError(f"Dataset key '{dataset_key}' not found in config.")
        
    
    base_path_from_config = os.path.join(DATA_DIR, config['local_name'])
    path_suffix = config.get('path_suffix')
    
    if path_suffix:
        final_path = os.path.join(base_path_from_config, path_suffix)
    else:
        final_path = base_path_from_config
        
    
    print(f"\nCollecting data for {dataset_key} from: {final_path}")
    
   
    try:
        class_names = sorted([d for d in os.listdir(final_path) if os.path.isdir(os.path.join(final_path, d))])
    except FileNotFoundError:
         raise FileNotFoundError(f"Directory not found: {final_path}. Check your config.py 'local_name' and 'path_suffix'.")
         
    num_classes = len(class_names)

    if num_classes != config['num_classes']:
         raise ValueError(
             f"CRITICAL ERROR: {dataset_key} found {num_classes} classes "
             f"({class_names}), but config expects {config['num_classes']}! Update config.py."
         )
    
 
    file_pattern = os.path.join(final_path, '*', '*') 
    
    list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    
    
    label_map = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(class_names, tf.constant(list(range(num_classes)), dtype=tf.int32)), 
        default_value=-1
    )
    
    def get_path_and_label(file_path):
       
        parts = tf.strings.split(file_path, os.path.sep)
        label_str = parts[-2]
        
        
        int_label = label_map.lookup(label_str)
        one_hot_label = tf.one_hot(int_label, num_classes)
        
        return file_path, one_hot_label

    
    dataset = list_ds.map(get_path_and_label, num_parallel_calls=tf.data.AUTOTUNE)


    def wrapper_fn(path, label):
        return tf.py_function(
            func=_decode_and_preprocess_image,
            inp=[path, label, IMAGE_SIZE],
            Tout=[tf.float32, tf.float32] 
        )
        
   
    final_ds = dataset.map(wrapper_fn, num_parallel_calls=tf.data.AUTOTUNE)
    final_ds = final_ds.map(
        lambda image, label: (
            tf.ensure_shape(image, IMAGE_SIZE + (CHANNELS,)),
            tf.ensure_shape(label, tf.TensorShape([config['num_classes']]))
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    final_ds = final_ds.cache().shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return final_ds




if __name__ == '__main__':
    # Test all three final tasks
    TEST_DATASET_KEYS = ["UCMERCED", "MED_WASTE", "AID"] 
    
    for key in TEST_DATASET_KEYS:
        print("="*60)
        print(f"--- STARTING DATA LOADER TEST FOR: {key} ---")
        
        try:
            pipeline = create_dataset_pipeline(key)
            
            # Use .take(1) to check the pipeline quickly
            for images, labels in pipeline.take(1):
                print(f"\n✅ SUCCESS: Pipeline created for {key}.")
                print(f"   Batch Image Shape: {images.shape}")
                print(f"   Batch Label Shape: {labels.shape}")
                print(f"   Inferred Classes (from pipeline): {labels.shape[1]}")
                break 
                
        except Exception as e:
            print(f"\n❌ FATAL ERROR: Data loading failed for {key}.")
            print(f"   Error: {e}")
            break