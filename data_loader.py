import os
import tensorflow as tf
import numpy as np
from PIL import Image # Essential for reading .tif files
import random # For the 70/10/20 split
from config import DATASET_CONFIG, IMAGE_SIZE, CHANNELS, DATA_DIR, BATCH_SIZE


def _decode_and_preprocess_image(path_tensor, label_tensor): # <-- FIXED: Removed img_size
    """
    Decodes the image, handles TIF files, ensures 3-channels, and converts 
    to float32 for TensorFlow processing.
    
    NOTE: Uses global IMAGE_SIZE imported from config.
    """
    path = path_tensor.numpy().decode('utf-8')
    img_tensor = None
    
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            
            img = Image.open(path).convert("RGB")
            
            img_array = np.array(img, dtype=np.uint8)
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
        except Exception:
            # Fallback to black image if TIF loading fails
            img_array = np.zeros(IMAGE_SIZE + (3,), dtype=np.uint8) # <-- Using global IMAGE_SIZE
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
            
    else:
    
        img = tf.io.read_file(path)
        img_tensor = tf.image.decode_image(img, channels=3)
    
 
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
    

    img_tensor = tf.image.resize(img_tensor, IMAGE_SIZE) # <-- Using global IMAGE_SIZE
    
    img_tensor.set_shape(list(IMAGE_SIZE) + [3]) # <-- Using global IMAGE_SIZE
    
    
    return img_tensor, label_tensor



def create_dataset_pipeline(dataset_key, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2): 
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise ValueError(f"Dataset key '{dataset_key}' not found in config.")
        
    # Check that ratios sum to 1.0 (with a small float tolerance)
    if not 0.999 <= (train_ratio + val_ratio + test_ratio) <= 1.001:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")
        
    
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
    
  
    
    all_file_paths = tf.io.gfile.glob(file_pattern)
    random.shuffle(all_file_paths)
    
    dataset_size = len(all_file_paths)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    train_paths = all_file_paths[:train_size]
    val_paths = all_file_paths[train_size:train_size + val_size]
    test_paths = all_file_paths[train_size + val_size:]
    
    train_list_ds = tf.data.Dataset.from_tensor_slices(train_paths)
    val_list_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    test_list_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    
    print(f"   Total Samples: {dataset_size}")
    print(f"   Train Samples: {len(train_paths)} ({train_ratio*100:.0f}%)")
    print(f"   Validation Samples: {len(val_paths)} ({val_ratio*100:.0f}%)")
    print(f"   Test Samples: {len(test_paths)} ({test_ratio*100:.0f}%)")
    
  
    
    
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

    
    train_dataset = train_list_ds.map(get_path_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_list_ds.map(get_path_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_list_ds.map(get_path_and_label, num_parallel_calls=tf.data.AUTOTUNE)


    def wrapper_fn(path, label):
        return tf.py_function(
            func=_decode_and_preprocess_image,
            inp=[path, label], # <-- FIXED: Removed IMAGE_SIZE from input tensor list
            Tout=[tf.float32, tf.float32] 
        )
        
   
    # Processing for Training Dataset (includes shuffle)
    final_train_ds = train_dataset.map(wrapper_fn, num_parallel_calls=tf.data.AUTOTUNE)
    final_train_ds = final_train_ds.map(
        lambda image, label: (
            tf.ensure_shape(image, IMAGE_SIZE + (CHANNELS,)),
            tf.ensure_shape(label, tf.TensorShape([config['num_classes']]))
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    final_train_ds = final_train_ds.cache().shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Processing for Validation Dataset (no shuffle)
    final_val_ds = val_dataset.map(wrapper_fn, num_parallel_calls=tf.data.AUTOTUNE)
    final_val_ds = final_val_ds.map(
        lambda image, label: (
            tf.ensure_shape(image, IMAGE_SIZE + (CHANNELS,)),
            tf.ensure_shape(label, tf.TensorShape([config['num_classes']]))
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    final_val_ds = final_val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Processing for Test Dataset (no shuffle)
    final_test_ds = test_dataset.map(wrapper_fn, num_parallel_calls=tf.data.AUTOTUNE)
    final_test_ds = final_test_ds.map(
        lambda image, label: (
            tf.ensure_shape(image, IMAGE_SIZE + (CHANNELS,)),
            tf.ensure_shape(label, tf.TensorShape([config['num_classes']]))
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    final_test_ds = final_test_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return final_train_ds, final_val_ds, final_test_ds 


if __name__ == '__main__':
    
    TEST_DATASET_KEYS = ["UCMERCED", "MED_WASTE", "AID"] 
    
    for key in TEST_DATASET_KEYS:
        print("="*60)
        print(f"--- STARTING DATA LOADER TEST FOR: {key} ---")
        
        try:
            
            train_pipeline, val_pipeline, test_pipeline = create_dataset_pipeline(key)
            
            
            for images, labels in train_pipeline.take(1):
                print(f"\n✅ SUCCESS: Train Pipeline created for {key}.")
                print(f"   Batch Image Shape: {images.shape}")
                print(f"   Batch Label Shape: {labels.shape}")
                break 
                
            for images, labels in val_pipeline.take(1):
                print(f"\n✅ SUCCESS: Validation Pipeline created for {key}.")
                print(f"   Batch Image Shape: {images.shape}")
                break 
                
            for images, labels in test_pipeline.take(1):
                print(f"\n✅ SUCCESS: Test Pipeline created for {key}.")
                print(f"   Batch Image Shape: {images.shape}")
                break 
                
        except Exception as e:
            print(f"\n❌ FATAL ERROR: Data loading failed for {key}.")
            print(f"   Error: {e}")
            break