import os
import tensorflow as tf
from config import DATASET_CONFIG, IMAGE_SIZE, CHANNELS, DATA_DIR, BATCH_SIZE


def create_dataset_pipeline(dataset_key):
    """
    Gathers image data, fixes nested paths for current datasets, and creates an 
    optimized TF Dataset ready for the HGAO-DenseNet model.
    """
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise ValueError(f"Dataset key '{dataset_key}' not found in config.")
    
    
    base_path_from_config = os.path.join(DATA_DIR, config['local_name'])

    
    if dataset_key == 'BREAKHIS':
        final_path = os.path.join(base_path_from_config, 'classificacao_multiclasse')
        
    elif dataset_key == 'MED_WASTE':
        final_path = os.path.join(base_path_from_config, 'Medical Waste 4.0')
        
    elif dataset_key == 'AID':
        final_path = base_path_from_config
    
  
    
    
    if not os.path.isdir(final_path):
        raise FileNotFoundError(
            f"Could not find required directory: {final_path}. "
            f"Please verify the exact subfolder names inside your 'data/' directory."
        )

    print(f"\nLoading data for {dataset_key} from: {final_path}")
    
    # Keras Utility to load images and labels
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=final_path,
        labels='inferred',
        label_mode='categorical',
        class_names=None, 
        image_size=IMAGE_SIZE,
        batch_size=None, 
        shuffle=True,
        follow_links=True 
    )
    
   #normalizing pixels
    dataset = dataset.map(lambda image, label: (image / 255.0, label), num_parallel_calls=tf.data.AUTOTUNE)

   
    dataset = dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == '__main__':
    TEST_DATASET_KEYS = ["BREAKHIS", "MED_WASTE", "AID"] 
    
    for key in TEST_DATASET_KEYS:
        print("="*60)
        print(f"--- STARTING DATA LOADER TEST FOR: {key} ---")
        
        try:
            pipeline = create_dataset_pipeline(key)
            
            
            for images, labels in pipeline.take(1):
                print(f"\n✅ SUCCESS: Pipeline created for {key}.")
                print(f"   Batch Image Shape: {images.shape}")
                print(f"   Batch Label Shape: {labels.shape}")
                print(f"   Inferred Classes: {labels.shape[1]}")
                break 
                
        except FileNotFoundError as fnf_e:
            print(f"\n❌ FATAL ERROR: File/Path not found for {key}!")
            print(f"   Fix the path in config.py or the conditional logic.")
            print(f"   Error: {fnf_e}")
            break
        except Exception as e:
            print(f"\n❌ FATAL ERROR: Data loading failed for {key}.")
            print(f"   Check class counts or image format.")
            print(f"   Error: {e}")
            break 