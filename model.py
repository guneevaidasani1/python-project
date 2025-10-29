import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from config import DATASET_CONFIG , IMAGE_SIZE          


def build_densenet_model(dataset_key, dropout_rate):
    """
    Builds the DenseNet-121 base model with a custom classification head.

    Args:
        dataset_key (str): Key for the target dataset (e.g., 'LC25000').
        dropout_rate (float): The rate set by the HGAO algorithm.

    Returns:
        tf.keras.Model: The uncompiled DenseNet model.
    """
    temp_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    del temp_model
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise ValueError(f"Dataset key '{dataset_key}' not found in config.")

    
    num_classes = config['num_classes'] 
    input_shape = config['input_shape'] 

    print(f"-> Building DenseNet-121 for {dataset_key} with {num_classes} classes...")

    
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,  
        input_shape=input_shape
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x) 

    
    x = Dropout(dropout_rate)(x)

    
    predictions = Dense(num_classes, activation='softmax')(x)

    
    model = Model(inputs=base_model.input, outputs=predictions)

    
    for layer in base_model.layers:
        layer.trainable = False
        
    return model

