import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_densenet_model(dataset_key, dropout_rate):
    """
    Builds the DenseNet-121 base model with a custom classification head.

    Args:
        dataset_key (str): Key for the target dataset (e.g., 'LC25000').
        dropout_rate (float): The rate set by the HGAO algorithm.

    Returns:
        tf.keras.Model: The uncompiled DenseNet model.
    """
    config = DATASET_CONFIG.get(dataset_key)
    if not config:
        raise ValueError(f"Dataset key '{dataset_key}' not found in config.")

    
    num_classes = config['num_classes'] ##number of possible classifications
    input_shape = config['input_shape'] #dimensions of the image

    print(f"-> Building DenseNet-121 for {dataset_key} with {num_classes} classes...")

    # 1. Load the DenseNet-121 base model for Transfer Learning
    # We use 'imagenet' weights and exclude the original top layers (include_top=False)
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,  
        input_shape=input_shape
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x) 

    # This is the Dropout layer whose rate HGAO will tune
    x = Dropout(dropout_rate)(x)

    # Final output layer (softmax for multi-class classification)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine the base model and the new layers
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base layers for initial rapid tuning (Transfer Learning)
    for layer in base_model.layers:
        layer.trainable = False
        
    return model

