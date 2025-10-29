import tensorflow as tf
import time
import os
import sys


from setup import unzip_datasets 


tf.config.threading.set_inter_op_parallelism_threads(4) 
tf.config.threading.set_intra_op_parallelism_threads(8)


from config import HGAO_SEARCH_SPACE, DATASET_CONFIG
from hgao import hgao_optimization_search
from model import build_densenet_model
from data_loader import create_dataset_pipeline


FINAL_TRAINING_EPOCHS = 40


def get_user_dataset_choice(dataset_config):
    """
    Prompts the user to select a target dataset from the available options,
    displaying the local_name but returning the key.
    """
    
    dataset_items = [(k, v) for k, v in dataset_config.items() if isinstance(v, dict)]
    dataset_keys = [k for k, v in dataset_items]
    
    if not dataset_keys:
        print("CRITICAL ERROR: No datasets found in DATASET_CONFIG.")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("Hope you have the datasets on your local device")
    
    
    prompt_parts = []
    for i, (key, config) in enumerate(dataset_items):
        local_name = config.get('local_name', key)
        prompt_parts.append(f"{i+1} for {local_name}")
    
   
    prompt_message = "Enter " + ", ".join(prompt_parts)
    prompt_message += f" (Total {len(dataset_keys)} options)"

    print(prompt_message)
    print("="*60)
        
    while True:
        try:
            choice = input("Your choice: ")
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(dataset_keys):
                
                chosen_key = dataset_keys[choice_index]
                print(f"\nSelected Dataset: {chosen_key} ({dataset_config[chosen_key]['local_name']})\n")
                return chosen_key
            else:
                print("Invalid choice. Please enter a number corresponding to the options.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    
    
    unzip_datasets()
    
    
    TARGET_DATASET = get_user_dataset_choice(DATASET_CONFIG)
    
    start_time = time.time()
    
    print("="*60)
    print(f"       STARTING HGAO-DENSENET CLASSIFICATION FOR {TARGET_DATASET}        ")
    print("="*60)

    #running hgao to optimize model
    print("PHASE 1: Running HGAO Hyperparameter Search (Short Runs)...")
    
    optimal_params = hgao_optimization_search(TARGET_DATASET)
    
    optimal_lr = optimal_params['learning_rate']
    optimal_dropout = optimal_params['dropout_rate']
    
    print("\n" + "="*60)
    print(f"✅ HGAO Search Complete. Optimal Params Found:")
    print(f"  > Learning Rate: {optimal_lr:.6f}")
    print(f"  > Dropout Rate: {optimal_dropout:.3f}")
    print("="*60)


    #final model train
    print("\nPHASE 2: Final Model Training with Optimized Parameters (Full Epochs)...")
    
   
    final_train_ds, final_val_ds, final_test_ds = create_dataset_pipeline(TARGET_DATASET)
    
    #Using hgao we build the denze net model
    final_model = build_densenet_model(TARGET_DATASET, dropout_rate=optimal_dropout)
    
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # starting to train model
    print(f"Starting full training for {FINAL_TRAINING_EPOCHS} epochs...")
    final_model.fit(
        final_train_ds, 
        epochs=FINAL_TRAINING_EPOCHS,
        validation_data=final_val_ds, 
        verbose=1,
    )

    # testing model on the unseen TEST set
    loss, acc = final_model.evaluate(final_test_ds, verbose=0) 
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS for {TARGET_DATASET}:")
    print(f"Final Test Accuracy: {acc:.4f}")
    print(f"Total Runtime: {(time.time() - start_time) / 3600:.2f} hours")
    print("="*60)


if __name__ == "__main__":
    main()