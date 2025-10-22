import tensorflow as tf
import time
import os




tf.config.threading.set_inter_op_parallelism_threads(4) 
tf.config.threading.set_intra_op_parallelism_threads(8)
from config import TARGET_DATASET ,HGAO_SEARCH_SPACE,DATASET_CONFIG
from hgao import hgao_optimization_search
from model import build_densenet_model
from data_loader import create_dataset_pipeline


FINAL_TRAINING_EPOCHS = 30


def main():
    """
    Runs the HGAO optimization, then trains the final model.
    """
    
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
    
   #loading data
    final_ds = create_dataset_pipeline(TARGET_DATASET)
    
    #Using hgao we build the denze net model
    final_model = build_densenet_model(TARGET_DATASET, dropout_rate=optimal_dropout)
    
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    #starting to train model
    print(f"Starting full training for {FINAL_TRAINING_EPOCHS} epochs...")
    final_model.fit(
        final_ds,
        epochs=FINAL_TRAINING_EPOCHS,
        verbose=1,
    )

    #testing model
    loss, acc = final_model.evaluate(final_ds, verbose=0)
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS for {TARGET_DATASET}:")
    print(f"Final Accuracy: {acc:.4f}")
    print(f"Total Runtime: {(time.time() - start_time) / 3600:.2f} hours")
    print("="*60)


if __name__ == "__main__":
    main()