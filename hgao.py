import numpy as np
import tensorflow as tf
import gc 
import os

from config import HGAO_SEARCH_SPACE, DATASET_CONFIG, IMAGE_SIZE, CHANNELS
from model import build_densenet_model
from data_loader import create_dataset_pipeline


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'


def evaluate_fitness(hyperparameters, dataset_key):
    """
    Evaluates the performance (fitness) of a candidate hyperparameter set 
    by running a short, accelerated training trial.
    """
    lr = hyperparameters['learning_rate']
    dropout = hyperparameters['dropout_rate']
    

    try:
        train_ds = create_dataset_pipeline(dataset_key)
    except Exception as e:
        print(f"Error loading data for fitness evaluation: {e}")
        return 0.0 


    model = build_densenet_model(dataset_key, dropout_rate=dropout)
    
    # Get num_classes from config for correct compilation
    num_classes = DATASET_CONFIG[dataset_key]['num_classes']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 2. Train Model for a fixed, short number of epochs (Fitness Run)
    FITNESS_EPOCHS = 5
    
    # NOTE: The data loader already handles batching and shuffling
    history = model.fit(
        train_ds,
        epochs=FITNESS_EPOCHS,
        verbose=0 # Suppress verbose output to keep the search clean
    )
    
    # 3. Define Fitness: Use the final training accuracy
    fitness_value = history.history['accuracy'][-1]
    
    # Explicitly delete the model to free resources before the global cleanup
    del model 
    
    return fitness_value

# --- 2. HGAO ALGORITHM CORE LOGIC (Metaheuristic Search) ---
def hgao_optimization_search(dataset_key):
    """
    Implements the Hybrid Giant Armadillo and Horned Lizard Optimization (HGAO)
    to find the optimal hyperparameters for DenseNet.
    """
    P = HGAO_SEARCH_SPACE['P'] # Population size
    T = HGAO_SEARCH_SPACE['T'] # Max iterations
    
    search_bounds = {k: v for k, v in HGAO_SEARCH_SPACE.items() if isinstance(v, list)}

    # 1. Initialization: Create an initial population (P search agents)
    population = []
    for _ in range(P):
        agent = {
            'learning_rate': np.random.uniform(*search_bounds['learning_rate']),
            'dropout_rate': np.random.uniform(*search_bounds['dropout_rate']),
            'fitness': 0.0
        }
        population.append(agent)

    best_hyperparameters = None
    best_fitness = -1.0

    # 2. Iteration Loop (HGAO Phases)
    for t in range(T):
        print(f"\n--- HGAO Iteration {t+1}/{T} ---")
        
        # --- CRITICAL FIX: AGGRESSIVE CLEANUP BEFORE MODEL BUILDING ---
        # This prevents the silent hang/crash that was observed
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            # Explicitly reset the TF graph to handle lingering resources
            tf.compat.v1.reset_default_graph() 
        except Exception:
            pass 
        # --- END CRITICAL FIX ---
        
        # Phase A: Evaluation and Leader Selection
        for agent in population:
            # The model is built and trained inside evaluate_fitness
            agent['fitness'] = evaluate_fitness(agent, dataset_key)
            if agent['fitness'] > best_fitness:
                best_fitness = agent['fitness']
                best_hyperparameters = agent.copy()
        
        print(f"Current Best Fitness: {best_fitness:.4f}")
        print(f"Current Best LR: {best_hyperparameters['learning_rate']:.6f}, Dropout: {best_hyperparameters['dropout_rate']:.3f}")

        # Phase B: Adaptive Search (Updating agent positions based on HGAO logic)
        for i in range(len(population)):
            # Adaptive Weighting: Balance exploitation (beta1) and exploration (beta2)
            beta1 = np.random.uniform(*HGAO_SEARCH_SPACE['beta1_weight']) 
            beta2 = np.random.uniform(*HGAO_SEARCH_SPACE['beta2_weight']) 

            # Simplified HGAO Movement Logic
            if beta1 > beta2:
                 # Exploitation (Giant Armadillo): Move toward the current best solution
                new_lr = best_hyperparameters['learning_rate'] * (1 - t/T) # Simplified update
            else:
                 # Exploration (Horned Lizard): Search a new area
                new_lr = np.random.uniform(*search_bounds['learning_rate']) 
            
            # Update and Clip Hyperparameters
            population[i]['learning_rate'] = np.clip(new_lr, *search_bounds['learning_rate'])
            population[i]['dropout_rate'] = np.clip(
                np.random.normal(best_hyperparameters['dropout_rate'], 0.1),
                *search_bounds['dropout_rate']
            )
            
    return best_hyperparameters