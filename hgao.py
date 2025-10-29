import numpy as np
import tensorflow as tf
import gc 
import os

from config import HGAO_SEARCH_SPACE, DATASET_CONFIG, IMAGE_SIZE, CHANNELS
from model import build_densenet_model
from data_loader import create_dataset_pipeline



def evaluate_fitness(hyperparameters, dataset_key):
    try:
        
        train_ds, val_ds, _ = create_dataset_pipeline(dataset_key, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    except Exception as e:
        
        print(f"Error loading data: {e}. Returning low fitness.")
        return 0.0

    
    model = build_densenet_model(dataset_key, dropout_rate=hyperparameters['dropout_rate'])
    num_classes = DATASET_CONFIG[dataset_key]['num_classes']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

  
    history = model.fit(
        train_ds,
        epochs=5,
        verbose=0,
        validation_data=val_ds 
    )
    
   
    
    fitness_value = history.history['val_accuracy'][-1]
    
    del model 
    
    return fitness_value


def hgao_optimization_search(dataset_key):
    P = HGAO_SEARCH_SPACE['P']
    T = HGAO_SEARCH_SPACE['T'] 
    search_bounds = {k: v for k, v in HGAO_SEARCH_SPACE.items() if isinstance(v, list)}

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

    
    for t in range(T):
        
        
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            tf.compat.v1.reset_default_graph() 
        except Exception:
            pass 
        

        print(f"\n==============================================")
        print(f"🧬 HGAO Iteration {t+1}/{T} (Testing {P} Agents)")
        print(f"==============================================")
        
       
        for i, agent in enumerate(population):
            
            
            print(f"| Trial {i+1}/{P}: LR={agent['learning_rate']:.6f}, Drop={agent['dropout_rate']:.3f}...")
            
            agent['fitness'] = evaluate_fitness(agent, dataset_key)
            
           
            print(f"|    -> Fitness: {agent['fitness']:.4f}")
            
            if agent['fitness'] > best_fitness:
                best_fitness = agent['fitness']
                best_hyperparameters = agent.copy()
        
        
        print("\n--- ITERATION SUMMARY ---")
        print(f"| Current Best Fitness (Validation Accuracy): {best_fitness:.4f}")
        print(f"| Optimal LR: {best_hyperparameters['learning_rate']:.6f}, Dropout: {best_hyperparameters['dropout_rate']:.3f}")
        print("-------------------------\n")
        
        
        
        for i in range(len(population)):
            
            beta1 = np.random.uniform(*search_bounds['beta1_weight']) 
            beta2 = np.random.uniform(*search_bounds['beta2_weight']) 

            if beta1 > beta2:
                #exploitation
                new_lr = best_hyperparameters['learning_rate'] * (1 - t/T)
            else:
                 #exploration
                new_lr = np.random.uniform(*search_bounds['learning_rate']) 
            
            population[i]['learning_rate'] = np.clip(new_lr, *search_bounds['learning_rate'])
            population[i]['dropout_rate'] = np.clip(
                np.random.normal(best_hyperparameters['dropout_rate'], 0.1),
                *search_bounds['dropout_rate']
            )
            
    return best_hyperparameters