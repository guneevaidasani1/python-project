import numpy as np
import tensorflow as tf
import os

from config import HGAO_SEARCH_SPACE, TARGET_DATASET
from model import build_densenet_model
from data_loader import create_dataset_pipeline

def evaluate_fitness(hyperparameters, dataset_key):
    """
    Evaluates the performance (fitness) of a candidate hyperparameter set.
    Fitness is defined as the training accuracy of the DenseNet model 
    after a short, accelerated training run.
    """
    lr = hyperparameters['learning_rate']
    dropout = hyperparameters['dropout_rate']
    
   
    try:
        
        train_ds = create_dataset_pipeline(dataset_key) 
    except Exception as e:
        print(f"Error loading data for fitness evaluation: {e}")
        return 0.0

    #build and compile
    model = build_densenet_model(dataset_key, dropout_rate=dropout)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #train model for small epoch
    FITNESS_EPOCHS = 5 
    
    history = model.fit(
        train_ds,
        epochs=FITNESS_EPOCHS,
        verbose=0 
    )
    #define what fitness 
    fitness_value = history.history['accuracy'][-1]
    
    tf.keras.backend.clear_session() #clear memory
    return fitness_value


def hgao_optimization_search(dataset_key):
    """
    Implements the Hybrid Giant Armadillo and Horned Lizard Optimization (HGAO)
    to find the optimal hyperparameters for DenseNet.
    """
    P = HGAO_SEARCH_SPACE['P'] #population size
    T = HGAO_SEARCH_SPACE['T'] #max no of iterations
    search_bounds = {k: v for k, v in HGAO_SEARCH_SPACE.items() if isinstance(v, list)}

    #create initial population
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
        print(f"\n--- HGAO Iteration {t+1}/{T} ---")
        
        #phase1
        for agent in population:
            agent['fitness'] = evaluate_fitness(agent, dataset_key)
            if agent['fitness'] > best_fitness:
                best_fitness = agent['fitness']
                best_hyperparameters = agent.copy()

        print(f"Current Best Fitness: {best_fitness:.4f}")
        
       #phase2
        for i in range(len(population)):
            
            beta1 = np.random.uniform(*HGAO_SEARCH_SPACE['beta1_weight']) 
            beta2 = np.random.uniform(*HGAO_SEARCH_SPACE['beta2_weight']) 

            
            if beta1 > beta2:
                 #giant armadillo --> exploitation
                new_lr = population[i]['learning_rate'] + beta1 * (best_hyperparameters['learning_rate'] - population[i]['learning_rate']) 
            else:
                #horned lizard --> explore new area
                new_lr = np.random.uniform(*search_bounds['learning_rate']) 
            
            
            population[i]['learning_rate'] = np.clip(new_lr, *search_bounds['learning_rate'])
            population[i]['dropout_rate'] = np.clip(
                np.random.normal(best_hyperparameters['dropout_rate'], 0.1), 
                *search_bounds['dropout_rate']
            )
            
    return best_hyperparameters