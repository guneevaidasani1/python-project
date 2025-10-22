
PROJECT_NAME = "HGAO-DenseNet Multi-Domain Classifier"
SEED = 42            # For reproducibility
IMAGE_SIZE = (224, 224) 
CHANNELS = 3         
BATCH_SIZE = 32      




KAGGLE_DATASET_IDS = [
    "javaidahmadwani/lc25000",
    "yudhaislamisulistya/plants-type-datasets"
]

DATASET_CONFIG = {
    
    "LC25000": {
        "local_name": "lc25000",  
        "num_classes": 5,        
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    "SMOKE_CLASSIFICATION": { 
        "local_name": "IIITDMJ_Smoke", 
        "num_classes": 4,          
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    
    "PLANTS_TYPE": {
        "local_name": "plants-type-datasets", 
        "num_classes": 30,       
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    "TARGET_DATASET": "LC25000", 
}
DATA_DIR = "./data/"

TARGET_DATASET = DATASET_CONFIG["TARGET_DATASET"]


### HGAO HYPERPARAMETER SEARCH SPACE ###
HGAO_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-1], 
    "dropout_rate": [0.1, 0.6],    
    "P": 30,             
    "T": 10,             
    "beta1_weight": [0.1, 0.9], 
    "beta2_weight": [0.1, 0.9], 
}