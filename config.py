
PROJECT_NAME = "HGAO-DenseNet Multi-Domain Classifier"
SEED = 42            # For reproducibility
IMAGE_SIZE = (224, 224) 
CHANNELS = 3         
BATCH_SIZE = 32      




KAGGLE_DATASET_IDS = [
    "jiayuanchengala/aid-scene-classification-datasets"
]


DATASET_CONFIG = {
    "BREAKHIS": {
        "local_name": "dataset_cancer_v1", 
        "num_classes": 4,       
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    
    
    "MED_WASTE": {
        "local_name": "Medical Waste 4.0", 
        "num_classes": 12,       
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    
    # 3. REMOTE SENSING (AID Scene)
    "AID": {
        "local_name": "AID", 
        "num_classes": 30,      
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    
    "TARGET_DATASET": "BREAKHIS", 
}

DATA_DIR = "./data/"

TARGET_DATASET = DATASET_CONFIG["TARGET_DATASET"]


HGAO_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-1], 
    "dropout_rate": [0.1, 0.6],    
    "P": 30,             
    "T": 10,             
    "beta1_weight": [0.1, 0.9], 
    "beta2_weight": [0.1, 0.9], 
}