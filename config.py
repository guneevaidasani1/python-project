
PROJECT_NAME = "HGAO-DenseNet Multi-Domain Classifier"
SEED = 42             
IMAGE_SIZE = (224, 224) 
CHANNELS = 3          
BATCH_SIZE = 32       


KAGGLE_DATASET_IDS = [
   
    "abdulhasibuddin/uc-merced-land-use-dataset",
    "jiayuanchengala/aid-scene-classification-datasets"
]


DATASET_CONFIG = {
    "UCMERCED": {
        "local_name": "UCMerced_LandUse", 
        "num_classes": 21,       
        "path_suffix": "Images", 
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    

    "MED_WASTE": {
        "local_name": "Medical Waste 4.0", 
        "num_classes": 13, 
        "path_suffix": "Medical Waste 4.0", 
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    

    "AID": {
        "local_name": "AID", 
        "num_classes": 30,
        "path_suffix": None, 
        "input_shape": IMAGE_SIZE + (CHANNELS,),
    },
    
    "TARGET_DATASET": "MED_WASTE", 
}

DATA_DIR = "./data/"


TARGET_DATASET = DATASET_CONFIG["TARGET_DATASET"]

HGAO_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-1], 
    "dropout_rate": [0.1, 0.6],     
    "P": 5,              
    "T": 7,              
    "beta1_weight": [0.1, 0.9], 
    "beta2_weight": [0.1, 0.9], 
}