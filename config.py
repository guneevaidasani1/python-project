PROJECT_NAME = "HGAO-DenseNet Multi-Domain Classifier"
SEED = 42             
IMAGE_SIZE = (224, 224) 
CHANNELS = 3          
BATCH_SIZE = 8  


KAGGLE_DATASET_IDS = [
   
    "abdulhasibuddin/uc-merced-land-use-dataset"
    
]


DATASET_CONFIG = {
    "UCMERCED": {
        "local_name": "UCMerced_LandUse", 
        "num_classes": 21,       
        "path_suffix": "Images", 
        "input_shape": IMAGE_SIZE + (CHANNELS,),
        "zip_file_name": "UCMerced_LandUse.zip",
    },
    

    "MED_WASTE": {
        "local_name": "Medical Waste 4.0", 
        "num_classes": 13, 
        "path_suffix": "Medical Waste 4.0", 
        "input_shape": IMAGE_SIZE + (CHANNELS,),
        "zip_file_name": "Medical_Waste_4_0.zip",
    },
    

    "FETUS_US": {
        "local_name": "Ultrasound Fetus Dataset", 
        "num_classes": 3,
        "path_suffix": "Ultrasound Fetus Dataset/Data/Data", 
        "input_shape": IMAGE_SIZE + (CHANNELS,),
        "zip_file_name": "Fetus_US.zip",
    },
    
    "TARGET_DATASET": "FETUS_US", 
}

DATA_DIR = "./data/"


TARGET_DATASET = DATASET_CONFIG["TARGET_DATASET"]

HGAO_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-1], 
    "dropout_rate": [0.1, 0.6],     
    "P": 3,              
    "T": 5,              
    "beta1_weight": [0.1, 0.9], 
    "beta2_weight": [0.1, 0.9], 
}