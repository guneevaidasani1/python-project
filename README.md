# 🐍 python-project: HGAO-Optimized DenseNet for Image Classification

This project implements an advanced image classification solution where the **DenseNet** deep learning model is fine-tuned using a bio-inspired metaheuristic search algorithm: the **Horned Lizard and Giant Armadillo Optimization (HGAO)**.

The core objective is to automatically find the optimal set of hyperparameters for the DenseNet model, significantly improving classification accuracy across various image datasets without requiring manual tuning or exhaustive grid search. 

## ✨ Project Components Overview

The project is modular, with each file handling a specific part of the machine learning workflow:

| File | Role | Description |
| :--- | :--- | :--- |
| **`main.py`** | **Orchestrator** | Executes the entire hyperparameter search and model evaluation workflow. This is the **script you run**. |
| **`hgao.py`** | **Optimizer Engine** | Implements the **HGAO algorithm** (Horned Lizard and Giant Armadillo Optimization) to propose optimal hyperparameters. |
| **`model.py`** | **Model Definition** | Defines the DenseNet deep learning architecture based on the parameters provided by HGAO. |
| **`data_loader.py`** | **Data Handler** | Loads, preprocesses, and prepares the image datasets (including augmentation) for training. |
| **`config.py`** | **Configuration** | Stores fixed parameters (e.g., image size) and the search space bounds for the HGAO optimizer. |
| **`requirements.txt`** | **Dependencies** | Lists all required Python libraries. |
| **`setup.py`** | **Package Setup** | Standard Python file for building and distributing the project. |
| **`data/`** | **Data Directory** | Folder where the datasets will be extracted for use. |

---

## 📚 Supporting Materials

The repository includes supplementary files to explain and demonstrate the project:

* **`Python Project Presentation.pptx`**: A presentation detailing the methodology, results, and conclusions of the project.
* **`Python Project Video.mp4`**: A video demonstration or a detailed walkthrough of the project and its execution.

---

## 🚀 How to Run the Code

Follow these steps to set up the environment, download the data, and execute the main optimization script.

### 1. 💾 Download Datasets

Download the dataset ZIP files to your local device. The project is designed to work with the following datasets:

* **Medical Waste 4.0 Dataset**:
    * **[Download Link (Kaggle)](https://www.kaggle.com/datasets/mmasodulrahmanusmani/medical-waste-4-0)**
* **Ultrasound Fetus Dataset**:
    * **[Download Link (Kaggle)](https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset)**
* **UC Merced Land Use Dataset**:
    * **[Download Link (Kaggle)](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset)**

---

### 2. 📂 Rename Datasets and Place in Root Directory

Ensure the downloaded ZIP files are placed directly in the **root directory** of your project repository and are named precisely as follows:

* `Medical_Waste_4_0.zip`
* `Fetus_US.zip`
* `UCMerced_LandUse.zip`

Your root directory should contain all Python files and the three ZIP files.

---

### 3. 🛠️ Install Project Requirements

Navigate to the project's root directory in your terminal and install all necessary Python libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```
---

### 4. ▶️ Execute Main Script

The script will automatically unzip and extract the data files into the **`data/`** folder upon first execution.  
It will then prompt you to choose which dataset you wish to run the optimization on.  
Execute the main script to initiate the hyperparameter search for the DenseNet model:

```bash
python main.py
