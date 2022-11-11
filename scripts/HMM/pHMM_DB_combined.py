#!/usr/bin/env python3
import os

def check_existence_folder(folder, delete):
    """
    Function which checks whether a given file/dictionary already exists else it creates it.
    """
    check_dict = os.path.isdir(folder)

    if not check_dict:
        os.makedirs(folder)
    else:
        if delete:
            os.system(f"rm -rf {folder}")
            os.makedirs(folder)
    return None

def model_press(name, complete_db_name):
    check_existence_folder(f"{data_path}/pHMM_model/{name}", True)

    # cats all the different pHMM models into a single model.
    os.system(f"cat {data_path}/pHMM/{name}/* > {data_path}/pHMM_model/{name}/{complete_db_name}")
    # hmmpress of the full model is needed for the model to work.
    os.system(f"hmmpress {data_path}/pHMM_model/{name}/{complete_db_name}")

# Defining paths and files
data_path = os.getcwd()

# checks if a file with the name 'pHMM_model', else creates it
check_existence_folder(f"{data_path}/pHMM_model", True)

model_press("small", "small_pHMM")
model_press("large", "large_pHMM")




