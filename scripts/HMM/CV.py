#!/usr/bin/env python3
import os
import re
import sys
import time
from sort_combined import find_seq
from joblib import Parallel, delayed


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


if len(sys.argv) != 2:
    raise ValueError('Please provide the correct number of arguments: <program> <class_vs_acc.txt>')

# Path specification and opening files
data_path = os.getcwd()

# checks if correct folders exits
check_existence_folder(f"{data_path}/training", True)


# varibels used
correct_prediction_small = 0
wrong_prediction_small = 0
correct_prediction_large = 0
wrong_prediction_large = 0


# opens annotation file where the class with only a single entry is removed.
annotation_file = open(sys.argv[1], "r")

# loops through the annotation file. For each iteration a test entry/file is selected/created
# and the rest is used to make alignments and pHMM, lastly the test entry is used as for testing.
for i, line in enumerate(annotation_file):
    # creates the file for testing
    with open (f"{data_path}/testing.txt", "w") as file:
        sequence = find_seq(line.split('\t')[1][:-1], data_path + '/terp.faa')
        print(f">{line.replace('>', '')}{sequence}", file=file, end='')
        acc_header_test = line.replace('>', '').split("\t")
        small_label = acc_header_test[0]
        large_label = acc_header_test[1].split("_")[1][:-1]

    # removes the testing entry for the training set.

    with open (sys.argv[1], "r") as file:
        # reads the full annotation file as a string
        annotation_string = file.read()
    # replaces the line (=testing) with nothing
    training = annotation_string.replace(line, '')

    # Creates the training set in a file
    with open (f"{data_path}/training/training.txt", "w") as file:
        print(f">{training}", file=file, end='')

    # The training set is sorted into the different class
    os.system(f"./sort_combined.py {data_path}/training/training.txt")

    # Uses the function 'sort_combined' from 'alignment_combined'
    # To both make alignments and pHMM
    os.system(f"./aligment_combined.py")

    time.sleep(1)

    os.system(f"./pHMM_DB_combined.py")
    time.sleep(1)


    # hmmscan --tblout output-1 pHMM_model/large/large_pHMM testing.txt
    os.system(f"hmmscan --tblout output_large pHMM_model/large/large_pHMM testing.txt")
    os.system(f"hmmscan --tblout output_small pHMM_model/small/small_pHMM testing.txt")
    time.sleep(2)




    with open("output_large", "r") as data:
        for line in data:
            if line[0] != '#':
                result = re.split(r'\s+', line)
                result_HMM_large = result[0].replace(">", "")
                E_val_large = result[4]
                score_large = result[5]
                break
    with open("output_small", "r") as data:
        for line in data:
            if line[0] != '#':
                result = re.split(r'\s+', line)
                result_HMM_small  = result[0].replace(">", "")
                E_val_small = result[4]
                score_small = result[5]
                break



    # logging predictions
    if result_HMM_large == large_label.replace(">", ""):
        correct_prediction_large += 1
        with open("log_151022_BLOSUM62.txt", "a") as file:
            print(f"Correct\tpos {i} \tlarge\tCorr: {large_label}\tpred: {result_HMM_large}\tE-val: {E_val_large}\tScore: {score_large}" , file=file)
    else:
        wrong_prediction_large += 1
        with open("log_151022_BLOSUM62.txt", "a") as file:
            print(f"wrong\tpos {i} \tlarge\tCorr: {large_label}\tpred: {result_HMM_large}\tE-val: {E_val_large}\tScore: {score_large}" , file=file)

    if result_HMM_small == small_label.replace(">", ""):
        correct_prediction_small += 1
        with open("log_151022_BLOSUM62.txt", "a") as file:
            print(f"Correct\tpos {i} \tsmall\tCorr: {small_label}\tpred: {result_HMM_small}\tE-val: {E_val_small}\tScore: {score_small}", file=file)
    else:
        wrong_prediction_small += 1
        with open("log_151022_BLOSUM62.txt", "a") as file:
            print(f"wrong\tpos {i} \tsmall\tCorr: {small_label}\tpred: {result_HMM_small}\tE-val: {E_val_small}\tScore: {score_small}" , file=file)


    # print("\nNEW LINE\n")
    # if i == 3:
    #
    #     break

annotation_file.close()



with open("CV_result_151022_BLOSUM62.txt", "w") as file:
    print(f"Correct predicted large class: {correct_prediction_large}\nWrong predicted large class: {wrong_prediction_large}\nCorrect predicted small class: {correct_prediction_small}\nWrong predicted small class: {wrong_prediction_small}", file=file)
