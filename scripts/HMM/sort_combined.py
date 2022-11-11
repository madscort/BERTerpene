#!/usr/bin/env python3
import os
import sys
import pickle

def find_seq(acc, file_path):
    """
    Function to open a file. Then find a sequence for a specified accession number.
    Func. only works with fasta-format
    """
    fasta_file = open(file_path, "r")
    flag = False
    #print(acc)
    for line in fasta_file:
        if flag:
            return line

        if str(line[:-1]) == str(acc):
            flag = True
    fasta_file.close()

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

def create_file_for_value(dict, path_direction):
    """
    Function which loops through a inputted dict.
    For every value in a key-value-pair in the dict
    (values is class of terpenoid synthase) is a file created.

    Where each file contains all the fasta entries for that class.
    """
    for item in dict:
        # this needed for testing purposes
        seq = find_seq(item, data_path + '/terp_test_2609.faa')
        if seq == None:
            continue

        try:
            with open(f"{path_direction}/{dict[item]}.txt".replace(' ', ''), "a") as file:
                print(f"{item}\t{dict[item]}\n{find_seq(item, data_path + '/terp.faa')}", file=file, end='')
        except:
            print("Something went wrong with :", dict[item])

if len(sys.argv) != 2:
    raise ValueError('Please provide the correct number of arguments: <program> <class_vs_acc.txt>')

# Path specification and opening files
data_path = os.getcwd()



annotation_file = open(sys.argv[1], "r")

# checks if files/directory exits, if not creates it:
check_existence_folder(data_path + "/class", True)
check_existence_folder(data_path + "/class/small", True)
check_existence_folder(data_path + "/class/large", True)

# Creates variables/dicts
small_class = dict()
large_class = dict()

# Loops through the annotation_file.
# To create a dict where: (Key = accession num, Value = class)
for line in annotation_file:
    temp = line[:-1].split('\t')
    small_class[temp[1]] = temp[0]
    correct = temp[1].split("_")[1]
    large_class[temp[1]] = correct

annotation_file.close()

# Creates files for each specific class
create_file_for_value(small_class, f"{data_path}/class/small/")
create_file_for_value(large_class, f"{data_path}/class/large/")



# saves the dict to check accuracy in later script
# with open("dict.pkl","wb") as dic:
#     pickle.dump(acc_class, dic)
