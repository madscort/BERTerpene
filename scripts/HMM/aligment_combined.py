#!/usr/bin/env python3
import os
import re


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



def aligment_hmmbuild(file_path, name):
    # Using the Clustal software for MSA, documentation: http://www.clustal.org/omega/#Documentation
    # Hmmer documentation: http://hmmer.org/documentation.html
    files = os.listdir(f"{file_path}")

    for file in files:

        # opens each file, using regex to find the number of '\n'
        with open(f"{file_path}/{file}", 'r') as open_file:
            nums_of_new_lines = re.findall('\n', open_file.read(), re.DOTALL)

        # if a file contains > 2 sequences then use clustal-software to make a MSA
        if len(nums_of_new_lines) != 2:
            os.system(
                f"clustalo -i {file_path}/'{file}' > {data_path}/aligments/{name}/'{file[:-4]}.st' --outfmt=st ")

            # After the MSA is done. The profile HMM is computed
            os.system(
                f"hmmbuild --mx BLOSUM62 {data_path}/pHMM/{name}/'{file[:-4]}'.hmm {data_path}/aligments/{name}/'{file[:-4]}.st' ")

        # A solution to classes with a single seq. Is to make profile HMM using a sort of substitution score matrix
        # this solution is used here.
        # hmmer supports individual scoring matrices. This implementations uses substitution matrix.
        # 2-fold CV could be interesting to test the best matrix for the data
        # PAM 30 is so far the best one
        # An assumption used here is that a single sequence file contains two: '\n'
        elif len(nums_of_new_lines) == 2:
            os.system(
                f"hmmbuild --mx BLOSUM62 {data_path}/pHMM/{name}/'{file[:-4]}'.hmm {file_path}/'{file}' ")


data_path = os.getcwd()

check_existence_folder(f"{data_path}/aligments/small", True)
check_existence_folder(f"{data_path}/aligments/large", True)
check_existence_folder(f"{data_path}/pHMM/small", True)
check_existence_folder(f"{data_path}/pHMM/large", True)



aligment_hmmbuild(f"{data_path}/class/small", "small")
aligment_hmmbuild(f"{data_path}/class/large", "large")
