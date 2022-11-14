#!/usr/bin/env python3
import time
import os
import re


# Reading the thresholds into a dict
with open('thresholds_large', 'r') as threshold_file:
    threshold_large_dict = dict()
    for line in threshold_file:
        split = line.split('\t')
        threshold_large_dict[split[0]] = split[1][:-1]


flag = False
with open ('allterp.faa', 'r') as file:
    for i, line in enumerate(file):
        split = line.split('_')

        if flag:
            # Using HMM to predict class
            with open ('testing.txt', 'w') as outfile:
                print(header, line, end='', file=outfile, sep='')
            acc = header
            seq = line

            os.system(f"hmmscan --tblout output_large -o /dev/null pHMM_model/large/large_pHMM testing.txt ")
            time.sleep(.25)
            flag = False

            with open ('output_large', 'r') as prediction:
                for item in prediction:
                    if item[0] != '#':
                        result = re.split(r'\s+', item)
                        result_HMM_large = result[0].replace(">", "")
                        E_val = result[4]
                        score = result[5]
                        break
            if result_HMM_large in threshold_large_dict:
                specific_threshold_val = threshold_large_dict[result_HMM_large]
            else:
                # If the there is no known threshold. The avg of all threshold is apply.
                specific_threshold_val = (353.8 + 654.3 + 523.2 + 305.6 + 281.0 + 205.9) / 6

            if float(specific_threshold_val) < float(score):
                # print('Good prediction', i, result_HMM_large, score, specific_threshold_val)
                with open ('classfied_tc_allterp_HMM.txt','a') as outfile:
                    print(f'{header[:-1]}_{result_HMM_large}\n', seq, end='', file=outfile, sep='')
            else:
                # print('Bad prediction', i, result_HMM_large, score, specific_threshold_val)



        try:
            if split[1][:-1] =='tc':
                flag = True
                header = line
        except:
            pass

