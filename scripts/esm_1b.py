#!/usr/bin/env python
# coding: utf-8

# Install the transformers package WITH the ESM model. 
# It is unfortunately not available in the official release yet.
#!git clone -b add_esm-proper --single-branch https://github.com/liujas000/transformers.git 
#get_ipython().system('pip -q install ./transformers')


# Load packages

import sys
import math
from sklearn.model_selection import train_test_split
from transformers import pipeline, ESMForTokenClassification, ESMTokenizer, ESMForMaskedLM, ESMForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot

# Use MPS or CUDA if available:
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Data preprocessing

# Get sequences, accession number and main category labels:

sequence = ""
sequences = list()
acc_num = list()
main_cat = list()

first = True
with open("../data/terp.faa") as file:
    
    first_acc = file.readline()
    acc_num.append(first_acc.split(">")[1].strip())
    main_cat.append(first_acc.split("_")[1].strip())

    for line in file:
        if line.startswith(">"):
            sequences.append(sequence)
            sequence = ""
            acc_num.append(line.split(">")[1].strip())
            main_cat.append(line.split("_")[1].strip())
        else:
            sequence += line.strip()
    
    # Add last sequence
    sequences.append(sequence)

# Create numbered labels for main categories:

main2label = {c: l for l, c in enumerate(sorted(set(main_cat)))}
label2main = {l: c for c, l in main2label.items()}

# Create class translation dictionary for accession numbers:

acc2class = dict()

with open("../data/class_vs_acc_v2.txt", "r") as file:
    for line in file:
        t_class = line.split("\t")[0]
        acc = line.split("\t")[1].strip()[1:]
        acc2class[acc] = t_class

# Create numbered labels for classes:
        
class2label = {c: l for l, c in enumerate(sorted(set(acc2class.values())))}
label2class = {l: c for c, l in class2label.items()}

print(
    f"The files contain:",
    f"{len(sequences)} sequences in",
    f"{len(set(main_cat))} main categories and",
    f"{len(set(acc2class.values()))} classes")


# Possibly check class distribution here...

# Choose between category and class:
labels = main_cat
#labels = acc_num # This will translate to class later.

# Split into training and validation set. Is this necessary?

train_seq, val_seq, train_labels, val_labels = train_test_split(sequences, labels, test_size=.1)

print(f"Training size: {len(train_seq)} Validation size: {len(val_seq)}")


# Tokenizer:
tokenizer = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False)


class SequenceDataset(Dataset):
    def __init__(self, input_sequences, input_labels, categories=True):
        
        # Init is run once, when instantiating the dataset class.
        #
        # Either supply with:
        #  Main categories - category classification
        #  Accession numbers - class classifcation 
        
        # The xx2label turns the label from text to a number from 0-(N-1) 
        if categories:
            self.labels = [main2label[cat] for cat in input_labels]
        else:
            self.labels = [class2label[acc2class[acc]] for acc in input_labels]
        
        # Tokenize sequence and pad to longest sequence in dataset.
        # Return pytorch-type tensors
        self.sequences = tokenizer(
                                input_sequences,
                                padding = 'longest',
                                return_tensors = 'pt')
        # Save label type
        self.label_type_cat = categories
        
    def classes(self):
        
        # Returns the classes in the dataset (optional function)
        return self.labels

    def __len__(self):
        
        # Returns the number of samples in dataset (required)
        return len(self.labels)

    def __getitem__(self, idx):
        
        # Returns a sample at position idx (required)
        # The sample includes:
        # - Input id's for the sequence
        # - Attention mask (to only focus on the sequence and not padding)
        # - Label (one-hot encoded)
        
        input_ids = self.sequences['input_ids'][idx]
        attention_mask = self.sequences['attention_mask'][idx]
        label = torch.tensor(self.labels[idx])
        num_labels = len(main2label.values()) if self.label_type_cat else len(class2label.values())
        
        sample = dict()
        sample['input_ids'] = input_ids
        sample['attention_mask'] = attention_mask
        sample['label'] = one_hot(label,
                                  num_classes=num_labels).to(torch.float)
        return sample

# Training using pytorch only

num_labels = len(set(train_dataset.classes()))
batch_size = 1

model = ESMForSequenceClassification.from_pretrained(
    "facebook/esm-1b",
    num_labels = num_labels,
    problem_type = "multi_label_classification")

device = torch.device("cpu")
model.to(device)
model.train()

# Train loader returns an iter with the number of samples = batch size.
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(batch_size):
    for batch in train_loader:
        
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask = attention_mask, labels=label)
        
        loss = outputs[0]
        
        loss.backward()
        optim.step()

model.eval()
