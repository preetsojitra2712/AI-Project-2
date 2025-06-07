# I start by importing the math functions and timing tools
import math
import time
import csv

def euclid_dist(vec1, vec2):
    # I walk through each pair of values, square the difference, and sum them
    total = 0.0
    for a, b in zip(vec1, vec2):
        diff = a - b
        total += diff * diff
    # I take the square root to finish computing the Euclidean distance
    return math.sqrt(total)

def read_data(file_path):
    # I open the file—if it’s CSV I skip the header, otherwise I read space-delimited
    labels = []
    records = []
    if file_path.lower().endswith('.csv'):
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # I skip the header row
            for row in reader:
                if not row:
                    continue
                *features, label = row
                labels.append(int(float(label)))
                records.append([float(x) for x in features])
    else:
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                labels.append(int(float(parts[0])))
                records.append([float(x) for x in parts[1:]])
    # I return both class labels and feature vectors
    return labels, records

def compute_loocv(labels, records, feats):
    #loop over every instance, using the rest to predict its label
    correct = 0
    total = len(records)
    for i in range(total):
        best = float('inf')
        prediction = None
        vec_i = records[i]
        for j in range(total):
            if i == j:
                continue
            vec_j = records[j]
            #select only the features we're testing
            sel_i = [vec_i[k] for k in feats]
            sel_j = [vec_j[k] for k in feats]
            #measure how close they are
            d = euclid_dist(sel_i, sel_j)
            if d < best:
                best, prediction = d, labels[j]
        #check if my guess matches the true label
        if prediction == labels[i]:
            correct += 1
    return correct / total

