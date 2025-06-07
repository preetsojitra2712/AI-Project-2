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

