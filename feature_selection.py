#!/usr/bin/env python3
import math
import sys

def load_data(filename):
    #  Read the data file line by line. 
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # Skip blank lines (if any)
                    continue
                parts = line.split()
                # The first token is the class label. Convert to int.
                label = int(float(parts[0]))
                # The rest of the tokens are feature values (floats).
                features = [float(x) for x in parts[1:]]
                # Append a tuple: (label, [f1, f2, f3, ...])
                data.append((label, features))
    except FileNotFoundError:
        # If the file isn't found, print an error and quit.
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    return data

def euclid_dist(vec1, vec2):
   #in this I Compute Euclidean distance between two equal-length lists
    total = 0.0
    for a, b in zip(vec1, vec2):
        diff = a - b
        total += diff * diff
    return math.sqrt(total)

