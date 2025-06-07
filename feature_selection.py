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

def show_baseline(labels, records):
    #compute accuracy using every feature
    num_feats = len(records[0])
    all_feats = list(range(num_feats))
    acc = compute_loocv(labels, records, all_feats)
    pretty = ", ".join(str(f+1) for f in all_feats)
    print(f"Running nearest neighbor with all the features {{{pretty}}} has accuracy {acc*100:.2f}%\n")


def select_forward(labels, records):
    # add one feature at a time, keeping the one that improves accuracy most
    num_feats = len(records[0])
    chosen = []
    best_choice = []
    best_acc = 0.0
    print("=== Forward Selection ===")
    for _ in range(num_feats):
        candidate = None
        local_best = 0.0
        for f in range(num_feats):
            if f in chosen:
                continue
            trial = chosen + [f]
            acc = compute_loocv(labels, records, trial)
            pretty = ", ".join(str(x+1) for x in trial)
            print(f"   Using feature(s) {{{pretty}}} accuracy is {acc*100:.2f}%")
            if acc > local_best:
                local_best, candidate = acc, f
        if candidate is None:
            break
        chosen.append(candidate)
        pretty_chosen = ", ".join(str(x+1) for x in chosen)
        print(f"\nFeature set {{{pretty_chosen}}} was best with {local_best*100:.2f}%\n")
        if local_best > best_acc:
            best_acc, best_choice = local_best, chosen.copy()
    pretty_best = ", ".join(str(x+1) for x in best_choice)
    print(f"Finished Search! Best feature subset came out to be {{{pretty_best}}} with accuracy {best_acc*100:.2f}%\n")

def select_backward(labels, records):
    #start with all features, and drop one at a time if it improves accuracy
    num_feats = len(records[0])
    chosen = list(range(num_feats))
    best_choice = chosen.copy()
    best_acc = compute_loocv(labels, records, chosen)
    print("=== Backward Elimination ===")
    for _ in range(num_feats - 1):
        remove_feat = None
        local_best = 0.0
        for f in chosen:
            trial = [x for x in chosen if x != f]
            acc = compute_loocv(labels, records, trial)
            pretty = ", ".join(str(x+1) for x in trial)
            print(f"   Using feature(s) {{{pretty}}} accuracy is {acc*100:.2f}%")
            if acc > local_best:
                local_best, remove_feat = acc, f
        if remove_feat is None:
            break
        chosen.remove(remove_feat)
        pretty_chosen = ", ".join(str(x+1) for x in chosen)
        print(f"\nFeature set {{{pretty_chosen}}} was best with {local_best*100:.2f}%\n")
        if local_best > best_acc:
            best_acc, best_choice = local_best, chosen.copy()
    pretty_best = ", ".join(str(x+1) for x in best_choice)
    print(f"Finished Search! Best feature subset came out to be {{{pretty_best}}} with accuracy {best_acc*100:.2f}%\n")

def main():
   
    print("Select input method:")
    print("  1) Enter a test file name (choose CS205_small_Data__22.txt or CS205_large_Data__45.txt)")
    print("  2) Use my data (diabetes.csv)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        path = input("Enter path to your data file: ").strip()
    elif choice == "2":
        path = "diabetes.csv"
    else:
        print("Wrong selection!")
        return

    try:
        labels, records = read_data(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return

    feats = len(records[0])
    inst = len(records)
    print(f"\nThis dataset has {feats} features and {inst} instances.\n")
    show_baseline(labels, records)

    print("Please select an algorithm:")
    print("  1) Forward Selection")
    print("  2) Backward Elimination")
    algo = input("Enter 1 or 2: ").strip()
    print("\nBeginning Search...\n")
    start = time.perf_counter()
    if algo == "1":
        select_forward(labels, records)
    elif algo == "2":
        select_backward(labels, records)
    else:
        print("Wrong selection!")
        return
    end = time.perf_counter()

    print(f"Time taken by the algorithm: {(end - start)/60:.2f} minutes")

if __name__ == "__main__":
    main()
