import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import minkowski

# loading the csv file
def get_data(file):
    data = pd.read_csv(file)
    return data

# checking average and spread for 2 classes
def check_spread(data, label_col):
    labels = data[label_col].unique()[:2]
    d1 = data[data[label_col] == labels[0]].drop(columns=[label_col])
    d2 = data[data[label_col] == labels[1]].drop(columns=[label_col])

    # taking only number features
    d1 = d1.select_dtypes(include=[np.number])
    d2 = d2.select_dtypes(include=[np.number])

    # finding average of each feature
    m1 = d1.mean(axis=0)
    m2 = d2.mean(axis=0)

    # finding std dev
    s1 = d1.std(axis=0)
    s2 = d2.std(axis=0)

    # distance between both class centroids
    dist = np.linalg.norm(m1 - m2)

    return m1, s1, m2, s2, dist

# drawing histogram of any one column
def plot_hist(data, colname):
    vals = data[colname].dropna()
    m = np.mean(vals)
    v = np.var(vals)

    # making the histogram chart
    plt.hist(vals, bins=10, color='skyblue', edgecolor='black')
    plt.title(colname + " Histogram")
    plt.xlabel(colname)
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    return m, v

# plotting minkowski distance for r = 1 to 10
def draw_minkowski(data):
    num = data.select_dtypes(include=[np.number]).dropna()
    num = num[~num.isin([np.inf, -np.inf]).any(axis=1)]

    if len(num) < 2:
        print("Not enough rows.")
        return

    a = num.iloc[0].values
    b = num.iloc[1].values

    r_vals = range(1, 11)
    dists = [minkowski(a, b, p=r) for r in r_vals]

    # plotting the graph
    plt.plot(r_vals, dists, marker='o', color='green')
    plt.title('Minkowski r=1 to 10')
    plt.xlabel('r')
    plt.ylabel('Dist')
    plt.grid(True)
    plt.show()

# splitting into train and test sets
def split_data(data, label_col):
    data = data[data[label_col].isin(data[label_col].unique()[:2])]
    x = data.drop(columns=[label_col]).select_dtypes(include=[np.number]).dropna()
    y = data[label_col]
    return train_test_split(x, y.loc[x.index], test_size=0.3, random_state=42)

# training the model using kNN
def make_model(x1, y1, k=3):
    model1 = KNeighborsClassifier(n_neighbors=k)
    model1.fit(x1, y1)
    return model1

# checking how good it did
def get_acc(model1, x2, y2):
    return model1.score(x2, y2)

# getting predictions for some test points
def get_preds(model1, x2):
    return model1.predict(x2)

# seeing how accuracy changes with different k
def draw_acc_plot(x1, y1, x2, y2):
    k_vals = list(range(1, 12))
    accs = []

    for k in k_vals:
        mdl = KNeighborsClassifier(n_neighbors=k)
        mdl.fit(x1, y1)
        accs.append(mdl.score(x2, y2))

    # plot for accuracy vs k
    plt.plot(k_vals, accs, marker='o')
    plt.title('k vs Accuracy')
    plt.xlabel('k')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.show()

# checking confusion matrix + precision/recall/f1 etc
def check_matrix(model1, x1, y1, x2, y2):
    p1 = model1.predict(x1)
    p2 = model1.predict(x2)

    print("\nTrain:")
    print(classification_report(y1, p1))

    print("\nTest:")
    print(classification_report(y2, p2))

    print("Confusion:")
    print(confusion_matrix(y2, p2))


# main part where everything runs
if __name__ == "__main__":
    data = get_data("ecg_eeg_features.csv")  # reading the dataset
    label = data.columns[-1]  # using last col as label

    print("\nQ1")
    m1, s1, m2, s2, dist = check_spread(data, label)
    print(f"Centroid dist: {dist:.2f}")  # just printing the distance between classes

    print("\nQ2")
    m, v = plot_hist(data, data.select_dtypes(include=[np.number]).columns[0])
    print(f"Mean: {m:.2f}, Var: {v:.2f}")  # showing mean and variance of that column

    print("\nQ3")
    draw_minkowski(data)  # calling minkowski thing

    print("\nQ4")
    x_train, x_test, y_train, y_test = split_data(data, label)
    print(f"Training data shape: {x_train.shape}, Training labels: {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, Testing labels: {y_test.shape}")

    print("\nQ5")
    model = make_model(x_train, y_train, k=3)
    print(f"Model trained successfully with k = 3 and {len(x_train)} training samples")

    print("\nQ6")
    acc = get_acc(model, x_test, y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    print("\nQ7")
    p = get_preds(model, x_test[:5])
    print(f"Preds: {p.tolist()}")  # just showing first 5 predictions

    print("\nQ8")
    draw_acc_plot(x_train, y_train, x_test, y_test)  # drawing accuracy curve for diff k

    print("\nQ9")
    check_matrix(model, x_train, y_train, x_test, y_test)  # showing all metrics
