import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Name of file to process
filename = 'C:\\Users\\CHEELA RAMU HEMANTH\\OneDrive\\Desktop\\dataset\\mushroom\\agaricus-lepiota.data'

# Learn the names of all categories present in the dataset,
# and map them to 0,1,2,...
col_maps = {}

print("Processing", filename, "...")
with open(filename) as csvfile:
    fr = csv.reader(csvfile, delimiter=',') 
    rows = 0
    for row in fr:
        rows += 1
        if rows == 1:
            columns = len(row)
            for c in range(columns):
                col_maps[c] = {}

        for (c,label) in enumerate(row):
            if label not in col_maps[c]:
                index = len(col_maps[c])
                col_maps[c][label] = index
print("Done")
                
print("Read %d rows having %d columns." % (rows, columns))
print("Category maps:")
for c in range(columns):
    print("\t Col %d: " % c, col_maps[c])

# Construct matrix X, containing the mapped 
# features, and vector y, containing the mapped
# labels.
X = []
y = []

print("Converting", filename, "...")
with open(filename) as csvfile:
    fr = csv.reader(csvfile, delimiter=',') 
    for row in fr:
        label = row[0]
        y.append(col_maps[0][label])

        features = []
        for (c,label) in enumerate(row[1:]):
            features.append(col_maps[c+1][label])
        
        X.append(features)

print("Done")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and testing sets (1% for training, 99% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0], X_test.shape[0]))

# Define alpha values to try
alpha_values = np.linspace(2**-15, 2**5, 10000)

# Initialize lists to store performance metrics
auc_scores = []
accuracy_scores = []
f1_scores = []

min_categories = [x+1 for x in np.max(X, 0)]

# Train classifiers with different alpha values and evaluate performance
for alpha in alpha_values:
    classifier = CategoricalNB(alpha=alpha, min_categories=np.int64(min_categories))
    classifier.fit(X_train, y_train)
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    
    auc_scores.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Plot the performance metrics
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, auc_scores, label='ROC AUC')
plt.plot(alpha_values, accuracy_scores, label='Accuracy')
plt.plot(alpha_values, f1_scores, label='F1 Score')
plt.xlabel('Smoothing Hyperparameter- Alpha')
plt.ylabel('Performance Metric')
plt.title('Performance of Categorical Naive Bayes Classifier with Different Alpha Values')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Find the alpha value that maximizes AUC
max_auc_index = np.argmax(auc_scores)
optimal_alpha = alpha_values[max_auc_index]
print("Optimal Alpha Value:", optimal_alpha)

# Train the classifier with the optimal alpha value
classifier = CategoricalNB(alpha=optimal_alpha, min_categories=np.int64(min_categories))
classifier.fit(X_train, y_train)
y_pred_proba_optimal = classifier.predict_proba(X_test)
y_pred_optimal = classifier.predict(X_test)

# Calculate performance metrics for the optimal alpha value
optimal_auc_score = roc_auc_score(y_test, y_pred_proba_optimal[:, 1])
optimal_accuracy_score = accuracy_score(y_test, y_pred_optimal)
optimal_f1_score = f1_score(y_test, y_pred_optimal)

# Print the performance metrics for the optimal alpha value
print("Performance Metrics for Optimal Alpha Value (Alpha = {}):".format(optimal_alpha))
print("\tROC AUC:", optimal_auc_score)
print("\tAccuracy:", optimal_accuracy_score)
print("\tF1 Score:", optimal_f1_score)
