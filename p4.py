import csv
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'. 
    """
    return s.lower()



def strip_non_alpha(s):
    """ Remove non-alphabetic characters from the beginning and end of a string. 

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle 
    of the string should not be removed. E.g. "haven't" should remain unaltered."""

    s = s.strip()
    if len(s)==0:
        return s
    if not s[0].isalpha():    
        return strip_non_alpha(s[1:])         
    elif not s[-1].isalpha():       
        return strip_non_alpha(s[:-1])        
    else:
        return s

def clean(s):
    """ Create a "clean" version of a string 
    """
    return to_lower_case(strip_non_alpha(s))


# Directory of text files to be processed

directory = 'C:\\Users\\CHEELA RAMU HEMANTH\\OneDrive\\Desktop\\dataset\\sentence+classification\\SentenceCorpus\\labeled_articles\\'
 


# Learn the vocabulary of words in the corpus
# as well as the categories of labels used per text

categories = {}
vocabulary = {}


num_files = 0
for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    num_files +=1
    print("Processing",filename,"...",end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f,'r') as  fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                if label not in categories:
                    index = len(categories)
                    categories[label] = index
                    
                for word in words:
                    if word not in vocabulary:
                        index = len(vocabulary)
                        vocabulary[word] = index
    print(" done")            

n_words = len(vocabulary)
n_cats = len(categories)

print("Read %d files containing %d words and %d categories" % (num_files,len(vocabulary),len(categories)))

print(categories)


# Convert sentences into a "bag of words" representation.
# For example, "to be or not to be" is represented as
# a vector with length equal to the vocabulary size,
# with the value 2 at the indices corresponding to "to" and "be",
# value 1 at the indices corresponding to "or" and "not"
# and zero everywhere else. 


X = []
y = []

for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    print("Converting",filename,"...",end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f,'r') as  fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                y.append(categories[label])

                features = n_words * [0]

                bag = {}
                for word in words:
                    if word not in bag:
                        bag[word] = 1
                    else:
                        bag[word] += 1
                
                for word in bag:
                    features[vocabulary[word]] = bag[word]

                X.append(features)
    print(" done")            

# Save X and y to files

with open('X_snts.csv', 'w') as csvfile:
    fw = csv.writer(csvfile, delimiter=',')
    for features in X:
        fw.writerow(features)

with open('y_snts.csv', 'w') as csvfile:
    fw = csv.writer(csvfile, delimiter=',')
    for label in y:
        fw.writerow([label])



# Load X and y from files
X = np.loadtxt('X_snts.csv', delimiter=',')
y = np.loadtxt('y_snts.csv', delimiter=',')

# Define alpha values to try
alpha_values = [2**i for i in range(-15, 6)]

# Number of iterations for computing average accuracy and standard deviation
num_iterations = 10

# Initialize lists to store accuracy metrics
average_accuracy = []
std_dev_accuracy = []

# Repeat the experiment multiple times with different random splits
for alpha in alpha_values:
    accuracies = []
    for _ in range(num_iterations):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        # Train the classifier
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Compute accuracy and store
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Compute average accuracy and standard deviation
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    average_accuracy.append(avg_accuracy)
    std_dev_accuracy.append(std_accuracy)

# Plot the average accuracy as a function of alpha
plt.errorbar(alpha_values, average_accuracy, yerr=std_dev_accuracy, fmt='o-', capsize=5)
plt.xlabel('Smoothing Hyperparameter (Alpha)')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy of Multinomial Naive Bayes Classifier')
plt.xscale('log')
plt.grid(True)
plt.show()

# Find the alpha value that maximizes average accuracy
max_avg_accuracy_index = np.argmax(average_accuracy)
optimal_alpha = alpha_values[max_avg_accuracy_index]
print("Optimal Alpha Value:", optimal_alpha)

                

# Train the classifier with the optimal alpha value
clf = MultinomialNB(alpha=optimal_alpha)
clf.fit(X, y)

# Get the top 5 words for each class
top_words = {}
for i, class_label in enumerate(clf.classes_):
    class_prob = clf.feature_log_prob_[i]
    top_word_indices = np.argsort(class_prob)[-5:][::-1]
    top_words[class_label] = [word for word, index in vocabulary.items() if index in top_word_indices]

# Print the top 5 words for each class
for class_label, words in top_words.items():
    print("Class:", class_label)
    print("Top 5 words:", words)