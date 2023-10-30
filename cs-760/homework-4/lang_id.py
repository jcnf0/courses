import os
import numpy as np
import matplotlib.pyplot as plt

# Define the vocabulary
vocabulary = "abcdefghijklmnopqrstuvwxyz "

# Define the training data directory
data_dir = "./data/"

# Define the number of classes
num_classes = 3

# Define the smoothing parameter
alpha = 0.5

# Define the prior probabilities
prior_probs = np.zeros(num_classes)

# Define the count of each character in each class
char_counts = np.zeros((num_classes, len(vocabulary)))

# Loop through each training file and count the characters
for label in ['e', 'j', 's']:
    for i in range(10):
        # Get the class label from the filename
        filename = "{}{}.txt".format(label, i)
        if label == "e":
            class_idx = 0
        elif label == "j":
            class_idx = 1
        else:
            class_idx = 2
        
        # Read the file and count the characters
        with open(os.path.join(data_dir, filename), "r") as f:
            text = f.read().replace('\n', '')
            for char in text:
                if char in vocabulary:
                    char_counts[class_idx, vocabulary.index(char)] += 1

# Calculate the prior probabilities with additive smoothing
total_docs = len(os.listdir(data_dir))
num_docs = np.zeros(num_classes)
for label in ['e', 'j', 's']:
    for i in range(10):
        if label == "e":
            class_idx = 0
        elif label == "j":
            class_idx = 1
        else:
            class_idx = 2
        num_docs[class_idx] += 1

for i in range(num_classes):
    prior_probs[i] = np.log((num_docs[i] + alpha) / (np.sum(num_docs) + alpha*num_classes))

# Initialize the array to hold the probabilities
theta = np.zeros((num_classes, len(vocabulary)))

# Calculate the class conditional log probabilities for each class
for i in range(num_classes):
    theta[i] = np.log((char_counts[i] + alpha) / (np.sum(char_counts[i]) + alpha*len(vocabulary)))

# Show theta as a heatmap
plt.figure()
plt.imshow(np.exp(theta), cmap="hot", interpolation="nearest")
plt.title("Class Conditional Log Probabilities")
plt.xlabel("Character")
plt.ylabel("Class")
plt.xticks(np.arange(len(vocabulary)), list(vocabulary))
plt.yticks(np.arange(num_classes), ["English", "Japanese", "Spanish"])
# Make colorbar between 0 and 1
plt.clim(0, np.max(np.exp(theta)))
plt.colorbar()

plt.savefig("figures/theta_heatmap.pdf", bbox_inches="tight")

# Initialize the confusion matrix
confusion_matrix = np.zeros((num_classes, num_classes))

# Loop through each test file and calculate the posterior log probabilities
classes = ["English", "Japanese", "Spanish"]
for label in ['e', 'j', 's']:
    for i in range(10, 20):
        # Get the class label from the filename
        filename = "{}{}.txt".format(label, i)
        if label == "e":
            true_class_idx = 0
        elif label == "j":
            true_class_idx = 1
        else:
            true_class_idx = 2
        
        # Read the file and represent it as a bag-of-words count vector
        with open(os.path.join(data_dir, filename), "r") as f:
            text = f.read().replace('\n', '')
            x = np.zeros(len(vocabulary))
            for char in text:
                if char in vocabulary:
                    x[vocabulary.index(char)] += 1
        
        if label=="e" and i==10:
            print(x)
            # make a heatmap of x with vocabulary as x axis and 1 as y axis
            plt.figure()
            plt.imshow(x.reshape(1, len(vocabulary)), cmap="hot", interpolation="nearest")
            plt.title("Bag of Words Representation of English Text")
            plt.xlabel("Character")
            plt.xticks(np.arange(len(vocabulary)), list(vocabulary))
            plt.yticks([])
            plt.savefig("figures/bag_of_word_exp.pdf", bbox_inches="tight")

        # Calculate the class conditional log probabilities for each class
        class_probs = np.zeros(num_classes)
        for j in range(num_classes):
            class_probs[j] = np.sum(np.multiply(theta[j],x))
        print("Class probabilities for {}: {}".format(filename, class_probs))
        
        # Calculate the posterior log probabilities
        posterior_probs = class_probs + prior_probs
        
        # Determine the predicted class based on the maximum posterior log probability
        predicted_class_idx = np.argmax(posterior_probs)
        
        print("Posterior probabilities for {}: {}".format(filename, posterior_probs))
        print("Predicted class for {}: {}".format(filename, classes[predicted_class_idx]))
        # Update the confusion matrix
        confusion_matrix[true_class_idx, predicted_class_idx] += 1

# Print the confusion matrix
print("Confusion matrix:")
print(confusion_matrix)
