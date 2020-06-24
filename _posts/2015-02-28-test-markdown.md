---
layout: post
title: K Nearest Neighbors 
subtitle: K Nearest Neighbors and implementation on Iris dataset
gh-repo:ivaben
gh-badge: [star, fork, follow]

comments: true
---
**1.kNN and how it works**

Let’s consider an example with two classes and two-dimensional feature vectors.

![image](https://github.com/ivaben/Ivana-dashboard.github.io/blob/master/img/knn_jpg.PNG)

We have two classes a blue one and an orange one and two-feature vectors with two-dimensions (x1 and x2).  Also, there are some training samples and for each new sample (green one) that we want to classify. Then, we want to calculate the distance of the green sample to each of the training samples by looking at the nearest neighbors. (In this case we will look at the three nearest neighbors (k=3) and then we’ll choose or predict the label based on the most common class labels.
In order to calculate the distances, we used the Euclidean distance.  In 2D example:

![image](https://github.com/ivaben/Ivana-dashboard.github.io/blob/master/img/ED_css.PNG)

In ED in 2D case of 2 points is defined as the square root over and we have for each feature vector 0-component, square difference so we have x2-x1 + y2-y1 squared.

~~~
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
~~~


**2. Visualization of DATASET:**

Let’s look at our dataset and figure out what are X and y. We will use Iris dataset from scikit-learn module.
We will get some train samples and test samples and the associated training labels and test labels. 
Graph with 3 classes (red, green and blue).

**3. Results of Using Coding:**

We have to create a **class KNN** and then the methods that we want to implement:

First, we have to implement the **Init method** with self and k=3 (which is a number of nearest neighbors). Inside the init we want to store the k (self.k = k).
Next, we have to implement the conventions from other machine learning libraries (ex. Scikit-learn library).
Another implementation is the **fit method** that will fit the training samples and training labels.
And last, implement the **predict method**, where we want to predict new samples.

In our Fit method, kNN algorithm does not involve training steps, basically we want to store our training samples.

In our predict method, we will get multiple samples using (X). We will write helper method = so we want to do this for each of the samples. We will make list of the predicted labels = []. (list comprehension) and then we want to call this self._predict with one sample(x) for all of our samples in our tests samples and then convert the list to NumPy array. 
We have to create a helper **method _predict** with only one sample (x).
The _predict method needs to be computed. We will calculate all of distances and look at the nearest neighbors and the labels of the nearest neighbors and then we ‘ll choose our most common class label (1). 

~~~
class KNN:
    def __init__(self, k=3):
        self.k  = k
    
    # implement init method
    def fit(self, X, y):
        # Store the training samples
        self.X_train = X
        self.y_train = y


    # implement predict method
    def predict(self, X):
        # make a comprehension list, call this self._predict 
        # with one sample(x)for all of samples in the tests samples (X)
        y_pred = [self._predict(x) for x in X]
        # convert the list to NumPy array
        return np.array(y_pred)


    # get helper method with one sample
    def _predict(self, x):
        # calculate all of distances and define this in separate method (call it:euclidean_distance)
        # calculate the distance of this one sample(x) to all the training samples using list comprehensions
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get the k samples of the nearest neighbors and the labels of the nearest neighbors and
        # sort out our distances
        k_neigbor_samples = np.argsort(distances) [:self.k] # array from beggining until self.k
        k_neigbor_labels = [self.y_train[i] for i in k_neigbor_samples]

        # get the most common class labels
        # import Counter module
        # get first or very most common (1)
        most_common_class_labels = Counter(k_neigbor_labels).most_common(1)
        return most_common_class_labels[0][0]
~~~


Then in our **test file** we will create classifier and k=3 neighbors and then to try 5 neighbors to experiment. We will fit X_train and y_train and predict the test samples.  Last, we will calculate the accuracy (how many of our predictions are correctly classified) using the accuracy method.






