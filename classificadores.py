from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import numpy as np
import pandas as pd

from Model import Model

### Global
vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")

# TODO: this has to be dynamic
nSplits = 5

## Classifier Params
paramsKnn = {
  "n_neighbors": np.arange(1, 10, 2),               # Num of neighbors used in classification
  # "weights": ["uniform", "distance"],             # Weight used in prediction
  # "algorithm": ["ball_tree", "kd_tree", "brute"], # Algorithm used to compute the nearest neighbors
  # "leaf_size": np.arange(1, 60, 5),               # Leaf Size passet to BallTree or KDTree
  # "n_jobs": [1, 2, 3, -1]                         # Num of parallel jobs to run for neighbors search
  # "p": [1, 2, 3],                                 # Power parameter for the Minkowski metric
  # "metric": ["euclidean", "manhattan",            # Distance metric used for the tree
  #            "minkowski"],
}

paramsDecisionTree = {
  "criterion": ["gini", "entropy"],                         # Measures the quality of a split
  "splitter" : ["best", "random"],                          # The strategy used to choose the split at each node
  "max_depth": [None, 2, 4, 8, 16],                         # The maximum depth of the tree
  "min_samples_split": [1, 1.5, 1.75, 2, 2.5, 6],           # Minimum number of samples required to split an internal node
  "min_samples_leaf":  [1.2, 1.75, 2, 2.3, 4, 3.5],         # The minimum number of samples required to be at a leaf node.
  "min_weight_fraction_leaf": [0., 0.3, 0.5, 1.2, 2],       # Minimum wifhted fraction of the sum of total weights required to be a lead
  "max_features": ["auto", "sqrt", "log2", None, 1, 2.5, 3] # The number of features to consider when looking for the best split
  "random_state": [1, 4, 8, 16, None],                      # Seed for the random number generator
  "max_leaf_nodes": [1, 5, 10, 20, None],                   # Max number of leaf nodes
  "min_impurity_decrease": [0., 0.2, 0.7, 1.2, 3],          # Node will be split of this split induces decrease of impurity
  "presort": [True, False]                                  # Presort data
}

paramsNaiveBayes = {}

paramsLogisticReg = {}

paramsNeuralNetwork = {}

###

# Tunes the given algorithm's params and returns best params and score list
def tuneParamsKnn(x, y, params):                          
  gridSearch = GridSearchCV(KNeighborsClassifier(), params, cv=5)
  gridSearch.fit(x, y)

  scores = list()
  means = gridSearch.cv_results_["mean_test_score"]
  stds = gridSearch.cv_results_["std_test_score"]
  params = gridSearch.cv_results_["params"]
  for mean, std, param in zip(means, stds, params):
    scores.append( {'mean':mean, 'std':std, 'params':param} )

  scoreList = list()
  scoreList = sorted(scores, key=lambda k: k["mean"], reverse=True)
  bestParams = scoreList[0]["params"]
  
  return bestParams, scoreList

# Extracts X and Y from the dataset
def getDataAndLabels():
  fileName = "Dialogos.csv"
  df = pd.read_csv(fileName)

  ## Debug Input (smaller - for tests only)
  # corpus = ["bom dia", "oi", "gostaria de fazer um pedido", "quero pedir"]
  # x = vectorizer.fit_transform(corpus)
  # y = np.array(["saudacao", "saudacao", "pedido", "pedido"])

  ## Real Input
  corpus = df[["sentenca"]].values
  x = vectorizer.fit_transform(corpus.ravel())
  y = df[["intencao"]].values.ravel()

  return (x,y)

if __name__ == "__main__":
  # Instantiating learning algorithms
  models = [
    Model("KNN", KNeighborsClassifier(), paramsKnn),
    Model("Decision Tree", DecisionTreeClassifier(), paramsDecisionTree),
    Model("Logistic Regression", MultinomialNB(), paramsNaiveBayes),
    Model("Neural Network", MLPClassifier(), paramsNeuralNetwork),
  ]

  for model in models:
    x, y = getDataAndLabels()

    leaveOneOut = LeaveOneOut()
    leaveOneOut.get_n_splits(x)

    for trainIndex, testIndex in leaveOneOut.split(x):
      dataTrain, dataTest = x[trainIndex], x[testIndex]
      labelsTrain, labelsTest = y[trainIndex], y[testIndex]
    
      model.setData(dataTrain)
      model.setLabels(labelsTrain)

      model.tune(nSplits)
      model.fit()
    
      predictedLabels = model.predict(dataTest)

      # TODO: We have to log this metrics
      accuracy = accuracy_score(labelsTest, predictedLabels)
      precisionScore = precision_score(labelsTest, predictedLabels)
      recallScore = recall_score(labelsTest, predictedLabels)

      print("\n\n")
      print("Acc: %f\nPrecision: %f\nRecall: %f\n\n\n"
        % (accuracy, precisionScore, recallScore))


  # text = "Que horas abre?"
  # inst = vectorizer.transform([text])
  # print(model.predict(inst))

  
