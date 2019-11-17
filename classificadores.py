# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
import operator

from Model import Model

### Global
vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")

# TODO: this has to be dynamic
nSplits = 5

## Classifier Params
paramsKnn = {
  "n_neighbors": np.arange(1, 10, 2),                       # Num of neighbors used in classification
  "weights": ["uniform", "distance"],                       # Weight used in prediction
  "algorithm": ["ball_tree", "kd_tree", "brute"],           # Algorithm used to compute the nearest neighbors
  "metric": ["euclidean", "manhattan"]                      # Distance metric used for the tree
}


paramsDecisionTree = {
  "criterion": ["gini", "entropy"],                         # Measures the quality of a split
  "splitter" : ["best", "random"],                          # The strategy used to choose the split at each node
  "max_depth": [None, 2, 4, 8, 16],                         # The maximum depth of the tree
}

paramsNaiveBayes = {
  "alpha": np.arange(0, 5, 0.5),                            # Smoothing parameter
  "fit_prior": [True, False],                               # Learn class prior probabilities
}

paramsLogisticReg = {
  "dual": [True, False],                                    # Dual or primal formulation
  "tol": [1e-3, 1e-4, 1e-5],                                # Tolerance for stopping criteria
  "C": np.arange(0.1, 5, 0.25),                             # Inverse of regularization strength
  "fit_intercept": [True, False],                           # Specifies if a constant should be added to the decision function
  "intercept_scaling": np.arange(0.1, 3, 0.5),              # Scales intercepting
  "class_weight": ["balanced", None],                       # Weights associated with classes
  "multi_class": ["ovr", "auto"],                           # Multi class
}

paramsNeuralNetwork = {
  "activation": ["identity", "logistic", "tanh", "relu"],   # Activation function for the hidden layer
  "solver": ["sgd", "adam"],                                # Solver for weight optimization
  "alpha": [1e-3, 1e-4, 1e-5],                              # L2 penalty parameter
  "learning_rate": ["constant", "invscaling", "adaptive"],  # Learning rate schedule for weight updates
  "max_iter": np.arange(200, 501, 100),                     # Maximum number of iterations
  "warm_start": [True, False],                              # Reuse the solution of the previous call to fit as initialization
}

###

# Extracts X and Y from the dataset
def getDataAndLabels():
  fileName = "Dialogos.csv"
  df = pd.read_csv(fileName)

  corpus = df[["sentenca"]].values
  x = vectorizer.fit_transform(corpus.ravel())
  y = df[["intencao"]].values.ravel()

  return (x,y)

if __name__ == "__main__":
  models = [
    # Model("KNN", KNeighborsClassifier(), paramsKnn),
    Model("Decision Tree", DecisionTreeClassifier(), paramsDecisionTree),
    Model("Naive Bayes", MultinomialNB(), paramsNaiveBayes),
    # Model("Logistic Regression", LogisticRegression(), paramsLogisticReg),
    # Model("Neural Network", MLPClassifier(), paramsNeuralNetwork),       # WARNING: Neural Network takes too long!
  ]

  modelScores = dict()
  x, y = getDataAndLabels()
  for model in models:
    leaveOneOut = LeaveOneOut()
    stratkFold = StratifiedKFold(nSplits)

    looScore = list()
    stratkFoldScores = list()

    # This will train with every row except one, then test with that one
    for trainIndex, testIndex in leaveOneOut.split(x):
      dataTrain, dataTest = x[trainIndex], x[testIndex]
      labelsTrain, labelTest = y[trainIndex], y[testIndex]

      model.tune(x, y, nSplits)

      model.bestParams = foldScore[0]["params"]
      model.setData(dataTrain)
      model.setLabels(labelsTrain)
      model.fit()
      predictedLabels = model.predict(dataTest)

      accuracyScore = accuracy_score(labelTest, predictedLabels)
      precisionScore = precision_score(labelTest, predictedLabels, average="micro")
      recallScore = recall_score(labelTest, predictedLabels, average="micro")
      
      # TODO: Log LOO score
      looScore.append({
        "accuracy": accuracyScore,
        "precision": precisionScore,
        "recall": recallScore,
        "params": model.bestParams
      })

    correct = 0
    paramDict = dict()
    for prediction in looScore:
      if (prediction["accuracy"] == 1.0):
        correct += 1
        
        param = str(prediction["params"])
        paramDict[param] = 0

    for prediction in looScore:
      if (prediction["accuracy"] == 1.0):
        param = str(prediction["params"])
        paramDict[param] += param.count(param)

    accuracy = 0
    precision = 0
    recall = 0
    looFinalScore = list()
    for score in looScore:
      accuracy += score["accuracy"]
      precision += score["precision"]
      recall += score["recall"]
    
    looFinalScore.append({
      "accuracy": accuracy / len(looScore),
      "precision": precision / len(looScore),
      "recall": recall / len(looScore)
    })

    model.bestParams = max(paramDict.items(), key=operator.itemgetter(1))[0]
    looFinalScore["params"] = model.bestParams
    modelScores[model.name] = looFinalScore

  # Find best classifier
  bestClassifier = max(modelScores.items())

  # Log best params for best classifier so we don't have to tune it every time

  # Infinite loop with best classifier