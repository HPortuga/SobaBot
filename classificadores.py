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

from Model import Model

### Global
vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")

# TODO: this has to be dynamic
nSplits = 5

## Classifier Params
paramsKnn = {
  {
    "n_neighbors": np.arange(1, 10, 2),                           # Num of neighbors used in classification
    "weights": ["uniform", "distance"],                           # Weight used in prediction
    "algorithm": ["ball_tree", "kd_tree", "brute"],               # Algorithm used to compute the nearest neighbors
    "leaf_size": np.arange(20, 50, 10),                           # Leaf Size passet to BallTree or KDTree
    "n_jobs": [1, 2, 3, -1],                                      # Num of parallel jobs to run for neighbors search
    "p": [1, 2, 3],                                               # Power parameter for the Minkowski metric
    "metric": "minkowski"                                         # Distance metric used for the tree
  },
  {
    "n_neighbors": np.arange(1, 10, 2),                           
    "weights": ["uniform", "distance"],                           
    "algorithm": ["ball_tree", "kd_tree", "brute"],               
    "leaf_size": np.arange(20, 50, 10),
    "metric": ["euclidean", "manhattan", "chebyshev"]
  }
}

paramsDecisionTree = {
  "criterion": ["gini", "entropy"],                             # Measures the quality of a split
  "splitter" : ["best", "random"],                              # The strategy used to choose the split at each node
  "max_depth": [None, 2, 4, 8, 16],                             # The maximum depth of the tree
  "min_samples_split": [1, 1.5, 1.75, 2, 2.5, 6],               # Minimum number of samples required to split an internal node
  "min_samples_leaf":  [0, 0.2, 0.5, 0.75, 1],                  # The minimum number of samples required to be at a leaf node.
  "min_weight_fraction_leaf": [0., 0.3, 0.5, 1.2, 2],           # Minimum wifhted fraction of the sum of total weights required to be a lead
  "max_features": ["auto", "sqrt", "log2", None, 1, 2.5, 3],    # The number of features to consider when looking for the best split
  "random_state": [1, 4, 8, 16, None],                          # Seed for the random number generator
  "max_leaf_nodes": [1, 5, 10, 20, None],                       # Max number of leaf nodes
  "min_impurity_decrease": [0., 0.2, 0.7, 1.2, 3],              # Node will be split of this split induces decrease of impurity
  "presort": [True, False]                                      # Presort data
}

paramsNaiveBayes = {
  "alpha": np.arange(0, 5, 0.5),                                # Smoothing parameter
  "fit_prior": [True, False],                                   # Learn class prior probabilities
}

paramsLogisticReg = {
  "penalty": ["none", "l1", "l2", "elasticnet"],                # Specify the norm used in the penalization
  "dual": [True, False],                                        # Dual or primal formulation
  "tol": [1e-3, 1e-4, 1e-5],                                    # Tolerance for stopping criteria
  "C": np.arange(0, 5, 0.25),                                   # Inverse of regularization strength
  "fit_intercept": [True, False],                               # Specifies if a constant should be added to the decision function
  "intercept_scaling": np.arange(0, 3, 0.5),                    # Scales intercepting
  "class_weight": ["balanced", None],                           # Weights associated with classes
  "random_state": [12, 34, 22233333, None, 91823765],           # Seed for the random number generator
  "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], # Algorithm for optimization
  "max_iter": np.arange(10, 200, 10),                           # Maximum number of iterations taken for the solvers to converge
  "multi_class": ["ovr", "multinomial", "auto"],                # Multi class
  "n_jobs": [1, 2, 3, -1],                                      # Num of parallel jobs to run for neighbors search
  "l1_ratio": [None, 0.0, 0.5, 1.0, 1.5, 2.0],                  # Elastic-net mixing parameter
}

paramsNeuralNetwork = {
  "hidden_layer_sizes": [(100,), (10,), (200,), (50,), (400,)], # Number of neurons in the hidden layer
  "activation": ["identity", "logistic", "tanh", "relu"],       # Activation function for the hidden layer
  "solver": ["lbfgs", "sgd", "adam"],                           # Solver for weight optimization
  "aplha": [1e-3, 1e-4, 1e-5],                                  # L2 penalty parameter
  "batch_size": ["auto", 10, 50, 100, 200],                     # Size of minibatches for stochastic optimizers
  "learning_rate": ["constant", "invscaling", "adaptive"],      # Learning rate schedule for weight updates
  "max_iter": np.arange(10, 400, 50),                           # Maximum number of iterations
  "tol": [1e-3, 1e-4, 1e-5],                                    # Tolerance for optimization
  "warm_start": [True, False],                                  # Reuse the solution of the previous call to fit as initialization
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
    Model("KNN", KNeighborsClassifier(), paramsKnn),
    # Model("Decision Tree", DecisionTreeClassifier(), paramsDecisionTree),
    # Model("Naive Bayes", MultinomialNB(), paramsNaiveBayes),
    # Model("Logistic Regression", LogisticRegression(), paramsLogisticReg),
    # Model("Neural Network", MLPClassifier(), paramsNeuralNetwork),
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

      # Test is (dataTest,labelsTest)
      # Leave the test aside and split the train data once more
      foldScore = list()
      for i, j in stratkFold.split(dataTrain, labelsTrain):
        xTrain, xTest = dataTrain[i], dataTrain[j]
        yTrain, yTest = labelsTrain[i], labelsTrain[j]

        model.setData(xTrain)
        model.setLabels(yTrain)
        model.randomSearchTune(nSplits)
        model.fit()
        predictedLabels = model.predict(xTest)

        # TODO: Log score for each fold
        accuracyScore = accuracy_score(yTest, predictedLabels)
        precisionScore = precision_score(yTest, predictedLabels, average="micro")
        recallScore = recall_score(yTest, predictedLabels, average="micro")
        
        foldScore.append({
          "accuracy": accuracyScore,
          "precision": precisionScore,
          "recall": recallScore,
          "params": model.bestParams
        })

        foldScore = sorted(foldScore, key=lambda k: k["accuracy"], reverse=True)

      # TODO: Log Strat K Fold score
      accuracy = 0
      precision = 0
      recall = 0
      for fold in foldScore:
        accuracy += fold["accuracy"]
        precision += fold["precision"]
        recall += fold["recall"]

      stratkFoldScores.append({
        "accuracy": accuracy / len(foldScore),
        "precision": precision / len(foldScore),
        "recall": recall / len(foldScore)
      })

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

    model.bestParams = max(paramDict.items())[0]
    model.finalScore = correct / len(looScore)
    modelScores[model.name] = model.finalScore

  # Find best classifier
  bestClassifier = max(modelScores.items())

  # Infinite loop with best classifier