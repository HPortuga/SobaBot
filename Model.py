from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import operator
import Logger
import ast

class Model():
  def __init__(self, name, classifier, possibleParams):
    self.name = name
    self.classifier = classifier
    self.possibleParams = possibleParams
    self.stratkFoldScores = list()
    self.looFinalScore = list()

  def setData(self, x):
    self.data = x

  def setLabels(self, y):
    self.labels = y

  # Fits data and labels to classifier
  def fit(self, x, y):
    self.classifier.fit(x, y)

  # Sets classifier's params
  def setParams(self, params):
    self.classifier.set_params(**params)

  # Returns predicted labels for given data
  def predict(self, data):
    return self.classifier.predict(data)

  # Uses param 'item' to sort list in decrescent order
  def sortListByItem(self, item, sortMe):
    return sorted(sortMe, key=lambda k: k[item], reverse=True)

  # Sets best param as most recurring param that made a correct prediction
  def setBestParams(self, scoreList):
    paramDict = dict()
    for prediction in scoreList:
      if (prediction["accuracy"] == 1.0):
        param = str(prediction["params"])
        if (param in paramDict):
          paramDict[param] += 1
        else:
          paramDict[param] = 1
      
    self.bestParams = max(paramDict.items(), key=operator.itemgetter(1))[0]
    self.looFinalScore[0]["params"] = self.bestParams
    self.setParams(ast.literal_eval(self.bestParams))

  def getScores(self, predictions, labels, scoreList):
    accuracyScore = accuracy_score(labels, predictions)
    precisionScore = precision_score(labels, predictions, average="micro")
    recallScore = recall_score(labels, predictions, average="micro")

    scoreList.append({
      "accuracy": accuracyScore,
      "precision": precisionScore,
      "recall": recallScore,
      "params": self.bestParams
    })

  # Gets the mean score from scoreList
  def getMeanScores(self, scoreList, singleScore):
    accuracy = 0
    precision = 0
    recall = 0
    for score in scoreList:
      accuracy += score["accuracy"]
      precision += score["precision"]
      recall += score["recall"]

    singleScore.append({
        "accuracy": accuracy / len(scoreList),
        "precision": precision / len(scoreList),
        "recall": recall / len(scoreList)
      })

  # Trains the classifier
  def train(self, nSplits):
    self.looScores = list()
    leaveOneOut = LeaveOneOut()

    for trainIndex, testIndex in leaveOneOut.split(self.data):
      dataTrain, dataTest = self.data[trainIndex], self.data[testIndex]
      labelsTrain, labelTest = self.labels[trainIndex], self.labels[testIndex]

      self.tune(dataTrain, labelsTrain, nSplits)
      self.fit(dataTrain, labelsTrain)

      predictedLabels = self.predict(dataTest)
      self.getScores(predictedLabels, labelTest, self.looScores)

    self.sortListByItem("accuracy", self.looScores)
    self.getMeanScores(self.looScores, self.looFinalScore)
    self.setBestParams(self.looScores)

  # Tunes params for classifier with Stratified K Fold
  def tune(self, x, y, nSplits):
    self.foldScores = list()
    stratkFold = StratifiedKFold(nSplits)

    for trainIndex, testIndex in stratkFold.split(x, y):
      dataTrain, dataTest = x[trainIndex], x[testIndex]
      labelsTrain, labelsTest = y[trainIndex], y[testIndex]

      self.randomSearchTune(dataTrain, labelsTrain, nSplits)
      self.fit(dataTrain, labelsTrain)

      predictedLabels = self.predict(dataTest)
      self.getScores(predictedLabels, labelsTest, self.foldScores)

    self.sortListByItem("accuracy", self.foldScores)
    self.getMeanScores(self.foldScores, self.stratkFoldScores)
    self.bestParams = self.foldScores[0]["params"]
    self.setParams(self.bestParams)

  # Uses random search to tune params
  def randomSearchTune(self, x, y, nSplits):
    randomSearch = RandomizedSearchCV(self.classifier, self.possibleParams, cv=nSplits)
    randomSearch.fit(x, y)

    scores = list()
    means = randomSearch.cv_results_["mean_test_score"]
    stds = randomSearch.cv_results_["std_test_score"]
    params = randomSearch.cv_results_["params"]
    for mean, std, param in zip(means, stds, params):
      scores.append( {'mean':mean, 'std':std, 'params':param} )

    self.scoreList = list()
    self.scoreList = sorted(scores, key=lambda k: k["mean"], reverse=True)
    self.bestParams = self.scoreList[0]["params"]
    self.setParams(self.bestParams)
    
    Logger.writeParameterTuningLog(self.scoreList, self.name, self.possibleParams)