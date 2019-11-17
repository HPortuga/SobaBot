from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import Logger

class Model():
  def __init__(self, name, classifier, possibleParams):
    self.name = name
    self.classifier = classifier
    self.possibleParams = possibleParams
    self.stratkFoldScores = list()


  def setData(self, x):
    self.data = x

  def setLabels(self, y):
    self.labels = y

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

  # Uses param 'item' to sort list in decrescent order
  def sortListByItem(self, item, sortMe):
    return sorted(sortMe, key=lambda k: k[item], reverse=True)

  # Tunes params for classifier with Stratified K Fold
  def tune(self, x, y, nSplits):
    self.foldScores = list()
    stratkFold = StratifiedKFold(nSplits)

    for trainIndex, testIndex in stratkFold.split(x, y):
      dataTrain, dataTest = x[trainIndex], x[testIndex]
      labelsTrain, labelsTest = y[trainIndex], y[testIndex]

      self.setData(dataTrain)
      self.setLabels(labelsTest)
      self.randomSearchTune(nSplits)
      self.fit()

      predictedLabels = self.predict(dataTrain)
      self.getScores(predictions, labelsTest, self.foldScores)

    self.sortListByItem("accuracy", self.foldScores)
    self.getMeanScores(self.foldScores, self.stratkFoldScores)









  # Uses random search to tune params
  def randomSearchTune(self, nSplits):
    randomSearch = RandomizedSearchCV(self.classifier, self.possibleParams, cv=nSplits)
    randomSearch.fit(self.data, self.labels)

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

  # Fits data and labels to classifier
  def fit(self):
    self.classifier.fit(self.data, self.labels)

  # Sets classifier's params
  def setParams(self, params):
    self.classifier.set_params(**params)

  # Returns predicted labels for given data
  def predict(self, data):
    return self.classifier.predict(data)