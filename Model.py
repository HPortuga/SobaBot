from sklearn.model_selection import GridSearchCV
import Logger

class Model():
  def __init__(self, name, classifier, possibleParams):
    self.name = name
    self.classifier = classifier
    self.possibleParams = possibleParams

  def setData(self, x):
    self.data = x

  def setLabels(self, y):
    self.labels = y

  # Tunes the given algorithm's params and returns best params and score list
  def tune(self):
    gridSearch = GridSearchCV(self.classifier, self.possibleParams, cv = 5)
    gridSearch.fit(self.data, self.labels)

    scores = list()
    means = gridSearch.cv_results_["mean_test_score"]
    stds = gridSearch.cv_results_["std_test_score"]
    params = gridSearch.cv_results_["params"]
    for mean, std, param in zip(means, stds, params):
      scores.append( {'mean':mean, 'std':std, 'params':param} )

    self.scoreList = list()
    self.scoreList = sorted(scores, key=lambda k: k["mean"], reverse=True)
    self.bestParams = self.scoreList[0]["params"]
    
    Logger.writeParameterTuningLog(self.scoreList, self.name, self.possibleParams)

    return self.bestParams, self.scoreList