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

  # Tunes the given algorithm's params, sets algorithm's params and logs score
  def tune(self, nSplits):
    gridSearch = GridSearchCV(self.classifier, self.possibleParams, cv = nSplits)
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