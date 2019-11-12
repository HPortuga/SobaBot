class Model():
  def __init__(self, name, classifier, possibleParams):
    self.name = name
    self.classifer = classifier
    self.possibleParams = possibleParams
    self.bestParams = None
    self.paramResults = None