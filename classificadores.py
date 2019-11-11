from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

### Global
vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")

## Classifier Params
paramsKnn = {
  "n_neighbors": np.arange(1, 10, 2),             # Num of neighbors used in classification
  # "weights": ["uniform", "distance"],             # Weight used in prediction
  # "algorithm": ["ball_tree", "kd_tree", "brute"], # Algorithm used to compute the nearest neighbors
  # "leaf_size": np.arange(1, 60, 5),               # Leaf Size passet to BallTree or KDTree
  # "n_jobs": [1, 2, 3, -1]                         # Num of parallel jobs to run for neighbors search
  # "p": [1, 2, 3],                                 # Power parameter for the Minkowski metric
  # "metric": ["euclidean", "manhattan",            # Distance metric used for the tree
  #            "minkowski"],
}

paramsDecisionTree = {
  "criterion": ["gini", "entropy"], # Measures the quality of a split
  "splitter" : ["best", "random"],  # calaboca vai tomanocu
}

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

# Logs the algorithm's result with GridSearchCV
def writeParameterTuningLog(scores, algorithm, params):
  log = open("./Logs/"+algorithm+".txt", "w")

  log.write("Parameter Tuning: %s" % algorithm)
  log.write("\n============================\n")
  log.write("The following parameters were tuned to the algorithm with exhaustive search:\n")
  
  for param in params:
    log.write("%s: %s\n" % (param, params[param]))

  log.write("\n============================\n")
  log.write("The results are:\n\n")

  i = 0
  for score in scores:
    log.write("%d\n" % i)
    log.write("  Mean: %f, Std: %f\n  Params: %s\n\n" % (score["mean"], score["std"], str(score["params"])))
    i += 1

  log.close()

if __name__ == "__main__":
  x, y = getDataAndLabels()

  bestParams, scoreKnn = tuneParamsKnn(x, y, paramsKnn)
  writeParameterTuningLog(scoreKnn, "KNN", paramsKnn)
  print(bestParams)

  model = KNeighborsClassifier().set_params(**bestParams)
  model.fit(x, y)

  accuracy = model.score(x, y)

  print("\nAcc: %f\n" % accuracy)

  text = "tem refrigerante"
  inst = vectorizer.transform([text])
  print(model.predict(inst))

  # model = KNeighborsClassifier(n_neighbors=1)
  # model.fit(x, y)

  # text = "Que horas abre?"
  # inst = vectorizer.transform([text])
  # print(model.predict(inst))

  