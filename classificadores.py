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
  "weights": ["uniform", "distance"],             # Weight used in prediction
  "algorithm": ["ball_tree", "kd_tree", "brute"], # Algorithm used to compute the nearest neighbors
  "leaf_size": np.arange(1, 60, 5),               # Leaf Size passet to BallTree or KDTree
  "n_jobs": [1, 2, 3, -1]                         # Num of parallel jobs to run for neighbors search
  # "p": [1, 2, 3],                                 # Power parameter for the Minkowski metric
  # "metric": ["euclidean", "manhattan",            # Distance metric used for the tree
  #            "minkowski"],
}

###

def tuneParamsKnn(x, y, params):
  gridSearch = GridSearchCV(KNeighborsClassifier(), params, cv=2)
  gridSearch.fit(x, y)

  scores = list()
  means = gridSearch.cv_results_["mean_test_score"]
  stds = gridSearch.cv_results_["std_test_score"]
  params = gridSearch.cv_results_["params"]
  for mean, std, param in zip(means, stds, params):
    scores.append( {'mean':mean, 'std':std, 'params':param} )
  
  return scores


# Extracts X and Y from the dataset
def getDataAndLabels():
  fileName = "Dialogos.csv"
  df = pd.read_csv(fileName)

  # corpus = ["ooi", "quero agua", "quero soba", "bom dia"]
  # x = vectorizer.fit_transform(corpus)
  # y = np.array(["saudacao", "pedido", "pedido", "saudacao"])
  corpus = df[["sentenca"]].values
  x = vectorizer.fit_transform(corpus.ravel())
  y = df[["intencao"]].values.ravel()

  return (x,y)

if __name__ == "__main__":
  x, y = getDataAndLabels()

  print(tuneParamsKnn(x, y, paramsKnn))


  # model = KNeighborsClassifier(n_neighbors=1)
  # model.fit(x, y)

  # text = "Que horas abre?"
  # inst = vectorizer.transform([text])
  # print(model.predict(inst))

  