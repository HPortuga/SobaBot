from sklearn.feature_extraction.text import TfidfVectorizer

import classificadores
from WhatsAPI.whats import Whats

if __name__ == "__main__":
  vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")
  
  chosenClassifier = classificadores.findBestModel()
  print("Best score was %.2f from %s" 
    % (chosenClassifier.looFinalScore[0]["accuracy"], chosenClassifier.name))

  whats = Whats()

  while(True):
    call, text = whats.run()
    vec = classificadores.vectorizer
    prediction = chosenClassifier.predict(vec.transform([text]))
    whats.answer(call, prediction)