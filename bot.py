from sklearn.feature_extraction.text import TfidfVectorizer

import classificadores
from WhatsAPI.whats import Whats

if __name__ == "__main__":
    chosenClassifier = classificadores.findBestModel()
    # print("Best score was %.2f from %s" 
    #   % (chosenClassifier.looFinalScore[0]["accuracy"], chosenClassifier.name))

    whats = Whats()

    while(True):
      call, text = whats.run()
      # prediction = chosenClassifier.predict(vectorizer.transform([text]))
      whats.answer(call, "oi")