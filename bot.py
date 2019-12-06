from sklearn.feature_extraction.text import TfidfVectorizer
from WhatsAPI.whats import Whats
import classificadores
import time
import pdb


if __name__ == "__main__":
  vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")
  
  chosenClassifier = classificadores.findBestModel()
  print("Best score was %.2f from %s\n" 
    % (chosenClassifier.looFinalScore[0]["accuracy"], chosenClassifier.name))

  whats = Whats()

  while(True):
    # pdb.set_trace()
    call, text = whats.waitForNewMessage()

    if (text != "" and call != None):
      vec = classificadores.vectorizer
      prediction = chosenClassifier.predict(vec.transform([text]))[0]
      whats.answer(call, prediction)
      time.sleep(0.75)

    call = None
    text = ""