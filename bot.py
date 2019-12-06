from sklearn.feature_extraction.text import TfidfVectorizer
from WhatsAPI.whats import Whats
import classificadores
import time
import pdb

def etiquetador(dicionario, frase):
  reconhecido = dict()

  palavras = frase.split(" ")

  for key, value in dicionario.items():
    for palavra in palavras:
      if (palavra in value):
        reconhecido[key] = palavra

  return reconhecido

if __name__ == "__main__":
  entidades = {
    "num": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "um", "uma", "dois", "duas",
            "tres", "quatro", "cinco", "seis", "sete", "oito", "nove"],
    "tam": ["g", "grande", "grandes", "m", "medio", "medios", "p", "pequeno", "pequenos"],
    "prato": ["soba", "sobas"],
    "tipo": ["bovino", "carne", "boi", "vaca", "frango", "galinha"],
    "adicional": ["gengibre", "shoyu", "shoyo", "cebolinha", "ovo", "omelete"],
    "detalhe": ["coloca", "colocar", "tira", "tirar", "com", "sem"],
    "bebidas": ["agua", "aguas", "com gas", "sem gas", "normal", "gaseificada", "suco",
                "sucos", "pessego", "uva", "coca", "cocas", "coca cola", "coca-cola",
                "cocas colas", "cocas cola", "colas", "fanta", "fanta laranja", "fantas",
                "refrigerante", "refrigerantes", "lata", "latas"]
  }

  vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")
  
  chosenClassifier = classificadores.findBestModel()
  print("Best score was %.2f from %s\n" 
    % (chosenClassifier.looFinalScore[0]["accuracy"], chosenClassifier.name))

  reconhecido = etiquetador(entidades, "Me ve um soba")
  print(reconhecido)

  # whats = Whats()

  # while(True):
  #   # pdb.set_trace()
  #   call, text = whats.waitForNewMessage()

  #   if (text != "" and call != None):
  #     vec = classificadores.vectorizer
  #     prediction = chosenClassifier.predict(vec.transform([text]))[0]
  #     whats.answer(call, prediction)
  #     time.sleep(0.75)

  #   call = None
  #   text = ""