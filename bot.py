#encoding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from WhatsAPI.whats import Whats
import classificadores
import time
import pdb
import estruturas

def etiquetador(dicionario, frase):
  reconhecido = list()

  frase = frase.replace(",", "")
  palavras = frase.split(" ")

  for palavra in palavras:
    for key, value in dicionario.items():
      if (palavra in value):
        reconhecido.append(str(key + " " + palavra))

  return reconhecido

if __name__ == "__main__":
  vectorizer = TfidfVectorizer(sublinear_tf=True,
    max_df=0.5, strip_accents="unicode")
  
  chosenClassifier = classificadores.findBestModel()
  print("Best score was %.2f from %s\n" 
    % (chosenClassifier.looFinalScore[0]["accuracy"], chosenClassifier.name))


  whats = Whats()

  while(True):
    call, text = whats.waitForNewMessage()

    if (text != "" and call != None):
      text = text.lower()
      vec = classificadores.vectorizer

      # pdb.set_trace()

      if (text == "sim"):
        answer = "Ok! Precisamos do seu endereco agora"
        prediction = "endereco"
      else :
        prediction = chosenClassifier.predict(vec.transform([text]))[0]

      if (prediction == "saudacao"):
        answer = "Boa noite! Bem vindo à sobaria do Ninja. Como posso ajudar?"

      elif (prediction == "cardapio"):
        answer = "Trabalhamos com os seguintes itens:\n"
        answer += "- Sobás:\n"
        answer += " Sobá G Bovino - R$19.99\n"
        answer += " Sobá M Bovino - R$18.00\n"
        answer += " Sobá P Bovino - R$15.00\n"
        answer += " Sobá G de Frango - R$19.99\n"
        answer += " Sobá M de Frango - R$18.00\n"
        answer += " Sobá P de Frango - R$15.00\n"
        answer += "- Bebidas:\n"
        answer += " Água sem Gás - R$3.00\n"
        answer += " Água com Gás - R$3.00\n"
        answer += " Fanta Laranja lata - R$5.00\n"
        answer += " Coca-Cola lata - R$5.00\n"
        answer += " Suco Del Valle Pêssego lata - R$5.00\n"
        answer += " Suco Del Valle Uva lata - R$5.00\n"
        answer += "- Adicionais\n"
        answer += " Bovino 100g - R$10.00\n"
        answer += " Frango 100g - R$8.00\n"
        answer += " Cebolinha - R$3.00\n"
        answer += " Omelete - R$3.00\n"

      elif (prediction == "pedido"):
        ents = etiquetador(estruturas.entidades, text)

        order = estruturas.montarPedido(ents)

        if (order != estruturas.naoEntendi()):
          answer = estruturas.confirmarPedido(order)
        else:
          answer = order
      
      elif (prediction == "endereco"):  # TODO
        pass

      else:
        continue

      whats.answer(call, answer)

      time.sleep(0.75)

    call = None
    text = ""