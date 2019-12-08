import pdb

entidades = {
  "numInt": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  "numStr": ["um", "uma", "dois", "duas", "tres", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez"],
  "tam": ["g", "grande", "grandes", "m", "medio", "medios", "p", "pequeno", "pequenos"],
  "prato": ["soba", "sobas"],
  "tipo": ["bovino", "carne", "boi", "vaca", "frango", "galinha"],
  "adicional": ["gengibre", "shoyu", "shoyo", "cebolinha", "ovo", "omelete", "adicional"],
  "bebidas": ["agua", "aguas", "água", "águas", "coca", "fanta", "suco"],
  "detalhe": ["com", "sem"],
  "tipoAgua": ["gas", "gaseificada", "normal"]
}

agua = ["agua", "aguas", "água", "águas"]
lata = ["coca", "fanta", "suco"]

def strToInt(numStr):
  if (numStr == "1" or numStr == "um" or numStr == "uma"):
    return 1
  if (numStr == "2" or numStr == "dois" or numStr == "duas"):
    return 2
  if (numStr == "3" or numStr == "tres" or numStr == "três"):
    return 3
  if (numStr == "4" or numStr == "quatro"):
    return 4

def isNum(num):
  return num == "numInt" or num == "numStr"

def isBebida(bebida):
  return bebida in entidades["bebidas"]

def isAgua(bebida):
  return bebida in agua

def naoEntendi():
  return "Nao entendi"

def montarPedido(tokens):
  state = 0
  quantidade = 1
  if (len(tokens) < 0):
    state = -1

  pares = list()
  pedido = list()
  
  for token in tokens:
    pares.append(tuple(token.split(" ")))

  pos = 0
  
  pdb.set_trace()

  while (True):
    if (pos == len(pares)):
      print("Cheguei no final")
      return pedido
    
    current = pares[pos]
    if (state == 0):  # Pedido
      if (isNum(current[0])):
        state = 1
      elif (isBebida(current[1])):
        state = 2
      else: 
        state = -1

    elif (state == 1):  # Quantidade
      quantidade = strToInt(current[1])
      pos += 1
      state = 0

    elif (state == 2): # Bebida
        if (isAgua(current[1])):
          state = 3

    elif (state == 3):  # Agua
      try:
        nextToken = pares[pos+1]
      except IndexError:
        pedido.append((quantidade, current[1]))
        pos += 1
        
    elif (state == -1): 
      return naoEntendi()

montarPedido(['numStr uma', 'bebidas agua'])

