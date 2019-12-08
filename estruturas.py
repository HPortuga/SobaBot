import pdb

entidades = {
  "numInt": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  "numStr": ["um", "uma", "dois", "duas", "tres", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez"],
  "tam": ["g", "grande", "grandes", "m", "medio", "medios", "p", "pequeno", "pequenos"],
  "prato": ["soba", "sobas"],
  "tipo": ["bovino", "carne", "boi", "vaca", "frango", "galinha"],
  "adicional": ["gengibre", "shoyu", "shoyo", "cebolinha", "ovo", "omelete", "adicional"],
  "bebidas": ["agua", "aguas", "água", "águas", "coca", "coca-cola", "cocas", "fanta", "fantas", "suco", "uva", "pessego", "delvale", "del", "vale"],
  "detalhe": ["com", "sem"],
  "tipoAgua": ["gas", "gaseificada", "normal", "natural"]
}

agua = ["agua", "aguas", "água", "águas"]
refri = ["coca", "coca-cola", "cocas", "fanta", "fantas"]
suco = ["suco", "uva", "pessego", "delvale", "del", "vale"]
lata = refri + suco

def strToInt(numStr):
  if (numStr == "1" or numStr == "um" or numStr == "uma"):
    return 1
  if (numStr == "2" or numStr == "dois" or numStr == "duas"):
    return 2
  if (numStr == "3" or numStr == "tres" or numStr == "três"):
    return 3
  if (numStr == "4" or numStr == "quatro"):
    return 4

def isCoca(coca):
  return coca == "coca" or coca == "cocas" or coca == "coca-cola"

def isRefri(ref):
  return ref in refri

def isSuco(suc):
  return suc in suco

def isLata(can):
  return can in lata

def isDetalhe(det):
  return det in entidades["detalhe"]

def isTipoAgua(tipo):
  return tipo in entidades["tipoAgua"]

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
        elif (isLata(current[1])):
          state = 6

    elif (state == 3):  # Agua
      try:
        nextToken = pares[pos+1]
      except IndexError:
        pedido.append((quantidade, current[1]))
        pos += 1
        state = 0
        continue

      if (not isDetalhe(nextToken[1]) and not isTipoAgua(nextToken[1])):
        pedido.append((quantidade, current[1]))
        pos += 1
        state = 0

      else:
        if (nextToken[1] == "com"):
          state = 4
        elif (nextToken[1] == "sem"):
          state = 5
        pos += 1

    elif (state == 4):  # Agua com Gas
      if (current[1] == "gaseificada"):
        pedido.append((quantidade, "agua com gas"))
        pos += 1
        state = 0

      elif (current[1] == "com"):
        try:
          nextToken = pares[pos+1]
          if (nextToken[1] == "gas"):
            pedido.append((quantidade, "agua com gas"))
            pos += 2
            state = 0

        except IndexError:
          state = -1
          continue

    elif (state == 5):  # Agua sem Gas
      if (current[1] == "natural" or current[1] == "normal"):
        pedido.append((quantidade, "agua"))
        pos += 1
        state = 0

      elif (current[1] == "sem"):
        try:
          nextToken = pares[pos+1]
          if (nextToken[1] == "gas"):
            pedido.append((quantidade, "agua"))
            pos += 2
            state = 0
        
        except IndexError:
          state = -1
          continue

    elif (state == 6):  # Lata
      if (isRefri(current[1])):
        state = 7
      elif (isSuco(current[1])):
        state = 8

    elif (state == 7): # Refri
      if (isCoca(current[1])):
        try:
          nextToken = pares[pos+1]
          if (nextToken[1] == "cola" or nextToken[1] == "colas"):
            pos += 2
          else:
            pos += 1
        except IndexError:
          pos += 1
        
        pedido.append((quantidade, "coca"))
        state = 0
        
    elif (state == 8): # Suco
      pass
    
    elif (state == -1): 
      return naoEntendi()

montarPedido(['numStr uma', 'bebidas coca', 'bebidas cola'])

