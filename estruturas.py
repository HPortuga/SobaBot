import pdb

entidades = {
  "numInt": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  "numStr": ["um", "uma", "dois", "duas", "tres", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez"],
  "tam": ["g", "grande", "grandes", "m", "medio", "medios", "p", "pequeno", "pequenos"],
  "prato": ["soba", "sobas"],
  "tipo": ["bovino", "frango"],
  "adicional": ["gengibre", "shoyu", "shoyo", "cebolinha", "ovo", "omelete", "adicional"],
  "bebidas": ["agua", "aguas", "água", "águas", "coca", "coca-cola", "cocas", "fanta", "fantas", "suco", "sucos", "uva", "pessego", "pêssego"],
  "detalhe": ["com", "sem"],
  "tipoAgua": ["gas", "gaseificada", "normal", "natural"]
}

agua = ["agua", "aguas", "água", "águas"]
refri = ["coca", "coca-cola", "cocas", "fanta", "fantas"]
suco = ["suco", "sucos", "uva", "pessego", "pêssego"]
saborSuco = ["uva", "pessego", "pêssego"]
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

def isUva(suco):
  return suco == "uva"

def isPessego(suco):
  return suco == "pessego" or suco == "pêssego"

def isFanta(fanta):
  return fanta == "fanta" or fanta == "fantas"

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

def isPrato(prato):
  return prato in entidades["prato"]

def naoEntendi():
  return "Nao entendi"

def montarPedido(tokens):
  pos = 0
  state = 0
  quantidade = 1

  pares = list()
  pedido = list()
  
  if (len(tokens) < 0):
    state = -1
  
  for token in tokens:
    pares.append(tuple(token.split(" ")))
  
  while (True):
    if (pos == len(pares)):
      return pedido
    
    current = pares[pos]
    if (state == 0):  # Pedido
      if (isNum(current[0])):
        state = 1
      elif (isBebida(current[1])):
        state = 2
      elif (isPrato(current[1])):
        state = 9
      else: 
        state = -1

    elif (state == 1):  # Quantidade
      quantidade = strToInt(current[1])
      pos += 1
      state = 0

    elif (state == 2):  # Bebida
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

    elif (state == 7):  # Refri
      if (isCoca(current[1])):
        try:
          nextToken = pares[pos+1]
          if (nextToken[1] == "cola" or nextToken[1] == "colas" or nextToken[1] == "lata"):
            pos += 2
          else:
            pos += 1
        except IndexError:
          pos += 1
        
        pedido.append((quantidade, "coca"))
        state = 0

      elif (isFanta(current[1])):
        try:
          nextToken = pares[pos+1]
          if (nextToken[1] == "lata"):
            pos += 2
          else:
            pos += 1
        except IndexError:
          pos += 1

        pedido.append((quantidade, "fanta"))
        state = 0
    
    elif (state == 8):  # Suco
      if (current[1] == "suco" or current[1] == "sucos"):
        nextToken = pares[pos+1]

        if (nextToken[1] in saborSuco):
          pedido.append((quantidade, "suco " + nextToken[1]))
          pos += 2
          state = 0
          continue

        else: 
          state = -1

      pos += 1
      state = 0
    
    elif (state == 9):  # Prato
      pdb.set_trace()

      try:
        nextToken = pares[pos+1]
        tam = ""
        tipo = ""

        if (nextToken[0] == "tam"):
          tam = nextToken[1]
          pos += 1

          nextToken = pares[pos+1]

          if (nextToken[0] == "tipo"):
            tipo = nextToken[1]
            pos += 2

        elif (nextToken[0] == "tipo"):
          tipo = nextToken[1]
          pos += 1

          nextToken = pares[pos+1]

          if (nextToken[0] == "tam"):
            tam = nextToken[1]
            pos += 2

        if (tam != "" or tipo != ""):
          pedido.append((quantidade, "soba " + tam + " " + tipo))
          state = 0
        else:
          state = -1

      except IndexError:
        state = -1

    elif (state == -1): 
      return naoEntendi()

# print(montarPedido(['numStr um', 'prato soba', 'tam m', 'tipo bovino']))

