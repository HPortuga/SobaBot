Calibração KNN:
  Para 1 vizinho, foram permutados os parâmetros e constata-se que nenhum 
  dos atributos fez diferença nos resultados. Na entrada #288, o número de 
  vizinhos foi alterado para 3. Percebemos uma queda na acurácia (80%->73%) 
  e um aumento do desvio padrão (0.4->0.7). Para os casos com 5, 7 e 9 
  vizinhos também ocorre a perda de acurácia. Para o KNN, então, utilizaremos 
  os seguintes parâmetros: 
    Params: {'algorithm': 'ball_tree', 'leaf_size': 1, 'n_jobs': 1, 'n_neighbors': 1, 'weights': 'uniform'}