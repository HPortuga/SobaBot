Trabalho desenvolvido por
  Matheus Soares
  Ícaro Brandão

O dataset foi criado no formato (intenção, sentença);

Escolhemos os seguintes classificadores:
  KNN
  Decision Tree
  Naive Bayes
  Regressão Logística
  Rede Neural (MLP)

Da linha 30 até a linha 65 estão definidos os possíveis parâmetros
para cada algoritmo de classificação. Alguns parâmetros foram 
omitidos devido ao alto custo computacional da calibração.

Deixamos a rede neural comentada no código pois sua calibração
levou mais que duas horas e não terminou. O trabalho foi concluído
na noite de domingo e não há mais tempo para rodar a rede neural.
Sua implementação ficará para a segunda fase do trabalho.

Para cada modelo instanciado, executamos a amostragem e o treinamento.
Esse processo foi feito da seguinte forma:

  A função 'train' executa o algoritmo Leave One Out. O mesmo 
  separa índices de treino e de teste, sendo que o teste contém
  apenas uma entrada. Os dados de treino restantes são encaminhados 
  para a função 'tune';

  A função 'tune' utiliza o algoritmo K Fold Estratificado, que 
  também separa os dados em treino e teste, mas utiliza uma 
  separação mais uniforme que o Leave One Out. Com os dados de 
  treino separados mais uma vez, passamos eles para a função
  'randomSearchTune';

  A função 'randomSearchTune' utiliza o algoritmo RandomSearchCV
  para calibrar hiper-parâmetros dos classificadores. Ela retorna
  a combinação de parâmetros que obteve o melhor desempenho;

  Após calibrar os parâmetros, voltamos para a função 'tune' e 
  realizamos os testes com os dados que o K Fold separou. O K Fold
  separa em 5 folds, então teremos 5 registros de métricas para
  analisar. Essas métricas são armazenadas numa lista ordenada 
  pela acurácia de forma decrescente, para que o parâmetro que 
  oferece o melhor desempenho esteja no topo da lista;

  Uma vez encontrados os melhores parâmetros para os Folds, voltamos
  para a função 'train'. Realizamos então o teste do Leave One Out.
  Para escolher o melhor parâmetro final, pegamos todos os parâmetros
  que acertaram o teste e retornamos o que apareceu mais vezes. É 
  feito então o log dos resultados;

Uma vez que o classificador foi treinado e o melhor parâmetro foi 
encontrado, voltamos para a função main. Todos os dados estão 
gravados, então procuramos qual classificador obteve o melhor 
desempenho. Para esse classificador, rodamos um loop infinito 
onde o usuário pode testar o classificador.

Sobre a pasta Logs:
A pasta Logs contém registros de desempenho dos classificadores.
Estão distribuídos em 3 subpastas:
  Calibration: A cada fold é feita a calibração de parâmetros do
  algoritmo. Os dados da calibração são sobrescritos a cada fold,
  mas o professor pode acompanhar durante a execução do programa.

  LeaveOneOut: Aqui ficam guardadas as métricas obtidas com o Leave
  One Out.

  Performance: Estão registradas as métricas finais de desempenho
  do classificador, assim como o melhor parâmetro selecionado para
  ele.
