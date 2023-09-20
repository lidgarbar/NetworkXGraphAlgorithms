# Shannon-Switching-Game-Python
El juego consiste en un grafo conexo y dos jugadores J1 y J2. J1 tiene que intentar conectar dos vértices previamente seleccionados y J2 tiene que evitarlo.En cada turno, J1 colorea una arista y J2 borra una arista. A continuación, se describen los pasos y requisitos que deben seguir vuestras soluciones:
* Paso 1 Define una clase para los grafos simples. Un grafo es un par (V, A) donde V es unconjunto de v ́ertices y A es un conjunto de aristas. En este caso, vamos a representarV con una lista donde cada v ́ertice se denotar ́a por un n ́umero, y el conjunto de aristascon una lista de tuplas ordenadas.
* Paso 2 En la clase de los grafos, define dos métodos para representar los grafos, unodescribiendo la adyacencia de cada v ́ertice mediante un diccionario, y otro que devuelvala matriz de adyacencia.
* Paso 3 Define una clase para el juego que cumpla lo siguiente:
  *   a) Los argumentos de entrada:Grafo de partida.Subgrafo que va coloreando J1.Un par formado por los v ́ertices objetivo de J1 (a, b) con a, b ∈ V.
  *   b) Define un m ́etodo para que J1 pueda colorear una arista. Para ello, a ̃nade losv ́ertices y la arista coloreada al subgrafo que va coloreando J1.
  *   c) Define un m ́etodo para que J2 pueda borrar una arista del grafo de partida queno haya sido coloreada previamente.
  *   d ) Define un m ́etodo que determine si J1 ya ha ganado. Pista: Usa una funci ́onrecursiva auxiliar que determine dada la representaci ́on por adyacencias si existeun camino entre dos v ́ertices.e) Define un m ́etodo que determine si J2 ha ganado, es decir, si ya no existe ning ́uncamino entre a y b en el grafo de partida
