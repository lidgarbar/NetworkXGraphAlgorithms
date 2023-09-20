class Grafo:
    def __init__(self,nodes,edges):
        self.nodes=nodes
        self.edges=edges
    
    def rep_dict():
        #Creamos el diccionario vacio
        d=dict()
        #Por cada nodo
        for n in self.nodes:
            #Creamos una lista de adyacentes
            ady=[]
            #Por cada arista 
            for e in self.edges:
                #Si el primer nodo (o el segundo) es el que estamos iterando
                if(e[0]==n or e[1]==n):
                    #Añadimos a la lista el nodo al que es adyacente
                    if(e[0]==n):
                        ady.append(e[1])
                    else:
                        ady.append(e[0])
            #Cuando acabemos le indexamos la lista de los adyacentes al diccionario
            d[n]=ady
        return d
    
    def matriz_adyacencia():
        #Vamos iterando por filas (num filas== num nodos)
        matriz=[]
        for n_fila in self.nodes:
            #Vamos iterando por columnas (num columnas==num nodos)
            fila=[]
            for n_columna in self.nodes:
                #Cogemos los adyacentes al nodo de la fila
                ady_fila=rep_dict()[n_fila]
                #Si el nodo de la columna s eencuentra en la lista de los adyacente le ponemos un 1
                if(n_columna in ady_fila):
                    fila.append(1)
                #Eoc le ponemos un 0
                else:
                    fila.append(0)
            matriz.append(fila)


class Juego:
    def __init__(self,grafo,subgrafo,vertices):
        self.grafo=grafo
        self.subgrafo=subgrafo
        self.vertices=vertices
    
    #Para colorea arista recibimos el subgrafo a colorear, el grafo en el estado que este y la arista
    def colorea_arista(grafo,subgrafo,arista):
        #Primero revisamos que la arista se encuentra en el grafo, sino se encuentra devolvemos el error 
        if(not (arista in grafo.edges)):
            return 'Esa arista no se encuentra en el grafo, pruebe con otra'