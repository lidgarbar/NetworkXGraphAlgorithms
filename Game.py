class Grafo:
    def __init__(self,nodes,edges):
        self.nodes=nodes
        self.edges=edges
    
    def rep_dict(self):
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
    
    def matriz_adyacencia(self):
        #Vamos iterando por filas (num filas== num nodos)
        matriz=[]
        for n_fila in self.nodes:
            #Vamos iterando por columnas (num columnas==num nodos)
            fila=[]
            for n_columna in self.nodes:
                #Cogemos los adyacentes al nodo de la fila
                ady_fila= self.rep_dict(self)[n_fila]
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
    def colorea_arista(self,grafo,subgrafo,arista):
        #Primero revisamos que la arista se encuentra en el grafo, sino se encuentra devolvemos el error 
        if(not (arista in grafo.edges)):
            return 'Esa arista no se encuentra en el grafo, pruebe con otra'
        
        #Si la arista si esta en el grafo, coloreamos dicha arista en el subgrafo
        subgrafo.edges.append(arista)

        #Miramos si vale True, en ese caso decimos que ha ganado el J1
        return  'El jugador 1 ha ganado' if(self.ganador(self,subgrafo)) else 'Turno jugador 1'


    def borra_arista(self,grafo,subgrafo,arista):
        #Primero revisamos que esa arista no ha sido coloreada, si es asi le devolvemos un errro
        if(arista in subgrafo.edges):
            return 'Esa arista no puede ser borrada, ya ha sido coloreada por el otro jugador'
        
        #Si la arista no estaba coloreada la borramos
        subgrafo.edges.pop(arista)

        #Miramos si vale False en ese caso ha ganado el J2
         
        return 'Turno jugador 1' if(self.ganador(self,grafo)) else 'El jugador 2 ha ganado'



    def ganador(self,subgrafo):
        #Cogemos como traget el ultimo vertice y source el primero
        source=self.vertices[0]
        target=self.vertices[1]
        
        #Cogemos la lista de adyacencias
        ady=subgrafo.rept_dict()
        
        for n in ady[source]:
            #Si ya hay una arista de source a target, ha ganado 
            if(n==target):
                return True
            #Sino hacemos llamada de la funcion auxiliar hayCamino para cada nodo asyacente a source
            else:
                #Quitamos de la lista de adyacencia 
                ady.pop(source)
                if(hayCamino(ady,n,target)): return True
        
        #Sino se hizo return es que no hemos llegado a un camino, por tanto devolvemos False
        return False

#Definimos una función auxiliar para ver si hay algun camino entre un vertic eu otro
def hayCamino(ady,source,target):
    #Si ya habiamos accedido a dicha source, habremos borrado los adyacente por tanto nos dara None y devolvemos False
    if(ady[source]==None):
        return False

    #Para cada nodo adyacente del source actual
    for n in ady[source]:
        #Miramos si ya hemos llegado al target y si es asi devolvemos True
        if(n==target):
            return True
        #Eoc, quitamos la lista de adyacencia del source y volvemos a llamar a hayCamino con todos los nodos de los targets
        else:
            ady.pop(source)
            if(hayCamino(ady,n,target)): return True
                
    #Sino se hizo return es que no hemos llegado a un camino, por tanto devolvemos False
    return False