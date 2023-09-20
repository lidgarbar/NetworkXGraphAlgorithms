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
                dicc=self.rep_dict()
                ady_fila= dicc[n_fila]
                #Si el nodo de la columna s eencuentra en la lista de los adyacente le ponemos un 1
                if(n_columna in ady_fila):
                    fila.append(1)
                #Eoc le ponemos un 0
                else:
                    fila.append(0)
            matriz.append(fila)
        
        #Finalmente devolvemos la matriz de adyacencia
        return matriz

class Juego:
    def __init__(self,grafo,subgrafo,vertices):
        self.grafo=grafo
        self.subgrafo=subgrafo
        self.vertices=vertices
        print('Se incia el juego, comienza el jugador 1. Elija una arista a colorear',grafo.edges)
    
    #Para colorea arista recibimos el subgrafo a colorear, el grafo en el estado que este y la arista
    def colorea_arista(self,grafo,subgrafo,arista):
        #Primero revisamos que la arista se encuentra en el grafo, sino se encuentra devolvemos el error 
        if(arista not in grafo.edges):
            return print('Esa arista no se encuentra en el grafo, pruebe con otra')
        
        #Si la arista se ha coloreado ya 
        if(arista in subgrafo.edges):
            return print('Esa arista ya ha sido coloreada, escoja otra porfavor')
        
        #Si la arista si esta en el grafo, coloreamos dicha arista en el subgrafo
        subgrafo.edges.append(arista)

        #Miramos si vale True, en ese caso decimos que ha ganado el J1
        return  print('\nEl jugador 1 ha ganado') if(self.ganador(subgrafo)) else print('\nSe ha coloreado la artista',arista,' \nTurno jugador 2, diga que arista desee borrar de las siguientes \n',grafo.edges)


    def borra_arista(self,grafo,subgrafo,arista):
        #Primero revisamos que esa arista no ha sido coloreada, si es asi le devolvemos un errro
        if(arista in subgrafo.edges):
            return print('Esa arista no puede ser borrada, ya ha sido coloreada por el otro jugador')
        
        #Si la arista no existe, le devolvemos un error
        if(arista not in grafo.edges):
            return print('Esa arista no existe en el grafo, escoja otra porfavor')
        

        #Eoc la borramos
        grafo.edges.remove(arista)

        #Miramos si vale False en ese caso ha ganado el J2
        return print('\nSe ha borrado la arista',arista,' \nTurno jugador 1, diga que arista desea colorear de las siguientes \n',grafo.edges) if(self.ganador(grafo)) else print('El jugador 2 ha ganado')



    def ganador(self,subgrafo):
        #Cogemos como traget el ultimo vertice y source el primero
        source=self.vertices[0]
        target=self.vertices[1]
        
        #Cogemos la lista de adyacencias
        ady=subgrafo.rep_dict()
        
        #Creamos una lista auxiliar para ver que nodos han sido viisitados
        visitados=[]
        
        #Por cada nodo adyacente al source
        for n in ady[source]:

            #Si ya hay una arista de source a target, ha ganado 
            if(n==target):
                return True
                
            #Sino hacemos llamada de la funcion auxiliar hayCamino para cada nodo adyacente a source
            else:
                #Añadimos a la lista de visitados el source 
                visitados.append(source)
                if(hayCamino(ady,visitados,n,target)): return True
        
        #Sino se hizo return es que no hemos llegado a un camino, por tanto devolvemos False
        return False

#Definimos una función auxiliar para ver si hay algun camino entre un vertic eu otro
def hayCamino(ady,visitados,actual,target):
    #Si ya habiamos accedido al nodo actual devolvemos False
    if(actual in visitados):
        return False

    #Para cada nodo adyacente del nodo actual
    for n in ady[actual]:
        #Miramos si ya hemos llegado al target y si es asi devolvemos True
        if(n==target):
            return True
        #Eoc, añadimos a visitados el actual y volvemos a llamar a hayCamino con todos los nodos adyacentes al actual
        else:
            visitados.append(actual)
            if(hayCamino(ady,visitados,n,target)): return True
                
    #Sino se hizo return es que no hemos llegado a un camino, por tanto devolvemos False
    return False

###########################################################
# Pruebas hechas para ver que todo funciona correctamente #
###########################################################
def carga_datos_ejemplo():
    vertices=[1,2,3,4]
    aristas=[(1,2),(1,4),(2,3),(2,4),(3,4)]
    grafo=Grafo(vertices,aristas)
    subgrafo_vacio=Grafo(vertices,list())
    subgrafo_no_vacio=Grafo(vertices,[(1,2)])
    nodosJuego=[1,3]
    print('Los datos han sido cargados correctamente')
    juego=Juego(grafo,subgrafo_vacio,nodosJuego)
    juego_iniciado=Juego(grafo,subgrafo_no_vacio,nodosJuego)
    return juego,juego_iniciado,grafo,subgrafo_vacio,subgrafo_no_vacio


def simulacion_juego_ejemplo_J1():
    juego,juego_iniciado,grafo,subgrafo_vacio,subgrafo_no_vacio=carga_datos_ejemplo()
    juego.colorea_arista(grafo,subgrafo_vacio,(1,2))
    juego.borra_arista(grafo,subgrafo_vacio,(1,4))
    juego.colorea_arista(grafo,subgrafo_vacio,(2,3))
    

def simulacion_juego_ejemplo_J2():
    juego,juego_iniciado,grafo,subgrafo_vacio,subgrafo_no_vacio=carga_datos_ejemplo()
    juego.colorea_arista(grafo,subgrafo_vacio,(1,2))
    juego.borra_arista(grafo,subgrafo_vacio,(2,3))
    juego.colorea_arista(grafo,subgrafo_vacio,(1,4))   
    juego.borra_arista(grafo,subgrafo_vacio,(3,4))
    
def test_errores_funciones():
    juego,juego_iniciado,grafo,subgrafo_vacio,subgrafo_no_vacio=carga_datos_ejemplo()
    print('\n###########################################################\n#             Pruebas funcion borrar arista               #\n###########################################################')
    print('\n* PRUEBA 1 Borramos una arista que no existe: ')
    juego.borra_arista(grafo,subgrafo_vacio,(1,5))
    print('\n* PRUEBA 2 Borramos una arista coloreada:')
    juego_iniciado.borra_arista(grafo,subgrafo_no_vacio,(1,2))
    
    print('\n###########################################################\n#           Pruebas funcion colorear arista               #\n###########################################################')
    print('\n* PRUEBA 1 Coloreamos una arista que no existe: ')
    juego.colorea_arista(grafo,subgrafo_vacio,(1,5))
    print('\n* PRUEBA 2 Coloreamos una arista que ya ha sido coloreada')
    juego_iniciado.colorea_arista(grafo,subgrafo_no_vacio,(1,2))

###########################################################
#                     Fin Pruebas                         #
###########################################################
