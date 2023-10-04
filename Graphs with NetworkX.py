import networkx as nx
import matplotlib.pyplot as plt

#1 trying the module networkx
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2 , 3])
G.add_edge(1 , 2)
G.add_edges_from([(1 , 2) , (1 , 3)])
G[1]
G.number_of_nodes()
G.number_of_edges()
G.remove_node(2)
G.remove_edge(1 , 3)

#2 Plotting the graph
nx.draw(G , with_labels = True , font_weight = 'bold')
plt.savefig( "graph.png" )
plt.show()


#3 Dominating set
#dominating_set(G , start_with = None )
#is_dominating_set(G , nbunch)

#Digraphs on networkx
DG = nx.DiGraph()
edges =[( " S " ," A " ) ,( " S " ," C " ) ,( " A " ," B " ) ,( " A " ," D " ) ,
         ( " C " ," D " ) ,( " D " ," B " ) ,( " B " ," T " ) ,( " D " ," T " )]
capacities = [{ " capacity " : i } for i in [2 ,3 ,2 ,1 ,3 ,1 ,2 ,2]]
DG.add_edges_from( edges )
attrs = dict ( zip ( edges , capacities ))
nx.set_edge_attributes( DG , attrs )

#Flows on networkx
nx.maximum_flow( DG , _s =1 , _t =3)
nx.maximum_flow_value( DG , _s =1 , _t =3)
nx.minimum_cut( DG , _s =1 , _t =3)
nx.minimum_cut_value( DG , _s =1 , _t =3)

#Examples of cuts and their capacity
print(nx.cuts(DG))


