from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister,ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, partial_trace
from qiskit.circuit import Measure
import itertools
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.primitives import SamplerResult
from qiskit.providers.basic_provider import BasicProvider
from qiskit import transpile
import numpy as np
from latticecode_with_nx import *
from CSSCodesGottesman import *
from networkx.algorithms import bipartite
import math 

        
def syndrome_measurement(LatticeCircuit, tanner_graph, checks, label):
    
    ## label is 'X' or 'Z' 
    options = {'X' : LatticeCircuit.cx, 'Z': LatticeCircuit.cz}
    
    
    for check_node in checks:
        syndrome = AncillaRegister(1)
        meas = ClassicalRegister(1)
        LatticeCircuit.add_register(syndrome)
        LatticeCircuit.add_register(meas)
        
        LatticeCircuit.h(syndrome)
        
        
        for idx in tanner_graph.neighbors(check_node):
            options[label](syndrome, LatticeCircuit.qubits[idx])
            
        LatticeCircuit.h(syndrome)
        
        LatticeCircuit.measure(syndrome,meas)
        
        
            
        # ### reset syndrome
        # LatticeCircuit.h(syndrome)
        
        
        # for idx in tanner_graph.neighbors(check_node):
        #     options[label](syndrome, LatticeCircuit.qubits[idx])
            
        # LatticeCircuit.h(syndrome)
        
        
        
    

def ApplyPauliError(quantum_circuit, qubits, p_error, print_option=False):
    
    
    for qubit in qubits:
        error_choice = np.random.choice([0,1,2,3],p=[1 - p_error, p_error/3, p_error/3, p_error/3])
        if error_choice == 1:
            if print_option:
                print( 'X on ' + str(qubit)  )
            quantum_circuit.x(qubit)
        if error_choice == 2:
            if print_option:
                print( 'Z on ' + str(qubit)  ) 
            quantum_circuit.z(qubit)
        if error_choice == 3:
            if print_option:
                print( 'Y on ' + str(qubit)  )
            quantum_circuit.y(qubit)
            

def SurfaceCode_single_round( x_0, k_0,k_1, p_error, report_error=False):
    
    Surface_code = SurfaceCode(k_0,k_1, initialize = x_0, p_error = p_error, report_error=report_error)
    
    Surface_code.syndrome_cycle()
    
    Surface_code.measure_data()
    
    return Surface_code 
    

class SurfaceCode:
    
    def __init__(self, rows, cols, initialize=0,p_error=0,report_error=False):
        
        self.rows = rows
        self.cols = cols
        self.lattice_grid =  nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows+3,cols+1), first_label=-(cols+1))
        self.edges = self.lattice_grid.edges()
        self.num_of_qubits = rows*(cols+1) + (rows+1)*(cols) + 2*(cols + 1)
        list_edges = list(self.edges)
        

    ########## produce cycles/faces on lattice
        self.lattice_cycles = []
        for row_idx in range(0, len(list_edges) - rows - cols - 1,cols*2 + 1):
            for col_idx in range(row_idx, row_idx + cols*2, 2):    
                cycle = [list_edges[col_idx],list_edges[col_idx +1],list_edges[col_idx +2], (list_edges[col_idx ][1],list_edges[col_idx +2][1])]
                for pair in cycle:
                    if (pair[0] < 0 and pair[1] < 0) or (pair[0] >= ( cols + 1 ) * (rows + 1 )  and pair[1] >= ( cols + 1 ) * (rows + 1 )  ):
                        cycle.remove(pair)
                        self.lattice_grid.remove_edge(pair[0],pair[1])  
                self.lattice_cycles.append(cycle)
        
        self.plaquettes = {}
        for cycle in self.lattice_cycles:
            P = Plaquette(cycle[0][0], cycle, self.lattice_grid.edge_subgraph(cycle))
            self.plaquettes[ cycle[0][0] ] = P
                   
    ## produce stars on lattice
        stars = [] 
        for label in self.lattice_grid.nodes():
            L = list(self.lattice_grid.neighbors(label))
            if len(L) > 1:
                L.append(label)
                star_subgraph = self.lattice_grid.subgraph(L)
                S = Star(label, star_subgraph)
                stars.append(S)            
        self.stars = dict(enumerate(stars ) )
        
        
    ######  initialize surface code on lattice  
        self.LatticeCircuit = self.initialize_LatticeCircuit()
        self.DataQubits = self.LatticeCircuit.qubits
        self.populate_plaquettes() 
        self.populate_stars()  
        
        self.generator_matrix = self.get_generator_matrix()
        self.X_graph = generate_tanner_graph( self.generator_matrix[0])
        self.Z_graph = generate_tanner_graph( self.generator_matrix[1])
        self.tanner_graphs = {'X': self.X_graph, 'Z': self.Z_graph}                      
        initializer = initialize_groundstate(self.generator_matrix)        
        self.LatticeCircuit.compose( initializer, inplace=True )
        
        self.Z_boundary_nodes = compute_boundary(self.Z_graph)   ##boundary nodes of the tanner graph
        self.X_boundary_nodes = compute_boundary(self.X_graph)
        # if initialize == 1:        
        #     self.LatticeCircuit.x([ self.LatticeCircuit.qubits[self.get_flat_index(edge)] for edge in self.boundary_edge])        

        if p_error:
    ###  apply random Pauli channel
            ApplyPauliError(self.LatticeCircuit, self.DataQubits, p_error, print_option=report_error)

    def get_generator_matrix(self):
        LatticeCircuit = self.LatticeCircuit
        ZBlock = sympy.Matrix()
        for plaquette in self.plaquettes.items():
            Zrow = sympy.zeros(1,self.num_of_qubits)
            for idx in [LatticeCircuit.qubits.index(qubit) for qubit in plaquette[1].qubits]:
                Zrow[0,idx] = 1
            ZBlock = ZBlock.col_join(Zrow)
            
        XBlock = sympy.Matrix()
        for star in self.stars.items():
            Xrow = sympy.zeros(1, self.num_of_qubits )
            for idx in [LatticeCircuit.qubits.index(qubit) for qubit in star[1].qubits]:
                Xrow[0,idx] = 1
            XBlock = XBlock.col_join(Xrow)
        return (XBlock,ZBlock)
            
    def marked_tanner_graph(self, label, marked_nodes):
        
        tanner_graph = self.tanner_graphs[label]
        marked_graph = nx.Graph()
        marked_graph.add_nodes_from(marked_nodes)
        edge_list = list(itertools.combinations(marked_nodes, 2))
        # marked_graph.add_edges_from( edge_list )
        
        pair_graph = marked_graph.copy()
        
        for edge in edge_list:
            path = nx.shortest_path(tanner_graph, edge[0], edge[1])
            marked_graph.add_edges_from( list(itertools.pairwise(path) ) )
            pair_graph.add_edge(edge[0],edge[1], weight = len(path) )
                
        # for edge in edge_list:
        #     path = [x for x in nx.shortest_path(tanner_graph, edge[0], edge[1]) if type(x) == int ]
        #     marked_graph.edges[edge[0],edge[1]]['weight'] = len(path)      
        
        
        
        # if boundary:
            
        boundary_nodes = []
            # active_boundary = []
            # available_boundary = list(set(boundary))            
        for node in marked_nodes:
            nearest_bdry = self.nearest_boundary(node, label)
            path = nx.shortest_path(tanner_graph, node, nearest_bdry)
            marked_graph.add_edges_from( list(itertools.pairwise(path)))
                
                # node_path = [x for x in path if type(x) == int ]
                # marked_graph.add_nodes_from(path)
                # marked_graph.add_edge( nearest_bdry, node, weight =   len(node_path) )
            boundary_node = 'b-' + node 
            boundary_nodes.append(boundary_node)
            marked_graph.add_edge( nearest_bdry, boundary_node, weight = 0 )
            pair_graph.add_edge( node, boundary_node, weight = len(path) )

                
            
            
            
            # pair_graph.add_edge( nearest_bdry, boundary_node, weight = 0 )
                
                
            #     active_boundary.append( nearest_bdry )
            #     path = nx.shortest_path(tanner_graph, node, nearest_bdry)
            #     # node_path = [x for x in path if type(x) == int]
            #     marked_graph.add_edge( nearest_bdry, node, weight = len(path) )

            #     available_boundary = list(  set(available_boundary) - set(active_boundary) )
                
            # active_boundary = []
            # available_boundary = list(set(boundary))
            # for node in marked_nodes:
            #     if available_boundary:
            #         dist, bdry_path = nx.multi_source_dijkstra(tanner_graph, available_boundary, node, weight='weight')
            #         marked_graph.add_edge( bdry_path[0], node , weight = dist)
            #         active_boundary.append(bdry_path[0])
            #         available_boundary = list(  set(available_boundary) - set(active_boundary) )
                
            
            boundary_edges_internal = list(itertools.combinations(boundary_nodes, 2))
            for edge in boundary_edges_internal:
                pair_graph.add_edge(edge[0], edge[1], weight = 0)
            
            # subgraph_nodes = marked_nodes.copy() 
            # subgraph_nodes.extend(boundary_nodes)
            
                
            
        
        return marked_graph, pair_graph
            
        
   

    def syndrome_cycle(self):      
    ##### syndrome measurements ##########
    ####################################


    ######### phase flips ###########
    
        star_checks = [ 'r' + str(idx) for idx in range(len(self.stars) ) ]
        syndrome_measurement(self.LatticeCircuit, self.X_graph, star_checks, 'X')


    # simulator = AerSimulator()
    # transpiled_circuit = transpile(LatticeCircuit, simulator)
    # job = simulator.run(transpiled_circuit, shots=1, memory=True)
    # transpiled_circuit = transpile(LatticeCircuit)
    
 
        job = AerSimulator().run(self.LatticeCircuit, shots=1, memory=True)   
    # job = AerSimulator().run(transpiled_circuit, shots=1, memory=True)
    
        result = job.result()
    # memory = result.get_memory(transpiled_circuit)
        memory = result.get_memory(self.LatticeCircuit)
        memory_result = memory[0][::-1].replace(' ','')[:len(self.stars)]
    

        positions = []
        for i in range(len(memory_result)):
            if memory_result[i] == '1':
                positions.append('r' + str(i) )
    
    # star_graph = lattice_grid.star_graph()
    # marked_star_graph = lattice_grid.star_graph( marked_stars=positions )     
           
        marked_star_graph,star_subgraph  = self.marked_tanner_graph('X', positions)
    
    ## address syndrome 
        star_matchings = nx.min_weight_matching(star_subgraph,  weight='weight')
        
        for match in star_matchings:
            if not ( (  match[0][0] == 'b' ) and (match[1][0] == 'b' ) ):
                path = [ node for node in nx.shortest_path(marked_star_graph, match[0], match[1] ) if type(node) == int  ]
                self.LatticeCircuit.z(path)




    ######### bit flips ###########
    
        plaquette_checks = [ 'r' + str(idx) for idx in range(len(self.plaquettes) ) ]
        syndrome_measurement(self.LatticeCircuit, self.Z_graph,plaquette_checks , 'Z')
    
    # transpiled_circuit = transpile(LatticeCircuit)
    # job = AerSimulator().run(transpiled_circuit, shots=1, memory=True)
    
        job = AerSimulator().run(self.LatticeCircuit, shots=1, memory=True)
        result = job.result()
    
    
    # memory = result.get_memory(transpiled_circuit)
        memory = result.get_memory(self.LatticeCircuit)
        memory_result = memory[0][::-1].replace(' ','')[-len(self.plaquettes):]


        positions = []
        for i in range(len(memory_result)):
            if memory_result[i] == '1':
                positions.append('r' + str(i) )
            
    
        marked_plaquette_graph, plaquette_pairing_graph = self.marked_tanner_graph('Z', positions)

    ## address syndrome 
        plaquette_matchings = nx.min_weight_matching( plaquette_pairing_graph,  weight='weight')

        for match in plaquette_matchings:
            if not ( (  match[0][0] == 'b' ) and (match[1][0] == 'b'  ) ):
                path = [ node for node in nx.shortest_path( marked_plaquette_graph , match[0], match[1] ) if type(node) == int  ]
                self.LatticeCircuit.x(path)
   
    def measure_data(self):
    # ##### final logical Z-parity measurements ####
    # ##############################################
        ZReadAncilla = AncillaRegister(1)
        ZReadout = ClassicalRegister(1)

        self.LatticeCircuit.add_register(ZReadAncilla)
        self.LatticeCircuit.add_register(ZReadout)

        self.LatticeCircuit.h(ZReadAncilla[0])
        boundary_edge = nx.shortest_path(self.lattice_grid, list(self.lattice_grid.nodes)[0]+self.rows, list(self.lattice_grid.nodes)[-1])

        for edge in itertools.pairwise(boundary_edge):    
            qubit = self.LatticeCircuit.qubits[list(self.edges).index(edge)]
            self.LatticeCircuit.cz(ZReadAncilla[0], qubit)
        
        self.LatticeCircuit.h(ZReadAncilla[0])  
    
        self.LatticeCircuit.measure(ZReadAncilla[0],ZReadout[0]) 
          

    def nearest_boundary(self, node, label):
        # return node closest to given node, on the tanner graph 
        # associated to the label (which is 'X' or 'Z')
        node_int = int(node.replace('r',''))
         
        if label == 'Z':
            left = (self.rows)*math.floor(node_int/self.rows)
            right = (self.rows)*math.ceil(node_int/self.rows) - 1
            if abs( node_int - left ) <= abs(node_int - right):
                peripheral = 'r' + str(left)
            else: 
                peripheral = 'r' + str(right) 
                
            return list(set(self.Z_boundary_nodes) & set(self.Z_graph.neighbors(peripheral)))[0]  
        
        if label == 'X':
            top = node_int - ((self.rows+1)* math.floor( node_int/(self.rows+1) ))
            bottom = top + (self.rows+1 )*(self.cols) 
            if abs( node_int - top ) <= abs(node_int - bottom):
                peripheral = 'r' + str(top) 
            else:
                peripheral = 'r' + str(bottom) 
            return list(set(self.X_boundary_nodes) & set(self.X_graph.neighbors(peripheral)))[0]
            
            
    def get_flat_index(self, edge):
        return list(self.edges).index(edge)
    
    def initialize_LatticeCircuit(self):
        LatticeCircuit = QuantumCircuit()
        for edge in self.edges:
            qubit = QuantumRegister(1, name=str(edge))
            LatticeCircuit.add_register(qubit) 
        return LatticeCircuit 
    
    
    def populate_plaquettes(self):
        LatticeCircuit = self.LatticeCircuit
        for plaquette in self.plaquettes.items():
            for edge in plaquette[1].edges:
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])       
                plaquette[1].qubits.append(LatticeCircuit.qubits[self.get_flat_index(edge)])
                
      
    def populate_stars(self):
        LatticeCircuit = self.LatticeCircuit
        for star in self.stars.items():
            for edge in star[1].edges:
                if edge[0] > edge[1]:
                    edge = (edge[1], edge[0])       
                star[1].qubits.append(LatticeCircuit.qubits[self.get_flat_index(edge)])                
                