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
    
    
    
# def KitaevSurfaceCode( x_0, k0, k1 , p_error, error=True, report_error=False):

#  #### initialize lattice data ###
#  ##############################
#     rows = k0
#     cols = k1 
    
#     LatticeGrid = Lattice(k0,k1)
#     LatticeCircuit = LatticeGrid.initialize_LatticeCircuit()
    
#     DataQubits = LatticeCircuit.qubits
#     LatticeGrid.populate_plaquettes(LatticeCircuit) 
#     LatticeGrid.populate_stars(LatticeCircuit)       
#     # LatticeGrid.populate_boundary_plaquettes(LatticeCircuit)
#     generator_matrix = LatticeGrid.generator_matrix(LatticeCircuit)
#     X_graph = generate_tanner_graph( generator_matrix[0])
#     Z_graph = generate_tanner_graph( generator_matrix[1])      
#     # initialize_code(generator_matrix, LatticeCircuit, return_generators = False)
#     initializer = initialize_groundstate(generator_matrix)
#     initializer_dg = initializer.inverse()
    
#     LatticeCircuit.compose( initializer, inplace=True )
#     # Circuit_inverse = LatticeCircuit.inverse() 
    
    
#     boundary_edge = list( nx.utils.pairwise( [ x for x in range(LatticeGrid.rows + 1 ) ] ) ) 
#     boundary_nodes = compute_boundary(Z_graph)
    
# ##### prepare initial state ####
# ################################
#     # LatticeCircuit.compose(PrepareGroundState(ToricLattice),qubits = DataQubits ,inplace = True)
#     if x_0 == 1:        
#         LatticeCircuit.x([ LatticeCircuit.qubits[LatticeGrid.get_flat_index(edge)] for edge in boundary_edge])
        
        
        
        
#     # if x_1 == 1:
#     #     LatticeCircuit.compose( LogicalX1_circuit(ToricLattice),  qubits = DataQubits, inplace = True )
        
#     if error == True:
#     ###  apply random Pauli channel
#         ApplyPauliError(LatticeCircuit, DataQubits, p_error, print_option=report_error)
        
# ##### syndrome measurements ##########
# ####################################


#     ######### phase flips ###########
    
#     star_checks = [ 'r' + str(idx) for idx in range(len(LatticeGrid.stars) ) ]
#     syndrome_measurement(LatticeCircuit, X_graph, star_checks, 'X')


#     # simulator = AerSimulator()
#     # transpiled_circuit = transpile(LatticeCircuit, simulator)
#     # job = simulator.run(transpiled_circuit, shots=1, memory=True)
#     # transpiled_circuit = transpile(LatticeCircuit)
    
 
#     job = AerSimulator().run(LatticeCircuit, shots=1, memory=True)   
#     # job = AerSimulator().run(transpiled_circuit, shots=1, memory=True)
    
#     result = job.result()
#     # memory = result.get_memory(transpiled_circuit)
#     memory = result.get_memory(LatticeCircuit)
#     memory_result = memory[0][::-1].replace(' ','')[:len(LatticeGrid.stars)]
    

#     positions = []
#     for i in range(len(memory_result)):
#         if memory_result[i] == '1':
#             positions.append('r' + str(i) )
    
#     # star_graph = LatticeGrid.star_graph()
#     # marked_star_graph = LatticeGrid.star_graph( marked_stars=positions )     
           
#     marked_star_graph = LatticeGrid.marked_tanner_graph(X_graph, positions)
    
#     ## address syndrome 
#     star_matchings = nx.min_weight_matching(marked_star_graph,  weight='weight')
    
#     for match in star_matchings:
#         LatticeCircuit.z([ node for node in nx.shortest_path( X_graph, match[0], match[1] ) if type(node) == int  ])


#     ######### bit flips ###########
    
#     plaquette_checks = [ 'r' + str(idx) for idx in range(len(LatticeGrid.plaquettes) ) ]
#     syndrome_measurement(LatticeCircuit, Z_graph,plaquette_checks , 'Z')
    
#     # transpiled_circuit = transpile(LatticeCircuit)
#     # job = AerSimulator().run(transpiled_circuit, shots=1, memory=True)
    
#     job = AerSimulator().run(LatticeCircuit, shots=1, memory=True)
#     result = job.result()
    
    
#     # memory = result.get_memory(transpiled_circuit)
#     memory = result.get_memory(LatticeCircuit)
#     memory_result = memory[0][::-1].replace(' ','')[-len(LatticeGrid.plaquettes):]


#     positions = []
#     for i in range(len(memory_result)):
#         if memory_result[i] == '1':
#             positions.append('r' + str(i) )
            
#     # boundary = [idx)  for idx in boundary_nodes ]
    
#     marked_plaquette_graph = LatticeGrid.marked_tanner_graph(Z_graph, positions, boundary_nodes)

#     ## address syndrome 
#     plaquette_matchings = nx.min_weight_matching(marked_plaquette_graph,  weight='weight')

#     for match in plaquette_matchings:
#         if not ( (  match[0] in boundary_nodes) and (match[1] in boundary_nodes ) ):
#             path = [ node for node in nx.shortest_path( Z_graph, match[0], match[1] ) if type(node) == int  ]
#             LatticeCircuit.x(path)
#     # positions.append('bdry')
    
#     # marked_plaquette_graph = LatticeGrid.marked_tanner_graph(Z_graph, positions)

#     # ## address syndrome 
#     # plaquette_matchings = nx.min_weight_matching(marked_plaquette_graph,  weight='weight')            
    
#     # for match in plaquette_matchings:
#     #     LatticeCircuit.x([ node for node in nx.shortest_path( Z_graph, match[0], match[1] ) if type(node) == int  ])
 
   
#     # ##### final logical Z-parity measurements ####
#     # ##############################################
#     ZReadAncilla = AncillaRegister(1)
#     ZReadout = ClassicalRegister(1)

#     LatticeCircuit.add_register(ZReadAncilla)
#     LatticeCircuit.add_register(ZReadout)

#     LatticeCircuit.h(ZReadAncilla[0])
#     boundary_edge = nx.shortest_path(LatticeGrid.lattice_grid, list(LatticeGrid.lattice_grid.nodes)[0]+rows, list(LatticeGrid.lattice_grid.nodes)[-1])

#     for edge in itertools.pairwise(boundary_edge):    
#         qubit = LatticeCircuit.qubits[list(LatticeGrid.lattice_grid.edges).index(edge)]
#         LatticeCircuit.cz(ZReadAncilla[0], qubit)
        
#     LatticeCircuit.h(ZReadAncilla[0])  
    
#     LatticeCircuit.measure(ZReadAncilla[0],ZReadout[0]) 
          
#     # LatticeCircuit.compose( initializer_dg ,  inplace=True )

#     # classicalMeasure = ClassicalRegister( len(DataQubits))
#     # LatticeCircuit.add_register(classicalMeasure)
#     # LatticeCircuit.measure(DataQubits, classicalMeasure) 

            
#     return LatticeCircuit



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
            
    def marked_tanner_graph(self, tanner_graph, marked_nodes, boundary=[]):
        marked_graph = nx.Graph()
    
        marked_graph.add_nodes_from(marked_nodes)
        edge_list = list(itertools.combinations(marked_nodes, 2))
        marked_graph.add_edges_from( edge_list )
        

        for edge in edge_list:
            path = [x for x in nx.shortest_path(tanner_graph, edge[0], edge[1]) if type(x) == int ]
            marked_graph.edges[edge[0],edge[1]]['weight'] = len(path)      
        
       
        if boundary:
            active_boundary = []
            available_boundary = list(set(boundary))
            for node in marked_nodes:
                if available_boundary:
                    dist, bdry_path = nx.multi_source_dijkstra(tanner_graph, available_boundary, node, weight='weight')
                    marked_graph.add_edge( bdry_path[0], node , weight = dist)
                    active_boundary.append(bdry_path[0])
                    available_boundary = list(  set(available_boundary) - set(active_boundary) )
                
                
            boundary_edges_internal = list(itertools.combinations(active_boundary, 2))
            for edge in boundary_edges_internal:
                marked_graph.add_edge(edge[0], edge[1], weight = 0) 
                
            
        
        return marked_graph
            
        
   

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
           
        marked_star_graph = self.marked_tanner_graph(self.X_graph, positions, self.X_boundary_nodes)
    
    ## address syndrome 
        star_matchings = nx.min_weight_matching(marked_star_graph,  weight='weight')
        
        for match in star_matchings:
            if not ( (  match[0] in self.X_boundary_nodes) and (match[1] in self.X_boundary_nodes ) ):
                path = [ node for node in nx.shortest_path( self.X_graph, match[0], match[1] ) if type(node) == int  ]
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
            
    
        marked_plaquette_graph = self.marked_tanner_graph(self.Z_graph, positions, self.Z_boundary_nodes)

    ## address syndrome 
        plaquette_matchings = nx.min_weight_matching(marked_plaquette_graph,  weight='weight')

        for match in plaquette_matchings:
            if not ( (  match[0] in self.Z_boundary_nodes) and (match[1] in self.Z_boundary_nodes ) ):
                path = [ node for node in nx.shortest_path( self.Z_graph, match[0], match[1] ) if type(node) == int  ]
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
                