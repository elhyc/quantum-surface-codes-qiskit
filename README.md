# Kitaev surface codes 

(...and other lattice based codes with MWPM-type decoders)



This repository contains a Qiskit implementation of Kitaev surface codes, a topological quantum CSS code family based on planar lattices. It shares much in common with the toric code, but the surface code is more general, flexible and its planar arrangement is easier to implement in practise. A Qiskit implementation of the toric code can be found 
[here]( https://github.com/elhyc/Kitaev-Toric-Code ).


While an implementation of the surface code is the primary motivation of this repository, the implementation can be used in more general contexts. This implementation may be used to implement *any* quantum CSS code with a MWPM (minimum-weight-perfect-matching) syndrome decoding procedure -- that is, a quantum CSS code for which syndrome measurements either come in pairs or in isolation.



## Overview


The basic premise of the planar surface code is similar to the idea behind the toric code -- [please see the description here for details](https://github.com/elhyc/Kitaev-Toric-Code). Qubits are placed on the edges of a $k_{0} \times k_{1}$ lattice grid, but unlike the toric code the lattice boundary is not considered to be periodic, so that one works with a planar lattice instead (so that the lattice topologically represents a contractible disc with boundary). Below is a figure of such a planar $5 \times 5$ lattice:


<p align="center">
<img src="./figures/lattice5x5planar.png" alt="example lattice" width="500"/>
</p>


The planar lattices have two kinds of boundary edges: edges that make up the "smooth boundary" and edges that make up the "rough boundary". In the example above, the edges making up the "smooth boundary" are the edges $(0,6), (6,12), (12,18), (18,24), (24,30)$;
$(5,11), (11,17), (17,23), (23,29), (29,35)$. The edges that make up the "rough boundary" are the edges $(0,-6), (1,-5), (2,-4), (3,-3), (4,-2), (5,-1)$ ; $(35,41), (34,40), (33,39), (32,38), (31,37), (30,36)$. 

The qubits on the "smooth boundary edges" belong to a single plaquette (and two stars), while the qubits on the "rough boundary edges" belong to a single star (and two plaquettes).
In the case of the toric code, each edge of the lattice belongs to two plaquettes and two stars -- therefore, a $X$ or $Z$ flip on a qubit (on any edge) will always be detected by syndrome measurements for pairs of plaquette or star operators. In the planar case with boundary, a $X$ or $Z$ flip on a qubit belonging to a boundary edge will only be detected by a syndrome measurement for a single plaquette or star in isolation.


## Implementation details

Apart from Qiskit, the [main code](./src/KitaevSurfaceCode.py) uses [NetworkX](https://networkx.org/) to implement the necessary data structures required for surface codes. 
Unlike the implementation found [here](https://github.com/elhyc/Kitaev-Toric-Code), this implementation uses more general algorithms available for CSS codes to: 1) produce logical states, 2) construct implement plaquette and star lattices (and their associated operators) 




