# Kitaev surface codes 

(...and other lattice based codes with MWPM-type decoders)



This repository contains a Qiskit implementation of Kitaev surface codes, a topological quantum CSS code family based on planar lattices. It shares much in common with the toric code, but the surface code is more general, flexible and its planar arrangement is easier to implement in practise. A Qiskit implementation of the toric code can be found 
[here]( https://github.com/elhyc/Kitaev-Toric-Code ).


While an implementation of the surface code is the primary motivation for the code contained in this repository, the code can be used in more general contexts. This implementation may be used to implement *any* quantum CSS code with a MWPM (minimum-weight-perfect-matching) syndrome decoding procedure. More specifically, quantum CSS codes with MWPM syndrome decoding procedures are codes for which syndrome measurements either come in pairs or in isolation.