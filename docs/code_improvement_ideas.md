# Performance improvement
- [ ] Store Hamiltonian in forward order, density in reverse order and overlap in forward order
    - [ ] That way the trace-matmuls are automatically aligned
    - [ ] Investigate whether this makes sense first

# Code organisation
- [ ] IrrepToMatrix could only have static methods, since the q matrices are stored elsewhere anyways
- [ ] BlockMatrix diag() and offdiag() should return a BlockMatrix
- [ ] Same for IrrepsBlockData
- [ ] Put the k-space functions into BlockMatrix
- [ ] Rename Factory.add_snapshot() to Factory.add_openmx_snapshot()
- [ ] Combine datasets (train, val, [test]) and mapper into one thing (DataModule)
