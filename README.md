# Diffusion-Limited Lipid Droplet Growth

This repository contains a numerical simulation of diffusion-limited lipid droplet growth.

The model describes lipid transport in the endoplasmic reticulum (ER) membrane and the resulting growth of lipid droplets attached to the membrane.

The simulation combines:

• diffusion PDE modeling  
• finite element discretization  
• numerical time stepping  
• scientific visualization  

---

## Mathematical Model

Lipid concentration evolves according to the diffusion equation

∂c/∂t = D ∇²c  +Q

with absorbing boundary conditions at the droplet surface.

The droplet radius evolves according to the diffusive flux into the droplet.

---

## Methods

The simulation uses:

• Python  
• NumPy / SciPy  
• finite element mesh generation  
• numerical PDE solvers  

---

##  Output

• concentration fields around droplets  
• droplet growth curves  
• comparison with theoretical scaling laws  

---

## Repository Structure

src/        simulation code  
notebooks/  analysis notebooks  
figures/    simulation results  

---

## Author

Emmanuel Nworie  
PhD Candidate — Computational & Applied Mathematics  
Southern Methodist University.
