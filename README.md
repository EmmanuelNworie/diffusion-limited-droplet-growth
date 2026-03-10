# Diffusion-Limited Droplet Growth

This repository contains a scientific computing project on diffusion-limited lipid droplet growth.

The goal is to model how lipid concentration evolves in time and how a droplet grows by absorbing material from its surrounding environment. The project is motivated by transport processes in biological systems, especially lipid droplet growth associated with the endoplasmic reticulum.

## Project Overview

This project demonstrates:

- diffusion-based mathematical modeling
- numerical simulation in Python
- scientific visualization
- reproducible computational workflow

## Mathematical Idea

A concentration field \( c(x,y,t) \) evolves by diffusion and is depleted at the droplet boundary. A simplified governing model is

\[
\frac{\partial c}{\partial t} = D \nabla^2 c
\]

where:

- \( c(x,y,t) \) is the concentration
- \( D \) is the diffusion coefficient

The droplet radius changes in time according to the flux absorbed at the droplet interface.

## Repository Structure

- `src/` — simulation scripts
- `notebooks/` — exploratory analysis and visualization
- `figures/` — output plots
- `data/` — input or processed data files

## Current Status

This repository is being organized into a clean public portfolio version of a broader research workflow in scientific computing and diffusion-driven transport modeling.

## Tools Used

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

## Author

**Emmanuel Nworie**  
PhD Candidate, Computational & Applied Mathematics  
Southern Methodist University
