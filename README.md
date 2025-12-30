## Project Overview: PINN vs. Data-Driven Optimization

Physics-Informed Neural Networks (PINNs) have emerged as a powerful tool for solving Partial Differential Equations (PDEs). However, unlike traditional supervised learning (Data-Driven) approaches, PINNs often suffer from "pathological" optimization behaviors.

### The Problem
The loss landscape of a PINN is determined by high-order derivatives of the network. This often results in a landscape that is highly non-convex, rough, and ill-conditioned, particularly as the frequency of the target solution increases.

### Objective
This repository investigates the spectral bias and optimization complexity of neural PDE solvers. We solve a multiscale Poisson equation using two distinct approaches:
1.  **PINN (Residual-based):** Training purely on the physics equation $-\Delta u = f$.
2.  **Data-Driven (Supervised):** Training directly on ground-truth solution data $(x, u)$.

The goal is to visualize and compare how the loss landscape degrades for both methods as the complexity (frequency $K$) of the solution increases.

## Problem Statement

We focus on a prototypical linear elliptic PDE, the **Poisson Equation**, defined on a 2D square domain $D = [0, 1]^2$[cite: 26]:

$$
\begin{aligned}
-\Delta u &= f, \quad \text{in } D, \\
u &= 0, \quad \text{on } \partial D.
\end{aligned}
$$

To control the complexity of the problem, we define a source term $f$ comprised of $K$ spatial scales. The source term and its corresponding analytical solution $u(x,y)$ are given by:

$$
f(x,y) = \frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^r \sin(\pi i x) \sin(\pi j y)
$$

$$
u(x,y) = \frac{1}{\pi K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{r-1} \sin(\pi i x) \sin(\pi j y)
$$

where $r = 0.5$ and coefficients $a_{ij} \sim \mathcal{N}(0,1)$[cite: 32]. The parameter $K$ acts as an indicator for the problem's complexity: as $K$ increases, the solution contains higher frequency components.

### Reference:

Krishnapriyan, A. S., Mudigonda, M., Karniadakis, G. E., & Prabhat, M. (2021). 
*Characterizing possible failure modes in physics-informed neural networks*. 
Advances in Neural Information Processing Systems, 34, 26548-26560. 
https://arxiv.org/abs/2109.01050

Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). 
*Visualizing the loss landscape of neural nets*. 
Advances in Neural Information Processing Systems, 31. 
https://arxiv.org/abs/1712.09913
