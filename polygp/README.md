# PolyGP

**UNDER CONSTRUCTION**

**NOT RELEASED AS PACKAGE YET**

This library of functions builds an optimized polynomial with its base elements belonging to the list of $M$ kernel functions selected from a list of $N$ available base kernel functions. The kernels are based on the selection available from the _scikit-learn_ package at [Gaussian process kernels](https://scikit-learn.org/stable/modules/gaussian_process.html) and the optimization package is [_optuna_](https://optuna.readthedocs.io/en/stable/index.html).

The generic polynomial that can be built is 

$P(\mathbf{x}) = \epsilon (x) + \sum_{K_1, K_2, \ldots , K_M\in\{K_1,\dots, K_N\}} a_{K_1, K_2, \ldots, K_M} K_1(x) K_2(x) \ldots K_M(x)$,

where $a_{K_1, K_2, \ldots, K_M}$ represents the ConstantKernel and $E(x)$ represents the noise kernel function.
The optimal kernel is a subset of this generic polynomial and maximizes the log-likelihood of the Gaussian process regression model that uses it to fit the given training data. For more information, the user can look at the notebooks in this [repository](https://github.com/alessiot/polygp-sklearn) or this [article](_posts/01_01_2024_polygp_sklearn.html) 

Notes: https://packaging.python.org/en/latest/tutorials/packaging-projects/
