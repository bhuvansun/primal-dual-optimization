The folder contains 6 MATLAB files. 'main.m' is the file with the code execution, while the other 5 files are user-defined function files.

In the main file, the first section, i.e., Chambolle and Pock's Primal-Dual Algorithm is the required algorithm that was presented in the 
paper given to us: 'An efficient primal-dual method for solving non-smooth machine learning problem'.

In the sections that follow, we implemented two algorithms, 1) Gradient Descent, 2) Adam Optimization, and compared these with the main 
Primal-Dual Algorithm using |x| as an example loss function.

Though GD and ADAM were much faster than Primal-Dual, with GD taking 0.02 seconds, ADAM taking 0.03 seconds and Primal-Dual taking 0.82 
seconds approximately for computation of smooth approximation, it is to be noted that Primal-Dual produced a more convincing approximation 
curve compared to the other two, while also maintaining the loss function error lesser than the two. The results have been visualized with 
graphs.

'kernelGenerator.m' and 'primal-dual.m' are the files used for the Primal-Dual Algorithm.
'gradient_descent.m' is the file used for the Gradient Descent Algorithm.
'adam_optimizaton.m' and 'nonSmoothApproximation.m' are the files used for the Adam Optimization Algorithm.