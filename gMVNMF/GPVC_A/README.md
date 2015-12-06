Aim: Incorporate Partial Multi View Clustering with Graph Regularization

Structure of the code:

main*.m : The driver program which reads datasets to create variables and calls GPVCClust to compute the scores given the required inputs.
GPVCclust.m : Calls GPVC.m to get the final results, then computes scores using some supporting programs present in partialMV folder.
UpdatePc.m : Updates Pc according to the update rules derived.
GPVC.m : Declares a function with,
		- Input : [Xc, Yc, X, Y, W1, W2, options/parameters]	(W's are the weight matrix)
		- Ouptut : [U1, U2, Vc, V1, V2, objectiveValue]
	    Steps involved,
		- Initialize U1, U2, P1, P2 with GNMF() declared in GNMF folder
		- Initialize Pc with appropriate formula (To be decided)
		- Repeat the following,
			- Update U's, P1, P2 fixing Pc using the multiplicative updates (Or using PerViewNMF())
			- Update Pc using the formula (To be decided)
		- Normalise U's and V's at the end (Or during it (Depends))
