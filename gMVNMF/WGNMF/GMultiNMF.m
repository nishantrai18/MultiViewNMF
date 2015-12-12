function [finalU, finalV, finalcentroidV, weights, log] = GMultiNMF(X, K, W, label,options)
%	Notation:
% 	X ... a cell containing all views for the data
% 	K ... number of hidden factors
% 	W ... weight matrix of the affinity graph 
% 	label ... ground truth labels

%	Writen by Jialu Liu (jliu64@illinois.edu)
% 	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)
%   Further modified by Nishant Rai (nishantr AT iitk DOT ac.in)

%	References:
% 	J. Liu, C.Wang, J. Gao, and J. Han, ��Multi-view clustering via joint nonnegative matrix factorization,�� in Proc. SDM, Austin, Texas, May 2013, vol. 13, pp. 252�C260.
% 	Zhenfan Wang, Xiangwei Kong, Haiyan Fu, Ming Li, Yujia Zhang, FEATURE EXTRACTION VIA MULTI-VIEW NON-NEGATIVE MATRIX FACTORIZATION WITH LOCAL GRAPH REGULARIZATION, ICIP 2015.

% Weights represent the weights for each view (They are common for both the
% divergence with consensus and also the graph laplacian
% options.gamma represents the parameter for handling the weights
% options.beta is or handling the graph regularization term
% options.varWeight is 1 if we want to weights to be varied, otherise it's 0

tic;
viewNum = length(X);
Rounds = options.rounds;
gamma = options.gamma;
beta = options.beta;
nSmp=size(X{1},2);

U = cell(1, viewNum);
V = cell(1, viewNum);
L = cell(1,viewNum);

log = 0;
ac=0;

%% Compute the graph laplacians
for i = 1:viewNum
    if beta > 0
        Wtemp = beta*W{i};
        DCol = full(sum(Wtemp,2));
        D = spdiags(DCol,0,nSmp,nSmp);
        L{i} = D - Wtemp;
    else
        L{i} = [];
    end
end
%%

%% Initiliaze weights
weights(1:viewNum) = (1/viewNum);
%%

%% initialize basis and coefficient matrices, initialize on the basis of standard GNMF algorithm
tic;
Goptions.alpha=options.Gaplpha;
rand('twister',5489);
[U{1}, V{1}] = GNMF(X{1}, K, W{1}, options);        %In this case, random inits take place
rand('twister',5489);
%printResult(V{1}, label, options.K, options.kmeans);        
for i = 2:viewNum
    rand('twister',5489);
    [U{i}, V{i}] = GNMF(X{i}, K, W{i}, Goptions);
    rand('twister',5489);
    %printResult(V{i}, label, options.K, options.kmeans);
end
toc;
%%

optionsForPerViewNMF = options;
oldac=0;
maxac=0;
j = 0;
sumRound=0;
while j < Rounds
    sumRound=sumRound+1;
    j = j + 1;

    %Update centroid V
    centroidV = (weights(1)^gamma)*V{1};
    for i = 2:viewNum
        centroidV = centroidV + (weights(i)^gamma)*V{i};
    end
    centroidV = centroidV / sum(weights.^gamma);            %Check if the array is modified or not
    
    %Update the weights if the corresponding option is set
    if (options.varWeight > 0)
        H = [];
        for i = 1:viewNum
            tmp1 = (X{i} - U{i}*V{i}');
            tmp2 = (V{i} - centroidV);
            val = gamma*(sum(sum(tmp1.^2))+(sum(sum(tmp2.^2)))+sum(sum((V{i}'*L{i}).*V{i}')));  
            val = val^(1.0/(1-gamma));
            H(end+1) = val;
        end
        tmpSum = sum(H);
        weights = (H./tmpSum);    
    end
    
    for i=1:viewNum
        fprintf('%.3f ',weights(i));
    end
    fprintf(' comp %d\n',j);

    %Compute loss
    logL = 0;
    for i = 1:viewNum
        alpha = weights(i)^gamma;
        tmp1 = (X{i} - U{i}*V{i}');
        tmp2 = (V{i} - centroidV);
        logL = logL + alpha*(sum(sum(tmp1.^2)) + (sum(sum(tmp2.^2)))+sum(sum((V{i}'*L{i}).*V{i}')));  %�޸ģ�����SampleW��V'*L*V
    end
    
    %logL
    log(end+1)=logL;
    rand('twister',5489);
    ac = printResult(centroidV, label, options.K, options.kmeans);
    if ac>oldac
        tempac=ac;
        tempU=U;
        tempV=V;
        tempcentroidV=centroidV;

    elseif oldac>maxac
        maxac=oldac;
        maxU=tempU;
        maxV=tempV;
        maxcentroidV=tempcentroidV;
    end
    oldac=ac;
    if(tempac>maxac)
        finalU=tempU;
        finalV=tempV;
        finalcentroidV=tempcentroidV;

    else
        finalU=maxU;
        finalV=maxV;
        finalcentroidV=maxcentroidV;
    end
    
    if sumRound==Rounds
        break;
    end
    
    %Update the individual view, the weights do not have any role here
    for i = 1:viewNum
        optionsForPerViewNMF.alpha = options.Gaplpha;
        rand('twister',5489);
        [U{i}, V{i}] = PerViewNMF(X{i}, K, centroidV, W{i}, optionsForPerViewNMF, finalU{i}, finalV{i}); 
    end
    
end
toc