function [finalU, finalV, finalcentroidV, log] = GMultiNMF(X, K, W, label,options)
%	Notation:
% 	X ... a cell containing all views for the data
% 	K ... number of hidden factors
% 	W ... weight matrix of the affinity graph 
% 	label ... ground truth labels

%	Writen by Jialu Liu (jliu64@illinois.edu)
% 	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)

%	References:
% 	J. Liu, C.Wang, J. Gao, and J. Han, ��Multi-view clustering via joint nonnegative matrix factorization,�� in Proc. SDM, Austin, Texas, May 2013, vol. 13, pp. 252�C260.
% 	Zhenfan Wang, Xiangwei Kong, Haiyan Fu, Ming Li, Yujia Zhang, FEATURE EXTRACTION VIA MULTI-VIEW NON-NEGATIVE MATRIX FACTORIZATION WITH LOCAL GRAPH REGULARIZATION, ICIP 2015.

%Note that columns are data vectors here

tic;
viewNum = length(X);
Rounds = options.rounds;
nSmp=size(X{1},2);                          %Number of data points

U = cell(1, viewNum);
V = cell(1, viewNum);

j = 2;
log = 0;
ac=0;

%% initialize basis and coefficient matrices, initialize on the basis of
% standard GNMF algorithm
tic;
while j < 3                   %Modify this loop for managing initializations
                               %Analyse which 'options' give better results
    j = j + 1;
    workspace
    Goptions.alpha=options.Gaplpha;
    if j == 1
        
        %[U{1}, V{1}] = NMF1(X{1}, K,  options, U_, V_);
        rand('twister',5489);
        [U{1}, V{1}] = GNMF(X{1}, K, W{1}, Goptions);                       %In this case, random inits take place
        rand('twister',5489);
        printResult(V{1}, label, options.K, options.kmeans);
    else
        %[U{1}, V{1}] = NMF1(X{1}, K, options, U_, V{viewNum});
        rand('twister',5489);
        [U{1}, V{1}] = GNMF(X{1}, K, W{1}, options);        %In this case, random inits take place
        rand('twister',5489);
        printResult(V{1}, label, options.K, options.kmeans);        
    end
    for i = 2:viewNum
    %         [U{i}, V{i}] = NMF1(X{i}, K, options, U_, V{i-1});
        rand('twister',5489);
        [U{i}, V{i}] = GNMF(X{i}, K, W{i}, Goptions);
        rand('twister',5489);
        printResult(V{i}, label, options.K, options.kmeans);
    end
end
toc;
%%

%%Alternate Optimisations for consensus matrix and individual view matrices
optionsForPerViewNMF = options;
oldac=0;                        %Old accuracy
maxac=0;                        %Maximum accuracy
j = 0;
sumRound=0;
while j < Rounds                            %Number of rounds of AO
    sumRound=sumRound+1;
    j = j + 1;
    if j==1
        centroidV = V{1};                       %Basic initialization for consensus matrix
    else
        centroidV = options.alphas(1) * V{1};           %From the paper, we have a definite solution for V*
        for i = 2:viewNum                               %CHECK why have we not considered Q in the normalization
                                                        %Already normalized during the optimizations
            centroidV = centroidV + options.alphas(i) * V{i};
        end
        centroidV = centroidV / sum(options.alphas);
    end
    logL = 0;                                   %Loss for the round
    for i = 1:viewNum
        alpha=options.alphas(i);
        if alpha > 0
            Wtemp = options.beta*alpha*W{i};            %Modify the weight matrix with the involved parameters
            DCol = full(sum(Wtemp,2));
            D = spdiags(DCol,0,nSmp,nSmp);
            L = D - Wtemp;                              %Get matrix L
            if isfield(options,'NormW') && options.NormW
                D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
                L = D_mhalf*L*D_mhalf;
            end
        else
            L = [];
        end
        %Compute the losses
        tmp1 = (X{i} - U{i}*V{i}');
        tmp2 = (V{i} - centroidV);
        logL = logL + sum(sum(tmp1.^2)) + alpha* (sum(sum(tmp2.^2)))+sum(sum((V{i}'*L).*V{i}'));  %�޸ģ�����SampleW��V'*L*V
    end
    
    %logL
    log(end+1)=logL;                %End indicates last index of array, so basically push operation
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
    
    %Find the Best combination for U, V based on accuracy
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
    
    if sumRound==30
        break;
    end
    
    for i = 1:viewNum
        optionsForPerViewNMF.alpha = options.alphas(i);
        rand('twister',5489);
        [U{i}, V{i}] = PerViewNMF(X{i}, K, centroidV, W{i} , optionsForPerViewNMF, finalU{i}, finalV{i});
        %Peform optimization with V* (centroidV) fixed and inits finalU, finalV
    end

end
toc