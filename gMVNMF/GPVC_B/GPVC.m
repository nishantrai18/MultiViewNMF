function [Ux,Uy,centroidPc,P1,P2,objValue] = GPVC(X2,Y2,X1,Y3,W1,W2,options)
%      partial view multi-view data set :   X2: n2*dx  Y2: n2*dy  X1: n1*dx  Y3: n3*dy
%      (X2,Y2); examples appearing in both views; 
%      (X1,);   examples appearing in only view x
%      (,Y3):   examples appearing in only view y  
%      W1 : Weight matrix for the affinity graph of View 1
%      W2 : Weight matrix for the affinity graph of View 2
%      dx,dy:   feature dimension of viewx viewy
%      option.alpha: parameter controls the importance of views
%      option.latentdim: the feature dimension of latent space(default value: cluster number)
%   ouptut: 
%     Ux, Uy:   (k*dx, k*dy)  basis of latent space
%     P2,P1,P3: (n2*k, n1*k, n3*k)  data representations for examples in the latent space 
%     centroidP2 : Refers to the consensus matrix for the complete datapoints
%	Writen by Jialu Liu (jliu64@illinois.edu)
% 	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)
%   Further modified by Nishant Rai (nishantr AT iitk DOT ac DOT in)

%Note that columns are data vectors here
%Number of views are assumed to be 2 in this code

tic;
Rounds = options.rounds;
alpha=options.alpha;
beta=options.beta;
error=options.error;

K = options.K;
A1 = horzcat(X1,X2);
A2 = horzcat(Y2,Y3);

nSmp = size(W1,1);
Wtemp = beta*alpha*W1;            %Modify the weight matrix with the involved parameters
DCol = full(sum(Wtemp,2));
D = spdiags(DCol,0,nSmp,nSmp);
L1 = D - Wtemp;                              %Get matrix L

nSmp = size(W2,1);
Wtemp = beta*alpha*W1;            %Modify the weight matrix with the involved parameters
DCol = full(sum(Wtemp,2));
D = spdiags(DCol,0,nSmp,nSmp);
L2 = D - Wtemp;                              %Get matrix L
    
[numCom,~]=size(X2');

objValue = [];

%% initialize basis and coefficient matrices, initialize on the basis of standard GNMF algorithm
tic;
j = j + 1;
Goptions.alpha=options.Gaplpha;
rand('twister',5489);
[Ux, P1] = GNMF(A1, K, W1, options);        %In this case, random inits take place
rand('twister',5489);
printResult(P1, label, options.K, options.kmeans);        
rand('twister',5489);
[Uy, P2] = GNMF(A2, K, W2, Goptions);
rand('twister',5489);
printResult(P2, label, options.K, options.kmeans);

toc;
%%

%% Alternate Optimisations for consensus matrix and individual view matrices
optionsPGNMF = options;
j = 0;
sumRound=0;
while j < Rounds                            %Number of rounds of AO
    sumRound=sumRound+1;
    j = j + 1;
    if j==1
        centroidPc = P1(1:numCom,:);                       %Basic initialization for consensus matrix
    else
        centroidPc = (alpha*P1(1:numCom,:)) + (alpha*P2((numCom+1):end,:));           %From the paper, we have a definite solution for V*
        centroidPc = centroidPc / sum(options.alphas);
    end
    logL = 0;                                   %Loss for the round
    
    %Compute the losses
    tmp1 = (A1 - Ux*P1');
    tmp2 = (P1 - centroidPc);
    logL = logL + sum(sum(tmp1.^2)) + alpha* (sum(sum(tmp2.^2)))+sum(sum((P1'*L1).*P1'));
    tmp1 = (A2 - Uy*P2');
    tmp2 = (P2 - centroidPc);
    logL = logL + sum(sum(tmp1.^2)) + alpha* (sum(sum(tmp2.^2)))+sum(sum((P2'*L2).*P2'));
    
    %logL
    objValue = [objValue logL];                %End indicates last index of array, so basically push operation
    
    if mod(j,10)==0
    fprintf('Iteration %d, objective value %g\n', j, objValue(j));
    end

    if j>1 && ((abs(objValue(j)-objValue(j-1))/objValue(j) < error)|| objValue(j)<=error)
        fprintf('Objective value converge to %g at iteration %d before the maxIteration reached \n',objValue(j),j);
        break;
    end

    if sumRound==30
        break;
    end
    
    optionsPGNMF.alpha = alpha;
    rand('twister',5489);
    [Ux, P1] = PartialGNMF(A1, K, centroidPc, W1, optionsPGNMF, Ux, P1);
    [Uy, P2] = PartialGNMF(A2, K, centroidPc, W2, optionsPGNMF, Uy, P2);
    %Peform optimization with Pc* (centroidPc) fixed and inits finalU, finalV

end
toc