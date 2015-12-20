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
gamma = options.gamma;

K = options.K;
A1 = horzcat(X1,X2);
A2 = horzcat(Y2,Y3);

[numInst1,Featx]=size(A1');                             %Number of instances, FeatX: Number of features in view 1
[numInst2,Featy]=size(A2');                             %Number of instances with view 2

P1=rand(numInst1,K);                                %Random initialization
P2=rand(numInst2,K);
Ux=rand(Featx,K);
Uy=rand(Featy,K);

nSmp = size(W1,1);
Wtemp = W1;            %Modify the weight matrix with the involved parameters
DCol = full(sum(Wtemp,2));
D = spdiags(DCol,0,nSmp,nSmp);
L1 = D - Wtemp;                              %Get matrix L

nSmp = size(W2,1);
Wtemp = W2;            %Modify the weight matrix with the involved parameters
DCol = full(sum(Wtemp,2));
D = spdiags(DCol,0,nSmp,nSmp);
L2 = D - Wtemp;                              %Get matrix L

singX=size(X1',1);
numCom=size(X2',1);

objValue = [];

weights = [0.5 0.5];

%% initialize basis and coefficient matrices, initialize on the basis of standard GNMF algorithm
tic;
Goptions.alpha = options.alpha*options.beta;
rand('twister',5489);
[Ux, P1] = GNMF(A1, K, W1, Goptions);        %In this case, random inits take place
rand('twister',5489);
[Uy, P2] = GNMF(A2, K, W2, Goptions);
toc;
%%

[Ux, P1] = Normalize(Ux, P1);
[Uy, P2] = Normalize(Uy, P2);

%% Alternate Optimisations for consensus matrix and individual view matrices
optionsPGNMF = options;
j = 0;
sumRound=0;
while j < Rounds                            %Number of rounds of AO
    sumRound=sumRound+1;
    j = j + 1;
    
    centroidPc = (weights(1)^gamma)*(options.alphas(1)*P1(singX+1:end,:)) + (weights(2)^gamma)*(options.alphas(2)*P2(1:numCom,:));          
    %From the paper, we have a definite solution for Pc*
    tmpSum = (weights(1)^gamma)*options.alphas(1) + (weights(2)^gamma)*(options.alphas(2));
    centroidPc = centroidPc / tmpSum;
        
    %Update the weights if the corresponding option is set
    if (options.varWeight > 0)
        H = [];
        tmp1 = (A1 - Ux*P1');
        tmp2 = (P1(singX+1:end,:) - centroidPc);
        H1 = gamma*(sum(sum(tmp1.^2)) + options.alphas(1)*(sum(sum(tmp2.^2)))+ (beta*alpha)*sum(sum((P1'*L1).*P1')));
        tmp1 = (A2 - Uy*P2');
        tmp2 = (P2(1:numCom,:) - centroidPc);
        H2 = gamma*(sum(sum(tmp1.^2)) + options.alphas(2)*(sum(sum(tmp2.^2)))+(beta*alpha)*sum(sum((P2'*L2).*P2')));
        H1 = H1^(1.0/(1-gamma));
        H2 = H2^(1.0/(1-gamma));
        tmpSum = H1+H2;
        weights(1) = (H1/tmpSum);
        weights(2) = (H2/tmpSum);
    end
    
    for i=1:2
        fprintf('%.3f ',weights(i));
    end
    fprintf(' weights: %d\n',j);
    
    logL = 0;                                   %Loss for the round
    
    %Compute the losses
    tmp1 = (A1 - Ux*P1');
    tmp2 = (P1(singX+1:end,:) - centroidPc);
    logL = logL + sum(sum(tmp1.^2)) + options.alphas(1)*(sum(sum(tmp2.^2)))+ (beta*alpha)*sum(sum((P1'*L1).*P1'));
    tmp1 = (A2 - Uy*P2');
    tmp2 = (P2(1:numCom,:) - centroidPc);
    logL = logL + sum(sum(tmp1.^2)) + options.alphas(2)*(sum(sum(tmp2.^2)))+(beta*alpha)*sum(sum((P2'*L2).*P2'));
    
    fprintf('LOG %.9f, ',logL);
    objValue = [objValue logL];                %End indicates last index of array, so basically push operation
    
    if mod(j,10)==0
    fprintf('Iteration %d, objective value %g\n', j, objValue(j));
    end

    if j>1 && ((abs(objValue(j)-objValue(j-1))/objValue(j) < error)|| objValue(j)<=error)
        fprintf('Objective value converge to %g at iteration %d before the maxIteration reached \n',objValue(j),j);
        break;
    end
    
    if sumRound==Rounds
        break;
    end
    
    optionsPGNMF.begins = singX + 1;
    optionsPGNMF.ends = size(P1,1);
    optionsPGNMF.alphaPriv = options.alphas(1);
    Ptmp = [P1(1:singX,:);centroidPc];
    [Ux, P1] = PartialGNMF(A1, K, Ptmp, W1, optionsPGNMF, Ux, P1);
    %W has not been multiplied by the weight

    optionsPGNMF.begins = 1;
    optionsPGNMF.ends = numCom;
    optionsPGNMF.alphaPriv = options.alphas(2);
    Ptmp = [centroidPc;P2(numCom+1:end,:)];
    [Uy, P2] = PartialGNMF(A2, K, Ptmp, W2, optionsPGNMF, Uy, P2);
    %Peform optimization with Pc* (centroidPc) fixed and inits finalU, finalV

end

tmp2 = (P1(singX+1:end,:) - centroidPc);
%sum(sum(tmp2.^2))
tmp2 = (P2(1:numCom,:) - centroidPc);
%sum(sum(tmp2.^2))   


P1 = P1(1:singX,:);
P2 = P2(numCom+1:end,:);
fprintf('\n');

%workspace
toc


function [U, V] = Normalize(U, V)
    [U,V] = NormalizeUV(U, V, 0, 1);

function [U, V] = NormalizeUV(U, V, NormV, Norm)
    nSmp = size(V,1);
    mFea = size(U,1);
    if Norm == 2
        if NormV
            norms = sqrt(sum(V.^2,1));
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sqrt(sum(U.^2,1));
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    else
        if NormV
            norms = sum(abs(V),1);
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sum(abs(U),1);
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = bsxfun(@times, V, norms);
        end
    end
