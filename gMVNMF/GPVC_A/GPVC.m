function [Ux,Uy,P2,P1,P3,objValue]=PVC(X2,Y2,X1,Y3,W1,W2,option)
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk.
%%  ATTN2
%   ATTN2: This package was developed by Ms. Shao-Yuan Li (lisy@lamda.nju.edu.cn). 
%   It has been further modified by Nishant Rai (nishantr@iitk.ac.in)
%%  Some varables used in the code
%   input: 
%      partial view multi-view data set :   X2: n2*dx  Y2: n2*dy  X1: n1*dx  Y3: n3*dy
%      (X2,Y2); examples appearing in both views; 
%      (X1,);   examples appearing in only view x
%      (,Y3):   examples appearing in only view y  
%      W1 : Weight matrix for the affinity graph of View 1
%      W2 : Weight matrix for the affinity graph of View 2
%      dx,dy:   feature dimension of viewx viewy
%      option.lamda: parameters controls the importance of graph regularization
%      option.latentdim: the feature dimension of latent space(default value: cluster number)
%   ouptut: 
%     Ux, Uy:   (k*dx, k*dy)  basis of latent space
%     P2,P1,P3: (n2*k, n1*k, n3*k)  data representations for examples in the latent space 

% Steps involved in algorithm:
% 		- Initialize U1, U2, P1, P2 with GNMF() declared in GNMF folder
% 		- Initialize Pc with appropriate formula, update/initialize U's alongside too
% 		- Repeat the following,
% 			- Update U's, P1, P2 fixing Pc using the multiplicative updates (Or using PerViewNMF())
% 			- Update Pc and U's using the formula
% 		- Normalise U's and V's at the end (Or during it (Depends))
%% End of Instruction
    rand('seed',1);
    
    error = option.error;
    lamda = option.lamda;  
    k = option.latentdim;                               %Get the parameters
    
    maxIterGPVC=20;
    maxIterInit = 20; 
    maxIterNMF=500; 
    trace=1;                                               %Whether to compute objective value per iteration
    [numInst1,Featx]=size(X1');                             %Number of instances, FeatX: Number of features in view 1
    [numInst2,Featx]=size(X2');                             %Number of instances with complete views
    [numInst3,Featy]=size(Y3');                             %Number of instances, FeatY: Number of features in view 2
    
    objValue=zeros(maxIterGPVC,1);                           %Objective Value after each iteration
 
    P1init=rand(numInst1,k);                                %Random initialization
    P2init=rand(numInst2,k);
    P3init=rand(numInst3,k);
    Uxinit=rand(Featx,k);
    Uyinit=rand(Featy,k);
    
    P1=P1init;
    P2=P2init;
    P3=P3init;
    
    Ux=Uxinit;
    Uy=Uyinit;
    
    nSmp = size(W1,1);    
    W1 = lamda*W1;
    DCol = full(sum(W1,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L1 = D - W1;
    
    nSmp = size(W2,1);
    W2 = lamda*W2;
    DCol = full(sum(W2,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L2 = D - W2;
    
    Goption.alpha=option.Gaplpha;

%% Initialize U1, U2, P1, P2 with GNMF() declared in GNMF folder   
    if (numInst1)    
       [Ux, P1] = GNMF(X1, k, W1( 1:size(X1,2), 1:size(X1,2) ), option);
    end
    if (numInst3)    
       [Uy, P3] = GNMF(Y3, k, W2( (size(X2,2)+1):end, (size(X2,2)+1):end ), option);
    end

%% Initialize Pc/P3 with appropriate formula (To be decided)

%norm(horzcat(X1,X2)-Ux*[P1;P2]','fro')+ norm(horzcat(Y2,Y3)-Uy*[P2;P3]','fro')+ sum(sum(([P1;P2]'*L1)*[P1;P2])) + sum(sum(([P2;P3]'*L2)*[P2;P3]))

    optionsPc.error = option.error;
    optionsPc.maxIter = option.maxIter;
    optionsPc.minIter = option.minIter;
    optionsPc.alpha = option.alpha;   
    optionsPc.rounds = option.rounds;   
    optionsPc.numCom = size(X2,2);

    %workspace
    
    [P2, Ux, Uy] = UpdatePcU(horzcat(X1,X2), horzcat(Y2,Y3), [P1;P2], P2, [P2;P3], k, W1, W2, optionsPc, Ux, Uy);
%%
%norm(horzcat(X1,X2)-Ux*[P1;P2]','fro')+ norm(horzcat(Y2,Y3)-Uy*[P2;P3]','fro')+ sum(sum(([P1;P2]'*L1)*[P1;P2])) + sum(sum(([P2;P3]'*L2)*[P2;P3]))
%norm(horzcat(X1,X2)-Ux*[P1;P2]','fro')+ norm(horzcat(Y2,Y3)-Uy*[P2;P3]','fro')

%% Repeated optimizations
   for iter=1:maxIterGPVC
        %iter
        % Update U's, P1, P2 fixing Pc using the multiplicative updates (Or using PerViewNMF())
        % Call GNMF() with initial values
        if (numInst1)    
           [Ux, P1] = GNMF(X1, k, W1( 1:size(X1,2), 1:size(X1,2) ), option, Ux, P1);
        end
        if (numInst3)    
           [Uy, P3] = GNMF(Y3, k, W2( size(X2,2)+1:end, size(X2,2)+1:end ), option, Uy, P3);
        end
        % Update Pc using the formula (To be decided)
        [P2, Ux, Uy] = UpdatePcU(horzcat(X1,X2), horzcat(Y2,Y3), [P1;P2], P2, [P2;P3], k, W1, W2, optionsPc, Ux, Uy);
        
        objValue(iter)=norm(horzcat(X1,X2)-Ux*[P1;P2]','fro')+ norm(horzcat(Y2,Y3)-Uy*[P2;P3]','fro')+ sum(sum(([P1;P2]'*L1)*[P1;P2])) + sum(sum(([P2;P3]'*L2)*[P2;P3]));

        %fprintf('%.10f SCORE 1\n',norm(horzcat(X1,X2)-Ux*[P1;P2]','fro')+ norm(horzcat(Y2,Y3)-Uy*[P2;P3]','fro')+ sum(sum(([P1;P2]'*L1)*[P1;P2])) + sum(sum(([P2;P3]'*L2)*[P2;P3])));
        
        if mod(iter,10)==0
        fprintf('Iteration %d, objective value %g\n', iter, objValue(iter));
        end
        
        if iter>1 && ((abs(objValue(iter)-objValue(iter-1))/objValue(iter) < error)|| objValue(iter)<=error)
            fprintf('Objective value converge to %g at iteration %d before the maxIteration reached \n',objValue(iter),iter);
            break;
        end
   end

   % Normalise U's and V's at the end (Or during it (Depends))
   %workspace;
   fprintf('%.10f SCORE 1\n',norm(horzcat(X1,X2)-Ux*[P1;P2]','fro')+ norm(horzcat(Y2,Y3)-Uy*[P2;P3]','fro')+ sum(sum(([P1;P2]'*L1)*[P1;P2])) + sum(sum(([P2;P3]'*L2)*[P2;P3])));
        
 end

  
