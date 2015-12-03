function [Ux,Uy,P,objValue]=PVCinit(X,Y,option)
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
%%  ATTN2
%   ATTN2: This package was developed by Ms. Shao-Yuan Li (lisy@lamda.nju.edu.cn). For any problem concerning the code,
%        please feel free to contact Ms. Li.
%%  Some varables used in the code
%  Input: 
%        X: n*dx  Y: n*dy:   multi-view data set in view X and Y
%       option(1): lamda: parameters controls the importance of  l1 
%       option(2): latentdim: the feature dimension of latent space
%       option(3): maxIterInit; the max iteration number of TwoViewLSA
%  Ouptut: 
%      Ux, Uy:   (k*dx, k*dy)  basis of latent space for view X and Y
%      P :       (n*k )  data representations for examples in the latent space 

% algorithm:
%         min(P>0,Ux>0,Uy>0) | X-P*Ux|_F^2 + |Y-P*Uy|_F^2+lamd*|P|_1 
%               P: n*k   Ux: k*dx, Uy: k*dy
%               paramter normalized:  lamda -> lamda * { row([X,Y])*col([X,Y]) }/ { row(P)*col(P)  }
%    
% Optimization: AO
%   1. fix P, optimize Ux,Uy:
%     min(Ux>0)  | X-P*Ux|_F^2     [Ux',objUx']=argminF2P2Constrlasso(X',P',X',P',Ux'init,[0,0,maxiter,trace,interval_0])
%     min(Uy>0)  | Y-P*Uy|_F^2     [Uy',objUy']=argminF2P2Constrlasso(Y',P',Y',P',Uy'init,[0,0,maxiter,trace,interval_0])
% 
%   2. fix Ux,Uy, optimize P
%      min(P>0) | X-P*Ux|_F^2 + |Y-P*Uy|_F^2+lamd*|P|_1     [P,objP]=argminF2P2Constrlasso(X,Ux,Y,Uy,Pinit,[lamda,maxiter,trace,interval_1])

%%  Reference:
%    S.-Y. Li, Y. Jiang nd Z.-H. Zhou. Partial Multi-View Clustering. In: Proceedings of the 28th AAAI Conference on 
%    Artificial Intelligence (AAAI'14),Qu¨¦bec, Canada ,2014.
%% End of Instruction
    rand('seed',1);

    lamda =option(1); 
    k =option(2); 
    maxIter = option(3);
    
    maxIterNMF=500;  
    
    
    [numInst,Featx]=size(X);
    [numInst,Featy]=size(Y);
    objValue=zeros(maxIter,1);
    
    Pinit=rand(numInst,k);
    Uxinit=rand(k,Featx);
    Uyinit=rand(k,Featy);
    
    P=Pinit;
    Ux=Uxinit;
    Uy=Uyinit;
    for iter=1:maxIter
        iter
%     fix P, optimize Ux,Uy:
%     min(Ux>0)  | X-P*Ux|_F^2     [Ux',objUx']=argminF2P2Constrlasso(X',P',X',P',Ux'init,[0,0,maxiter,trace,interval_0])
%     min(Uy>0)  | Y-P*Uy|_F^2     [Uy',objUy']=argminF2P2Constrlasso(Y',P',Y',P',Uy'init,[0,0,maxiter,trace,interval_0]) 
         [Uxt,objUxt]=argminF2P2Constrlasso(X',P',X',P',Ux',[0,0,maxIterNMF,0,0]);
         Ux=Uxt';
         [Uyt,objUyt]=argminF2P2Constrlasso(Y',P',Y',P',Uy',[0,0,maxIterNMF,0,0]);
         Uy=Uyt';
         
% min(P>0) | X-P*Ux|_F^2 + |Y-P*Uy|_F^2+lamd*|P|_1          
         [P,objP]=argminF2P2Constrlasso(X,Ux,Y,Uy,P,[1,lamda,maxIterNMF,0,0]);
        
        objValue(iter)=objP(maxIterNMF); 
        if mod(iter,50)==0
        fprintf('multi-viewLSA Iteration %d, objective value %g\n', iter, objValue(iter));
        end
 
        if iter>1 && ((abs(objValue(iter)-objValue(iter-1))/objValue(iter) < 1e-6)|| objValue(iter)<=1e-6)
            fprintf('multi-viewLSA  Objective value converge to %g at iteration %d before the maxIteration reached \n',objValue(iter),iter);
            break;
        end
    end
    
    
end

  