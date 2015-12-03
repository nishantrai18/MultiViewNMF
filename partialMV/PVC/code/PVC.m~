 function [Ux,Uy,P2,P1,P3,objValue]=PVC(X2,Y2,X1,Y3,option)
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
%%  ATTN2
%   ATTN2: This package was developed by Ms. Shao-Yuan Li (lisy@lamda.nju.edu.cn). For any problem concerning the code,
%        please feel free to contact Ms. Li.
%%  Some varables used in the code
%   input: 
%      partial view multi-view data set :   X2: n2*dx  Y2: n2*dy  X1: n1*dx  Y3: n3*dy
%      (X2,Y2); examples appearing in both views; 
%      (X1,);   examples appearing in only view x
%      (,Y3):   examples appearing in only view y  
%      dx,dy:   feature dimension of viewx viewy
%      option.lamda: parameters controls the importance of  and l1 (recommend value: 1e-2)
%      option.latentdim: the feature dimension of latent space(default value: cluster number)
%   ouptut: 
%     Ux, Uy:   (k*dx, k*dy)  basis of latent space
%     P2,P1,P3: (n2*k, n1*k, n3*k)  data representations for examples in the latent space 

% algorithm:
%         min(P1,P2,P3 >0,Ux>0,Uy>0) | X1-P1*Ux|_F^2 + | X2-P2*Ux|_F^2 + |Y2-P2*Uy|_F^2 + |Y3-P3*Uy|_F^2 + lamd*|P|_1 . 
%                P=[P1;P2;P3]: n*k   Ux: k*dx, Uy: k*dy
%  estimate latent reprensentation and basis from partial view  data

%  initialize: 1.1  [Ux,Uy,P2]=argmin(P2,Ux>0,Uy>0)  | X2-P2*Ux|_F^2+ |Y2-P2*Uy|_F^2+lamd2*|P2|_1 
% while 
%   1. update P1,P3 given Ux,Uy,X1,Y3
%   1.1   P1=argmin(P1>0)  |X1-P1*Ux|_F^2 + lamd2*|P1|_1
%   1.2   P3=argmin(P3>0)  |Y3-P3*Uy|_F^2 + lamd2*|P3|_1
%   2. update Ux,Uy given (P1;P2) (X1;X2)  and (P2:P3) (Y2:Y3)
%   2.1   Ux=argmin(Ux>0)  |X1-P1*Ux|_F^2 + | X2-P2*Ux|_F^2
%   2.2   Uy=argmin(Uy>0)  |Y2-P2*Uy|_F^2 + |Y3-P3*Uy|_F^2
%   3. update P2 given (Ux,Uy,X2,Y2)
%         P2=argmin(P2>0)  | X2-P2*Ux|_F^2+ |Y2-P2*Uy|_F^2 + lamd2*|P2|_1 
%  endwhile 
%%  Reference:
%    S.-Y. Li, Y. Jiang nd Z.-H. Zhou. Partial Multi-View Clustering. In: Proceedings of the 28th AAAI Conference on 
%    Artificial Intelligence (AAAI'14),Qu¨¦bec, Canada ,2014.
%% End of Instruction
    rand('seed',1);
    
    lamda = option.lamda;  
    k = option.latentdim; 
    
    maxIterPVC=20;
    maxIterInit = 20; 
    maxIterNMF=500; 
    trace=1;  
    [numInst1,Featx]=size(X1);
    [numInst2,Featx]=size(X2);
    [numInst3,Featy]=size(Y3);
    
    % parameter normalization
    lamdatmpP2=lamda*( (Featy+Featx)/k );
    lamdatmpP1=lamda*( Featx/k  );
    lamdatmpP3=lamda*( Featy/k  );
    
    objValue=zeros(maxIterPVC,1);
 
    P1init=rand(numInst1,k);
    P2init=rand(numInst2,k);
    P3init=rand(numInst3,k);
    Uxinit=rand(k,Featx);
    Uyinit=rand(k,Featy);
    
    P1=P1init;
    P2=P2init;
    P3=P3init;
    
    Ux=Uxinit;
    Uy=Uyinit;
 
%  initialize: 1.1  [Ux,Uy,P2]=argmin(P2>0,Ux>0,Uy>0)  | X2-P2*Ux|_F^2+ |Y2-P2*Uy|_F^2+lamd2*|P|_1   
     [Ux,Uy,P2,objValueInit]=PVCinit(X2,Y2,[lamdatmpP2,k,maxIterInit]);
     
   if(numInst1 || numInst2) 
    for iter=1:maxIterPVC
        iter
%   1. update P1,P3 given Ux,Uy,X1,Y3
%   1.1   [P1]=argmin(P1\in[0,1])   |X1-P1*Ux|_F^2 + lamd*|P1|_1
%   1.2   [P3]=argmin(P3\in[0,1])   |Y3-P3*Uy|_F^2 + lamd*|P3|_1
       if (numInst1)    
         [P1,objP1]=argminF2P2Constrlasso(X1,Ux,X1,Ux,P1,[0,lamdatmpP1,maxIterNMF,0,0]);
        end
        if (numInst3)
         [P3,objP3]=argminF2P2Constrlasso(Y3,Uy,Y3,Uy,P3,[0,lamdatmpP3,maxIterNMF,0,0]);
        end
%   2. update Ux,Uy given (P1;P2) (X1;X2)  and (P2:P3) (Y2:Y3)
%   2.1  min(Ux>0) |X1-P1*Ux|_F^2 + |X2-P2*Ux|_F^2
%   2.2  min(Uy>0) |Y2-P2*Uy|_F^2 + |Y3-P3*Uy|_F^2  
         [Uxt,objUxt]=argminF2P2Constrlasso( [X1;X2]',[P1;P2]',[X1;X2]',[P1;P2]',Ux',[0,0,maxIterNMF,0,0]);
          Ux=Uxt';
         [Uyt,objUyt]=argminF2P2Constrlasso( [Y2;Y3]',[P2;P3]',[Y2;Y3]',[P2;P3]',Uy',[0,0,maxIterNMF,0,0]);
          Uy=Uyt';
%   3. update P2 given (Ux,Uy,X2,Y2)
%      [P2]=argmin(P2>0)  | X2-P2*Ux|_F^2+ |Y2-P2*Uy|_F^2+lamd2*|P2|_1    
         [P2,objP2]=argminF2P2Constrlasso(X2,Ux,Y2,Uy,P2,[1,lamdatmpP2,maxIterNMF,0,0]);
       
        objValue(iter)=norm([X1;X2]-[P1;P2]*Ux,'fro')+ norm([Y2;Y3]-[P2;P3]*Uy,'fro')+ lamdatmpP1*sum(sum(P1)) + lamdatmpP2*sum(sum(P2))+ lamdatmpP3*sum(sum(P3));
        
        if mod(iter,50)==0
        fprintf('semi-paired LSA Iteration %d, objective value %g\n', iter, objValue(iter));
        end
        
        if iter>1 && ((abs(objValue(iter)-objValue(iter-1))/objValue(iter) < 1e-6)|| objValue(iter)<=1e-6)
            fprintf('multi-viewLSA  Objective value converge to %g at iteration %d before the maxIteration reached \n',objValue(iter),iter);
            break;
        end
    end
   end
 end

  