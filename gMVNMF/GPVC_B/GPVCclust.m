%function [Ux Uy P2 P1 P3 objValue F P R nmi avgent AR pure] = PVCclust(X2, Y2, X1, Y3, W1, W2, numClust, truth, option)
function [Ux Uy P2 P1 P3 objValue stats] = PVCclust(X2, Y2, X1, Y3, W1, W2, numClust, truth, option)
%%  This function calls GPVC method and then conduct k-means to get the clustering result  
%%  This package was developed by Ms. Shao-Yuan Li (lisy@lamda.nju.edu.cn).
%   It has been further modified by Nishant Rai (nishantr@iitk.ac.in)
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk.
%%  Some varables used in the code
%   input : 
%      partial view multi-view data set :   X2: n2*dx  Y2: n2*dy  X1: n1*dx  Y3: n3*dy; (dx,dy:   feature dimension of viewx viewy)
%      (X2,Y2); examples appearing in both views; 
%      (X1,);   examples appearing in only view x
%      (,Y3):   examples appearing in only view y  
%      W1 : Weight matrix for the affinity graph of View 1
%      W2 : Weight matrix for the affinity graph of View 2
%      numClust: number of clusters
%      truth:   groundtruth cluster id of each example
%      option.lamda: parameters controls the importance of  and l1 (recommend value: 1e-2)
%      option.latentdim: the feature dimension of latent space(default value: cluster number)
%   ouptut: 
%     Ux, Uy:   (k*dx, k*dy)  basis of latent space
%     P2,P1,P3: (n2*k, n1*k, n3*k)  data representations for examples in the latent space 
%% End of Instruction

if (min(truth)==0)
        truth = truth + 1;                  %Keep the minimum id of clusters 1? (probably)
  end

  [Ux,Uy,P2,P1,P3,objValue] = GPVC(X2,Y2,X1,Y3,W1,W2,option);
 
  %At this step we have our matrices, now perform stat collection
  
  UPI=[P2;P1;P3]; 
  
if (1)
    norm_mat = repmat(sqrt(sum(UPI.*UPI,2)),1,size(UPI,2));
    %%avoid divide by zero
    for i=1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    PIn = UPI./norm_mat;
    end
    %PIn = UPI;
    [~,stats] = ComputeStats(PIn, truth, numClust);

   