function [Ux Uy P2 P1 P3 objValue F P R nmi avgent AR] = PVCclust(X2,Y2,X1,Y3,numClust,truth,option)
%%  This function calls PVC method and then conduct k-means to get the clustering result  
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your
%   own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
%%  ATTN2
%   ATTN2: This package was developed by Ms. Shao-Yuan Li (lisy@lamda.nju.edu.cn). For any problem concerning the code,
%        please feel free to contact Ms. Li.
%%  Some varables used in the code
%   input : 
%      partial view multi-view data set :   X2: n2*dx  Y2: n2*dy  X1: n1*dx  Y3: n3*dy; (dx,dy:   feature dimension of viewx viewy)
%      (X2,Y2); examples appearing in both views; 
%      (X1,);   examples appearing in only view x
%      (,Y3):   examples appearing in only view y  
%      numClust: number of clusters
%      truth:   groundtruth cluster id of each example
%      option.lamda: parameters controls the importance of  and l1 (recommend value: 1e-2)
%      option.latentdim: the feature dimension of latent space(default value: cluster number)
%   ouptut: 
%     Ux, Uy:   (k*dx, k*dy)  basis of latent space
%     P2,P1,P3: (n2*k, n1*k, n3*k)  data representations for examples in the latent space 

%%  Reference:
%    S.-Y. Li, Y. Jiang nd Z.-H. Zhou. Partial Multi-View Clustering. In: Proceedings of the 28th AAAI Conference on 
%    Artificial Intelligence (AAAI'14),Qu¨¦bec, Canada ,2014.
%% End of Instruction
  if (min(truth)==0)
        truth = truth + 1;
  end

  [Ux,Uy,P2,P1,P3,objValue]=PVC(X2,Y2,X1,Y3,option);
 
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
  
    kmeans_avg_iter = 20;
    
    fprintf('running k-means...\n');
    
    for i=1: kmeans_avg_iter
        C = kmeans(PIn,numClust,'EmptyAction','drop');
        [A nmii(i) avgenti(i)] = compute_nmi(truth,C);
        [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C);
        [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C);
    end
    F = mean(Fi);
    P = mean(Pi);
    R = mean(Ri);
    nmi = mean(nmii);
    avgent = mean(avgenti);
    AR = mean(ARi);
    
   fprintf('nmi: %f(%f)\n', nmi, std(nmii));
    