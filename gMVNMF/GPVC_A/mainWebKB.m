clear all;                      %Remove all variables from the workspace
%clc;

addpath(genpath('../../partialMV/PVC/recreateResults/measure/'));
addpath(genpath('../../partialMV/PVC/recreateResults/misc/'));
addpath(genpath(('../code/')));
addpath('../tools/');
addpath('../print/');
addpath('../');

options = [];
options.maxIter = 200;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 30;
options.Gaplpha=0.1;                            %Graph regularisation parameter
options.alpha=0;
options.WeightMode='Binary';

options.alphas = [options.alpha, options.alpha];
options.kmeans = 1;
options.beta=0;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
dataname={'mfeat'};
num_views = 2;
options.K = 5;
numClust = 5;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
dataname={'cornell','texas','washington','wisconsin'};

scores = [];
pairPortion=[0,0.1,0.3,0.5,0.7,0.9];                  %The array which contains the PER
%pairPortion=[0.3,0.5];                  %The array which contains the PER
pairPortion = 1 - (pairPortion);
for idata=1:1%length(dataname)  
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');        %Just the datafile name
    datafname=cell2mat(dataf(1));       
    load (datafname);                                           %Loading the datafile

    Xf1=readsparse(X1);                                         %Loading a sparse matrix i.e. on the basis of edges                  
    Xf2=readsparse(X2);
    
    %% normalize data matrix
        Xf1 = Xf1 / sum(sum(Xf1));
        Xf2 = Xf2 / sum(sum(Xf2));
    %%
    
    X{1} =Xf1;                                                  %View 1
    X{2} =Xf2;                                                  %View 2
 
   load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
   [numFold,numInst]=size(folds);                                   %numInst : numInstances
   dir=strcat(resdir,cell2mat(dataname(idata)),'/'); %    train_target(idnon)=-1;   ranksvm treat weak label {-1: -1; 1:+1; 0:-1}
   mkdir(dir);                              %Creates new folder for storing the workspace variables 
    
   multiScore = [];
   for f=1:1%length(numFold)
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances
        for v1=1:num_views
           for v2=v1+1:num_views
               if v1==v2
               continue;
               end
                
               pscore = [];
               for pairedIdx=1:length(pairPortion)  %here it's 1 ;different percentage of paired instances
                   numpairedInst=floor(numInst*pairPortion(pairedIdx)+0.01);  % number of paired instances that have complete views
                   paired=instanceIdx(1:numpairedInst);                     %The paired instances
                   singledNumView1=ceil(0.5*(length(instanceIdx)-numpairedInst));
                   singleInstView1=instanceIdx(numpairedInst+1:numpairedInst+singledNumView1);   %the first view and second view miss half to half (Since they are mutually exclusive)
                   singleInstView2=instanceIdx(numpairedInst+singledNumView1+1:end);             %instanceIdx(numpairedInst+numsingleInstView1+1:end);
                   xpaired=X{v1}(paired,:);                                 %View 1 of paired
                   ypaired=X{v2}(paired,:);                                 %View 1 of single
                   xsingle=X{v1}(singleInstView1,:);                        %View 2 of paired
                   ysingle=X{v2}(singleInstView2,:);                        %View two of single
         
                  options.lamda=0;                                        %Sparsity parameter for Lasso norm
                  options.latentdim=numClust;
                  
                    W1 = constructW_cai([xsingle;xpaired],options);
                    W2 = constructW_cai([ypaired;ysingle],options);
                    %Weight matrix constructed for each view
      
                  [U1 U2 P2 P1 P3 objValue F P R nmi avgent AR] = GPVCclust(xpaired',ypaired',xsingle',ysingle',W1,W2,numClust,truthF,options);
                  %[5 unknowns objectiveValue 6 stats] = func([X12][2],X1,X2, numClust,trueClusts,Parameters); 
                  
                  pscore = [pscore;nmi];
                  
                  %save([dir,'PVC',num2str(v1),num2str(v2),'paired_',num2str(pairPortion(pairedIdx)),'f_',num2str(f),'.mat'],'U1','U2','P2','P1','P3','objValue','F','P','R','nmi','avgent','AR','truthF');       
                  %save (filenameWithDirectory, variables)
               end
               if f==1
                   multiScore = pscore;
               else
                   multiScore = horzcat(multiScore,pscore);
               end
      end
    end
   end
   multiScore
   list = mean(multiScore, 2);
   list'
    scores = [scores;list'];
end
scores

       
         
 