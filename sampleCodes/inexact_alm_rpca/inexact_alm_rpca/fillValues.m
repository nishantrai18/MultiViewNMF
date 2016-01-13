clear;                      %Remove all variables from the workspace
%clc;
 
addpath(genpath('../../../partialMV/PVC/recreateResults/misc/'));

resdir='data/result/';
datasetdir='../../../partialMV/PVC/recreateResults/data/';
dataname={'cora'};
num_views = 2;
numClust = 7;
options.K = numClust;
options.latentdim=numClust;

pairPortion=[0,0.1,0.3,0.5,0.7,0.9];                  %The array which contains the PER
pairPortion = 1 - (pairPortion);
for idata=1:length(dataname)  
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');        %Just the datafile name
    datafname=cell2mat(dataf(1));       
    load (datafname);                                           %Loading the datafile
    
    X1=readsparse(X1);                                         %Loading a sparse matrix i.e. on the basis of edges                  
    X2=readsparse(X2);
    %% normalize data matrix
        X1 = X1 / sum(sum(X1));
        X2 = X2 / sum(sum(X2));
    %%
    
    Xf1 = X1;                                                     %Directly loading the matrices
    Xf2 = X2;
    X{1} =Xf1;                                                  %View 1
    X{2} =Xf2;                                                  %View 2    
    %X should be row major i.e. rows are the data points
    
   load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
   
   [numFold,numInst]=size(folds);                                   %numInst : numInstances
   dir=strcat(resdir,cell2mat(dataname(idata)),'/'); %    train_target(idnon)=-1;   ranksvm treat weak label {-1: -1; 1:+1; 0:-1}
   %mkdir(dir);                              %Creates new folder for storing the workspace variables 
    
   views = cell(num_views,10,length(pairPortion));
   for f=1:10%numFold
           instanceIdx=folds(f,:);
           truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances                
           for i=1:num_views
                Xt{i} = X{i}(instanceIdx,:);
           end
           for pairedIdx=1:length(pairPortion)  %here it's 1 ;different percentage of paired instances
               numpairedInst=floor(numInst*pairPortion(pairedIdx));  % number of paired instances that have complete views
               paired=instanceIdx(1:numpairedInst);                     %The paired instances
               singNum=ceil((1.0/num_views)*(length(instanceIdx)-numpairedInst));
               
               singView = cell(1,num_views);
               data = cell(1,num_views);
               map = cell(1,num_views);
               
               Xfilled{pairedIdx} = [];
               rear = numpairedInst;                        %Zero to handle the matrix needed to be completed
               for i=1:num_views
                    if (rear+singNum > length(instanceIdx))
                        singNum = (length(instanceIdx) - rear);
                    end
                    Xfill{i} = repmat(mean(Xt{i}),length(instanceIdx)-numpairedInst,1);
                    map{i} = [(rear+1-numpairedInst):rear+singNum-numpairedInst];
                    nmap{i} = [(rear+1):rear+singNum];
                    Xfill{i}(map{i},:) = Xt{i}(nmap{i},:);
                    rear = rear+singNum;
                    Xfilled{pairedIdx} = [Xfilled{pairedIdx} Xfill{i}];
               end
               Xfilled{pairedIdx} = inexact_alm_rpca(Xfilled{pairedIdx});
               sum(sum(Xfilled{pairedIdx}))
               rear = 0;
               for i=1:num_views
                    tmpMat = Xfilled{pairedIdx}(:,rear+1:rear+size(Xt{i},2));
                    views{i,f,pairedIdx} = [Xt{i}(1:numpairedInst,:); tmpMat];
                    rear = rear+size(Xt{i},2);
               end
           end
   end
end