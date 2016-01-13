clear;                      %Remove all variables from the workspace
%clc;
 
addpath(genpath('../../partialMV/PVC/recreateResults/measure/'));
addpath(genpath('../../partialMV/PVC/recreateResults/misc/'));
addpath(genpath(('../code/')));
addpath('../tools/');
addpath('../print/');
addpath('../');

options = [];
options.maxIter = 100;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 20;
options.WeightMode='Binary';
options.varWeight = 1;
options.kmeans = 1;

options.gamma = 2;
options.Gaplpha=0.1;                            %Graph regularisation parameter
options.alpha=0.01;
options.delta = 0.01;
options.beta=0.01;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
dataname={'citeseer'};
num_views = 2;
numClust = 6;
options.K = numClust;
options.latentdim=numClust;

ovMean = cell(1,length(dataname));
ovStd = cell(1,length(dataname));
ovAvgStd = cell(1,length(dataname));
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
   folds = folds(:,1:800);
   
   [numFold,numInst]=size(folds);                                   %numInst : numInstances
   dir=strcat(resdir,cell2mat(dataname(idata)),'/'); %    train_target(idnon)=-1;   ranksvm treat weak label {-1: -1; 1:+1; 0:-1}
   %mkdir(dir);                              %Creates new folder for storing the workspace variables 
    
   multiMean = cell(1,length(pairPortion));
   multiStd = cell(1,length(pairPortion));
   for f=1:3%numFold
           instanceIdx=folds(f,:);
           truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances                
           for i=1:num_views
                Xt{i} = X{i}(instanceIdx,:);
           end
           meanStats = [];
           stdStats = [];
           for pairedIdx=1:1%length(pairPortion)  %here it's 1 ;different percentage of paired instances
               numpairedInst=floor(numInst*pairPortion(pairedIdx)+0.01);  % number of paired instances that have complete views
               paired=instanceIdx(1:numpairedInst);                     %The paired instances
               singNum=ceil((1.0/num_views)*(length(instanceIdx)-numpairedInst) + 0.01);
               
               singView = cell(1,num_views);
               data = cell(1,num_views);
               map = cell(1,num_views);
               
               rear=0;
               for i=1:num_views
                    singView{i} = instanceIdx(rear+1:rear+singNum);
                    map{i} = [1:numpairedInst (rear+1):rear+singNum];
                    data{i} = Xt{i}(map{i},:);
                    rear = rear+singNum;
               end
               
                %Create invMap
                invMap = cell(1,numInst);
                for i=1:num_views
                    for j=1:length(map{i})
                        id = map{i}(j);
                        invMap{1,id} = [invMap{1,id};[i,j]];
                    end
                end

                W = cell(1,num_views);

                for i=1:num_views
                    data{i} = data{i}';
                    W{i} = constructW_cai(data{i}',options);
                    %Weight matrix constructed for each view
                end

              [U, V, centroidV, weights, log] = PartialGNMF(data, numClust, W, map, invMap, truthF, options);

              [~,stats] = ComputeStats(centroidV, truthF, numClust);

              for s=1:size(stats,1)
                  meanStats(s) = mean(stats(s,:));
                  stdStats(s) = std(stats(s,:));
              end
              multiMean{pairedIdx} = [multiMean{pairedIdx};meanStats];
              multiStd{pairedIdx} = [multiStd{pairedIdx};stdStats];
           end
   end


   for t=1:length(multiMean)
       list = mean(multiMean{t},1);
       ovMean{idata} = [ovMean{idata}; list];
       list = mean(multiStd{t},1);
       ovAvgStd{idata} = [ovAvgStd{idata}; list];
       list = sqrt(mean(multiStd{t},1));
       ovStd{idata} = [ovStd{idata}; list];
   end
       ovMean{idata}
       ovAvgStd{idata}
end