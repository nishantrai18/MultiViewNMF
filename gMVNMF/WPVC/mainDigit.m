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
options.WeightMode='Binary';
options.kmeans = 1;

options.varWeight = 2;
options.delta = 0.1;
options.Gaplpha=1;
options.alpha=0.1;
options.gamma = 0.5;
options.beta=10;
options.rounds = 50;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
dataname={'mfeatbig'};
num_views = 5;
numClust = 10;
options.K = numClust;
options.latentdim=numClust;

ovMean = cell(1,length(dataname));
ovStd = cell(1,length(dataname));
ovAvgStd = cell(1,length(dataname));
ovWeights = cell(1,length(dataname));
pairPortion=[0,0.1,0.3,0.5,0.7,0.9];                  %The array which contains the PER
pairPortion = 1 - (pairPortion);
for idata=1:length(dataname)  
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');        %Just the datafile name
    datafname=cell2mat(dataf(1));       
    load (datafname);                                           %Loading the datafile
    
    %% normalize data matrix
    for i=1:num_views
        X{i} = X{i}/sum(sum(X{i}));
        X{i} = X{i};
    end
    %%
    %X should be row major i.e. rows are the data points
    
   load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
   
   [numFold,numInst]=size(folds);                                   %numInst : numInstances
    
   multiMean = cell(1,length(pairPortion));
   multiStd = cell(1,length(pairPortion));
   multiWeight = cell(1,length(pairPortion));
   for f=1:1%numFold
        fprintf('Fold No. %d\n',f);
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances
        Xt=cell(1,num_views);
        for i=1:num_views
                Xt{i} = X{i}(instanceIdx,:);
        end
           meanStats = [];
           stdStats = [];
           for pairedIdx=1:1%length(pairPortion)  %here it's 1 ;different percentage of paired instances
               fprintf('Paired Idx. %d\n',pairedIdx);
               numpairedInst=floor(numInst*pairPortion(pairedIdx)+0.01);  % number of paired instances that have complete views
               paired=instanceIdx(1:numpairedInst);                     %The paired instances
               singNum=ceil((1.0/num_views)*(length(instanceIdx)-numpairedInst));

               data = cell(1,num_views);
               map = cell(1,num_views);

               rear = numpairedInst;
               for i=1:num_views
                    if (rear+singNum > length(instanceIdx))
                        singNum = (length(instanceIdx) - rear);
                    end
                    map{i} = [1:numpairedInst (rear+1):rear+singNum];
                    data{i} = Xt{i}(map{i},:);
                    rear = rear+singNum;
               end

                %Create invMap
                invMap = cell(1,numInst);
                for i=1:num_views
                    for j=1:size(map{i},2)
                        id = map{i}(j);
                        invMap{1,id} = [invMap{1,id};[i,j]];
                    end
                end

                W = cell(1,num_views);
                
                for i=1:num_views
                    data{i} = data{i}';
                    options.WeightMode='Binary';
                    sz = size(data{i},2);
                    M = EuDist2(data{i}',[],0);          %Row Major input
                    options.t = sqrt(sum(sum(M.^2))/(sz*sz));
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
              multiWeight{pairedIdx} = [multiWeight{pairedIdx};weights];
    end
   end
   for t=1:length(multiMean)
       list = mean(multiMean{t},1);
       ovMean{idata} = [ovMean{idata}; list];
       list = mean(multiWeight{t},1);
       ovWeights{idata} = [ovWeights{idata}; list];
       list = mean(multiStd{t},1);
       ovAvgStd{idata} = [ovAvgStd{idata}; list];
       list = sqrt(mean(multiStd{t},1));
       ovStd{idata} = [ovStd{idata}; list];
   end
       ovMean{idata}
       ovAvgStd{idata}
       ovWeights{idata}
end

