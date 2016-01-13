clear all;                      %Remove all variables from the workspace
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
options.rounds = 100;
options.WeightMode='Binary';
options.kmeans = 1;

options.gamma = 2;
options.Gaplpha=1;
options.alpha=0.1;
options.delta = 0.1;
options.beta=0.1;
options.varWeight = 0;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
%dataname={'3sourcesbg','3sourcesbr','3sourcesgr'};
dataname={'bbcsport12'};
num_views = 2;
numClust = 5;
options.K = numClust;

ovMean = cell(1,length(dataname));
ovStd = cell(1,length(dataname));
ovAvgStd = cell(1,length(dataname));
pairPortion=[0,0.1,0.3,0.5,0.7,0.9];                  %The array which contains the PER
%pairPortion=[0];                  %The array which contains the PER
pairPortion = 1 - (pairPortion);
for idata=1:length(dataname)  
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');        %Just the datafile name
    datafname=cell2mat(dataf(1));       
    load (datafname);                                           %Loading the datafile
    
    %X1=readsparse(X1);                                         %Loading a sparse matrix i.e. on the basis of edges                  
    %X2=readsparse(X2);
    %% normalize data matrix
        X1 = X1 / sum(sum(X1));
        X2 = X2 / sum(sum(X2));
    %%
    
    Xf1 = X1';                                                     %Directly loading the matrices
    Xf2 = X2';
    X{1} =Xf1;                                                  %View 1
    X{2} =Xf2;                                                  %View 2    
    %X should be row major i.e. rows are the data points
    
   load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
   folds = folds(:,:);
   
   [numFold,numInst]=size(folds);                                   %numInst : numInstances
    
   multiMean = cell(1,length(pairPortion));
   multiStd = cell(1,length(pairPortion));
   for f=1:3%numFold
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances
        Xt=cell(1,num_views);
        for i=1:num_views
                Xt{i} = X{i}(instanceIdx,:);
        end
        for v1=1:num_views
           for v2=v1+1:num_views
               if v1==v2
               continue;
               end
                
               meanStats = [];
               stdStats = [];
               for pairedIdx=1:length(pairPortion)  %here it's 1 ;different percentage of paired instances
                   numpairedInst=floor(numInst*pairPortion(pairedIdx)+0.01);  % number of paired instances that have complete views
                   paired=instanceIdx(1:numpairedInst);                     %The paired instances
                   singledNumView1=ceil(0.5*(length(instanceIdx)-numpairedInst));
                   singleInstView1=instanceIdx(numpairedInst+1:numpairedInst+singledNumView1);   %the first view and second view miss half to half (Since they are mutually exclusive)
                   singleInstView2=instanceIdx(numpairedInst+singledNumView1+1:end);             %instanceIdx(numpairedInst+numsingleInstView1+1:end);
                   
                   data = cell(1,num_views);
                   map = cell(1,num_views);
                   
                   %{
                   %Construct Map
                   map{1} = [singleInstView1 paired];
                   map{2} = [paired singleInstView2];                   
                   data{1}=X{v1}(map{1},:);                                 %View 1 of paired
                   data{2}=X{v2}(map{2},:);                                 %View 1 of paired
                   %}
                   map{1} = [1:numpairedInst numpairedInst+1:numpairedInst+singledNumView1];
                   map{2} = [1:numpairedInst numpairedInst+singledNumView1+1:length(instanceIdx)];
                   %map{2} = [paired singleInstView2];                   
                   data{1}=Xt{v1}(map{1},:);                                 %View 1 of paired
                   data{2}=Xt{v2}(map{2},:);                                 %View 1 of paired
                                      
                   options.latentdim=numClust;
                   
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
                        options.WeightMode='Cosine';
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

ovMean{1}
ovAvgStd{1}
ovMean{2}
ovAvgStd{2}
ovMean{3}
ovAvgStd{3}
