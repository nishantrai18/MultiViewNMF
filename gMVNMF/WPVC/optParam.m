clear all;                      %Remove all variables from the workspace
%clc;
 
addpath(genpath('../../partialMV/PVC/recreateResults/measure/'));
addpath(genpath('../../partialMV/PVC/recreateResults/misc/'));
addpath(genpath(('../code/')));
addpath('../tools/');
addpath('../print/');
addpath('../');

options = [];
options.maxIter = 75;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 50;
options.WeightMode='Cosine';
options.kmeans = 1;

options.gamma = 2;
options.Gaplpha=1;
options.varWeight = 0;

alphas = [0.01, 0.1, 1, 10];
deltas = [0.01, 0.1, 1, 10];
betas = [0.1, 1, 10, 100];
%options.alpha=0.1;
%options.delta = 0.1;
%options.beta=0.1;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
%dataname={'3sourcesbg','3sourcesbr','3sourcesgr'};
dataname={'mfeat'};
num_views = 2;
numClust = 10;
options.K = numClust;

fprintf('%s\n',cell2mat(dataname(1)));

values = cell(length(alphas),length(deltas),length(betas));
pairPortion=[1];                  %The array which contains the PER
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
    
    Xf1 = X1;                                                     %Directly loading the matrices
    Xf2 = X2;
    X{1} =Xf1;                                                  %View 1
    X{2} =Xf2;                                                  %View 2    
    %X should be row major i.e. rows are the data points
    
   load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
   folds = folds(:,:);
   
   [numFold,numInst]=size(folds);                                   %numInst : numInstances
    
   multiMean = cell(1,length(pairPortion));
   multiStd = cell(1,length(pairPortion));
   for f=1:1%numFold
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances
        Xt=cell(1,num_views);
        for i=1:num_views
                Xt{i} = X{i}(instanceIdx,:);
        end
        for v1=1:length(alphas)
           for v2=1:length(deltas)
               for v3=1:length(betas)
                   
                   options.alpha=alphas(v1);
                   options.delta = deltas(v2);
                   options.beta=betas(v3);
                   for pairedIdx=1:length(pairPortion)  %here it's 1 ;different percentage of paired instances
                       numpairedInst=floor(numInst*pairPortion(pairedIdx)+0.01);  % number of paired instances that have complete views
                       paired=instanceIdx(1:numpairedInst);                     %The paired instances
                       singledNumView1=ceil(0.5*(length(instanceIdx)-numpairedInst));
                       singleInstView1=instanceIdx(numpairedInst+1:numpairedInst+singledNumView1);   %the first view and second view miss half to half (Since they are mutually exclusive)
                       singleInstView2=instanceIdx(numpairedInst+singledNumView1+1:end);             %instanceIdx(numpairedInst+numsingleInstView1+1:end);

                       data = cell(1,num_views);
                       map = cell(1,num_views);
                       map{1} = [1:numpairedInst numpairedInst+1:numpairedInst+singledNumView1];
                       map{2} = [1:numpairedInst numpairedInst+singledNumView1+1:length(instanceIdx)];
                       data{1}=Xt{1}(map{1},:);                                 %View 1 of paired
                       data{2}=Xt{2}(map{2},:);                                 %View 1 of paired

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
                            options.WeightMode='Binary';
                            W{i} = constructW_cai(data{i}',options);
                            %Weight matrix constructed for each view
                        end

                      [U, V, centroidV, weights, log] = PartialGNMF(data, numClust, W, map, invMap, truthF, options, 0);
                      fprintf('alpha: %.4f delta: %.4f beta: %.4f\n',alphas(v1),deltas(v2),betas(v3));
                      [~,stats] = ComputeStats(centroidV, truthF, numClust);
                      values{v1,v2,v3} = mean(stats,2)';
                   end
               end
           end
        end
   end
end


for v1=1:length(alphas)
   for v2=1:length(deltas)
       for v3=1:length(betas)
            fprintf('alpha: %.3f delta: %.3f beta: %.3f | ac: %.4f nmi: %.4f pur: %.4f\n',alphas(v1),deltas(v2),betas(v3),values{v1,v2,v3}(1),values{v1,v2,v3}(2),values{v1,v2,v3}(3));
       end
   end
end