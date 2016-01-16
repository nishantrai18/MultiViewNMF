clear;                      %Remove all variables from the workspace
%clc;
 
addpath(genpath('../../partialMV/PVC/recreateResults/measure/'));
addpath(genpath('../../partialMV/PVC/recreateResults/misc/'));
addpath(genpath('../../partialMV/PVC/recreateResults/data/'));
addpath(genpath('../../sampleCodes/exact_alm_rpca/inexact_alm_rpca/'));
addpath(genpath('../../sampleCodes/exact_alm_rpca/inexact_alm_rpca/PROPACK/'));
addpath(genpath('../print/'));

options = [];
options.maxIter = 75;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 50;
options.WeightMode='Binary';
options.varWeight = 0;
options.kmeans = 1;

options.Gaplpha=0;                            %Graph regularisation parameter
options.alpha=0;
options.delta = 0.1;
options.beta = 0;
options.gamma = 2;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
dataname={'mfeatbig'};
num_views = 5;
numClust = 10;
options.K = numClust;

ovMean = cell(1,length(dataname));
ovAvgStd = cell(1,length(dataname));
pairPortion=[0,0.1,0.3,0.5,0.7,0.9];                  %The array which contains the PER
pairPortion = 1 - (pairPortion);
for idata=1:length(dataname)  
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');        %Just the datafile name
    datafname=cell2mat(dataf(1));       
    load (datafname);                                           %Loading the datafile
    
    %X1=readsparse(X1);                                         %Loading a sparse matrix i.e. on the basis of edges                  
    %X2=readsparse(X2);
    %% normalize data matrix
    for i=1:num_views
        X{i} = X{i}/sum(sum(X{i}));
    end
    %%
    %X should be row major i.e. rows are the data points
    
    load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
   
    [numFold,numInst]=size(folds);                                   %numInst : numInstances
    dir=strcat(resdir,cell2mat(dataname(idata)),'/'); %    train_target(idnon)=-1;   ranksvm treat weak label {-1: -1; 1:+1; 0:-1}
    
    multiMean = cell(1,length(pairPortion));
    multiStd = cell(1,length(pairPortion));
   
    views = cell(1,num_views);
    truths = cell(1,10);
    for f=1:1
           instanceIdx=folds(f,:);
           truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances                
           truths{f} = truthF;
           for i=1:num_views
                Xt{i} = X{i}(instanceIdx,:);
           end
           meanStats = [];
           stdStats = [];
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
               if (~isempty(Xfilled{pairedIdx}))
                   Xfilled{pairedIdx} = inexact_alm_rpca(Xfilled{pairedIdx}')';
               end
               rear = 0;
               sigma = [];
               lambda = [];
               for i=1:num_views
                    tmpMat = Xfilled{pairedIdx}(:,rear+1:rear+size(Xt{i},2));
                    %views{i,f,pairedIdx} = tmpMat;
                    views{i} = [Xt{i}(1:numpairedInst,:); tmpMat];
                    rear = rear+size(Xt{i},2);
                    sigma(end+1) = optSigma(views{i});
                    lambda(end+1) = 0.01;
               end
               
			for i = 1:length(X)
            		data{i} = (views{i}(instanceIdx,:))';              %Column major
            		options.WeightMode='Binary';
            		W{i} = constructW_cai(data{i}',options);           %Need row major
        		end
        		
        		[U, V, centroidV, weights, log] = GMultiNMF(views, options.K, W, truthF, options);
 
               [~,stats] = ComputeStats(centroidV, truthF, numClust);
               stats = mean(stats,2);
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
       list = std(multiMean{t});
       ovAvgStd{idata} = [ovAvgStd{idata}; list];
   end
       ovMean{idata}
       ovAvgStd{idata}
end
