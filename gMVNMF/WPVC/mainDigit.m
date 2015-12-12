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
options.rounds = 20;
options.Gaplpha=1;                            %Graph regularisation parameter
options.alpha=0.1;
options.WeightMode='Binary';
options.gamma = 2;
options.varWeight = 0;

options.kmeans = 1;
options.beta=10;

resdir='data/result/';
datasetdir='../../partialMV/PVC/recreateResults/data/';
dataname={'mfeat'};
num_views = 2;
numClust = 10;
options.K = numClust;

ovMean = cell(1,length(dataname));
ovStd = cell(1,length(dataname));
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
   dir=strcat(resdir,cell2mat(dataname(idata)),'/'); %    train_target(idnon)=-1;   ranksvm treat weak label {-1: -1; 1:+1; 0:-1}
   %mkdir(dir);                              %Creates new folder for storing the workspace variables 
    
   multiMean = cell(1,length(pairPortion));
   multiStd = cell(1,length(pairPortion));
   for f=1:1%numFold
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances
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
                   map{1} = [singleInstView1 paired];
                   map{2} = [paired singleInstView2];
                   data{1}=X{v1}(map{1},:);                                 %View 1 of paired
                   data{2}=X{v2}(map{2},:);                                 %View 1 of paired
                   
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
                        W{i} = constructW_cai(data{i}',options);
                        %Weight matrix constructed for each view
                    end
                    
                  [U, V, centroidV, weights, log] = PartialGNMF(data, numClust, W, map, invMap, truth, options);
                                    
                  [~,stats] = ComputeStats(centroidV, truth, numClust);
    
                  for s=1:size(stats,1)
                      meanStats(s) = mean(stats(s,:));
                      stdStats(s) = std(stats(s,:));
                  end
                  multiMean{pairedIdx} = [multiMean{pairedIdx};meanStats];
                  multiStd{pairedIdx} = [multiStd{pairedIdx};stdStats];
                  %save([dir,'PVC',num2str(v1),num2str(v2),'paired_',num2str(pairPortion(pairedIdx)),'f_',num2str(f),'.mat'],'U1','U2','P2','P1','P3','objValue','F','P','R','nmi','avgent','AR','truthF');       
                  %save (filenameWithDirectory, variables)
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

