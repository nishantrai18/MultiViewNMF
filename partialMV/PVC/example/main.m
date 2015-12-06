clear all;                      %Remove all variables from the workspace
clc;                            %Gives clean screen
 
addpath(genpath('measure/'));
addpath(genpath('misc/'));
addpath(genpath(('../code/')));

resdir='data/result/';
datasetdir='data/';
dataname={'cornell'};
num_views =2;
numClust = 5;

pairPortion=[0.5];                  %The array which contains the PER
for idata=1:length(dataname)  
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');        %Just the datafile name
    datafname=cell2mat(dataf(1));       
    load (datafname);                                           %Loading the datafile
    Xf1=readsparse(X1);                                         %Loading a sparse matrix i.e. on the basis of edges                  
    Xf2=readsparse(X2);
    X{1} =Xf1;                                                  %View 1
    X{2} =Xf2;                                                  %View 2
 
    load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); %Loading the variable folds
    [numFold,numInst]=size(folds);                                   %numInst : numInstances
    %Folds may refer to permutation of the instances
    
    dir=strcat(resdir,cell2mat(dataname(idata)),'/'); %    train_target(idnon)=-1;   ranksvm treat weak label {-1: -1; 1:+1; 0:-1}
    mkdir(dir);                              %Creates new folder for storing the workspace variables 
    
    for f=1:1%numFold
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);                                  %Contains the true clusters of the instances
        for v1=1:num_views
           for v2=v1+1:num_views
               if v1==v2
               continue;
               end
           
               for pairedIdx=1:length(pairPortion)  %here it's 1 ;different percentage of paired instances
                   numpairedInst=floor(numInst*pairPortion(pairedIdx));  % number of paired instances that have complete views
                   paired=instanceIdx(1:numpairedInst);                     %The paired instances
                   singledNumView1=ceil(0.5*(length(instanceIdx)-numpairedInst));
                   singleInstView1=instanceIdx(numpairedInst+1:numpairedInst+singledNumView1);   %the first view and second view miss half to half (Since they are mutually exclusive)
                   singleInstView2=instanceIdx(numpairedInst+singledNumView1+1:end);             %instanceIdx(numpairedInst+numsingleInstView1+1:end);
                   xpaired=X{v1}(paired,:);                                 %View 1 of paired
                   ypaired=X{v2}(paired,:);                                 %View 1 of single
                   xsingle=X{v1}(singleInstView1,:);                        %View 2 of paired
                   ysingle=X{v2}(singleInstView2,:);                        %View two of single
         
                  option.lamda=1e-2;                                        %Sparsity parameter for Lasso norm
                  option.latentdim=numClust;
      
                  [U1 U2 P2 P1 P3 objValue F P R nmi avgent AR] = PVCclust(xpaired,ypaired,xsingle,ysingle,numClust,truthF,option);
                  %[5 unknowns objectiveValue 6 stats] = func([X12][2],X1,X2, numClust,trueClusts,Parameters); 
                  save([dir,'PVC',num2str(v1),num2str(v2),'paired_',num2str(pairPortion(pairedIdx)),'f_',num2str(f),'.mat'],'U1','U2','P2','P1','P3','objValue','F','P','R','nmi','avgent','AR','truthF');       
                  %save (filenameWithDirectory, variables)
        end
      end
    end
end
end

       
         
 