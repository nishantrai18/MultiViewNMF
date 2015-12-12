clear all;
%clc;
datasetdir='../data/';
dataname={'orl'};

for idata=1:length(dataname)  
    load 'orlHog.mat'
    X1 = feat.fea;
    X2 = feat.hogs;
    X1(X1<0) = 0;
    X2(X2<0) = 0;
    truth = feat.gnd;
    dataf=strcat(datasetdir,lower(dataname(idata)),'RnSp.mat');        %Just the datafile name
    save(cell2mat(dataf(1)), 'X1','X2', 'truth');

    folds = [];
    numInst = size(truth);
    numInst = numInst(1);
    numPerms = 30;
    for j=1:numPerms
        folds = [folds;randperm(numInst)];
    end
    
    dataf=strcat(datasetdir,lower(dataname(idata)),'Folds.mat');        %Just the datafile name
    save(cell2mat(dataf(1)), 'folds');
end    