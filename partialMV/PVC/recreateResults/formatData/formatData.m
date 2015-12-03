clear all;
clc;
addpath(genpath(('../../../../datasets/webKB')));
loaddata='../../../../datasets/webKB/';
datasetdir='../data/';
dataname={'Cornell','Texas','Washington','Wisconsin'};
delimeter = ' ';
headersInFile = 2;

for idata=1:length(dataname)  
    dataf = strcat(loaddata,dataname(idata),'/',dataname(idata),'/',lower(dataname(idata)),'_cites.mtx'); %Just the datafile name
    X1 = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    X1 = X1.data;
    dataf = strcat(loaddata,dataname(idata),'/',dataname(idata),'/',lower(dataname(idata)),'_content.mtx'); %Just the datafile name
    X2 = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    X2 = X2.data;
    dataf = strcat(loaddata,dataname(idata),'/',dataname(idata),'/',lower(dataname(idata)),'_act.txt'); %Just the datafile name
    truth = importdata(cell2mat(dataf(1)), delimeter, 0);
    
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
    