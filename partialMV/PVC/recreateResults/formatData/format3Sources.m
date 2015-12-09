clear all;
%clc;
addpath(genpath(('../../../../datasets/3sources')));
loaddata='../../../../datasets/3sources/';
datasetdir='../data/';
sufName='3sources';
dataname={'bbc','guardian','reuters'};
delimeter = ' ';
headersInFile = 0;

dataf = strcat(loaddata,lower(dataname(idata)),'-fou'); %Just the datafile name
cell2mat(dataf(1))
X1 = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
dataf = strcat(loaddata,lower(dataname(idata)),'-pix'); %Just the datafile name
X2 = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
truth = [];
for j=1:10
    for i=1:200
        truth=[truth;j];            
    end
end
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
    