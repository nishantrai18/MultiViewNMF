clear all;
%clc;
addpath(genpath(('../../../../datasets/mfeat')));
loaddata='../../../../datasets/mfeat/';
datasetdir='../data/';
dataname={'mfeat'};
delimeter = ' ';
headersInFile = 0;

for idata=1:length(dataname)
    X = cell(1,5);
    dataf = strcat(loaddata,lower(dataname(idata)),'-fou'); %Just the datafile name
    cell2mat(dataf(1))
    X{1} = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    dataf = strcat(loaddata,lower(dataname(idata)),'-pix'); %Just the datafile name
    X{2} = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    dataf = strcat(loaddata,lower(dataname(idata)),'-zer'); %Just the datafile name
    X{3} = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    dataf = strcat(loaddata,lower(dataname(idata)),'-mor'); %Just the datafile name
    X{4} = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    dataf = strcat(loaddata,lower(dataname(idata)),'-fac'); %Just the datafile name
    X{5} = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    %dataf = strcat(loaddata,lower(dataname(idata)),'-kar'); %Contains negative values
    %X{6} = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    truth = [];
    for j=1:10
        for i=1:200
            truth=[truth;j];            
        end
    end
    dataf=strcat(datasetdir,lower(dataname(idata)),'bigRnSp.mat');        %Just the datafile name
    save(cell2mat(dataf(1)), 'X','truth');

    folds = [];
    numInst = size(truth);
    numInst = numInst(1);
    numPerms = 30;
    for j=1:numPerms
        folds = [folds;randperm(numInst)];
    end
    
    dataf=strcat(datasetdir,lower(dataname(idata)),'bigFolds.mat');        %Just the datafile name
    save(cell2mat(dataf(1)), 'folds');
end
    