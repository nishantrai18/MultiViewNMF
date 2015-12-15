clear all;
%clc;
addpath(genpath(('../../../../datasets/3sources')));
addpath(genpath('../misc/'));

loaddata='../../../../datasets/3sources/';
datasetdir='../data/';
sufName='3sources';
truthFile='.disjoint.csv';
dataname={'bbc','guardian','reuters'};
delimeter = ' ';
headersInFile = 1;
views=cell(1,length(dataname));
ids=cell(1,length(dataname));
revMap=cell(length(dataname),500);     %500: A large number

numClust = 6;
dataf = strcat(loaddata,sufName,'.disjoint.csv');
clustMat = csvread(dataf);
truths = cell(1,500);
for i=1:numClust
    for j=1:size(clustMat,2)
        if(clustMat(i,j) ~= 0)
            truths{1,clustMat(i,j)} = i;
        end
    end
end

for idata=1:length(dataname)  
    dataf = strcat(loaddata,sufName,'_',lower(dataname(idata)),'.mtx'); %Just the datafile name
    cell2mat(dataf(1))
    X1 = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    X1 = X1.data;
    X1 = readsparse(X1);
    dataf = strcat(loaddata,sufName,'_',lower(dataname(idata)),'.docs');             %Just the datafile name
    X2 = importdata(cell2mat(dataf(1)), delimeter, 0);
    for i=1:length(X2)
        revMap{idata,X2(i)} = i;
    end
    views{idata} = X1;
    ids{idata} = X2;
end


X=cell(1,length(dataname));
truth = [];
for i=1:size(revMap,2)
    fg = 1;
    for v=1:length(dataname)
        if(isempty(revMap{v,i}))
            fg = 0;
            break;
        end
    end
    if fg==0
        continue;
    end
    for v=1:length(dataname)
        X{v}=horzcat(X{v},views{1,v}(:,revMap{v,i}));
    end
    truth=[truth;truths(i)];
end
truth=cell2mat(truth);

folds = [];
numInst = size(truth);
numInst = numInst(1);
numPerms = 30;
for j=1:numPerms
    folds = [folds;randperm(numInst)];
end
finName = ['3v'];
finName = cellstr(finName);
dataf=strcat(datasetdir,sufName,finName{1},'bigFolds.mat');        %Just the datafile name
save(dataf, 'folds');
dataf=strcat(datasetdir,sufName,finName{1},'bigRnSp.mat');        %Just the datafile name
save(dataf, 'X', 'truth');
