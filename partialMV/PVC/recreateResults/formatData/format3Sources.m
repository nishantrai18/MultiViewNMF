clear all;
%clc;
addpath(genpath(('../../../../datasets/3sources')));
loaddata='../../../../datasets/3sources/';
datasetdir='../data/';
sufName='3sources';
dataname={'bbc','guardian','reuters'};
delimeter = ' ';
headersInFile = 1;
views=cell(1,length(dataname));
ids=cell(1,length(dataname));
revMap=cell(length(dataname),500);     %500: A large number

for idata=1:length(dataname)  
    dataf = strcat(loaddata,sufName,'_',lower(dataname(idata)),'.mtx'); %Just the datafile name
    cell2mat(dataf(1))
    X1 = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    X1 = X1.data;
    dataf = strcat(loaddata,sufName,'_',lower(dataname(idata)),'.docs');             %Just the datafile name
    X2 = importdata(cell2mat(dataf(1)), delimeter, 0);
    for i=1:length(X2)
        revMap{idata,X2(i)} = i;
    end
    views{idata} = X1;
    ids{idata} = X2;
end

v1=1;
v2=2;
X1=[];
X2=[];
truth = [];
for i=1:size(revMap,2)
    if( ~isempty(revMap{v1,i}) && ~isempty(revMap{v2,i}) )
        fprintf('%d %d\n',revMap{v1,i},revMap{v2,i});
        X1=[X1;views{v1}(:,revMap{v1,i})];
        X2=[X2;views{v2}(:,revMap{v2,i})];        
    end
end
dataf=strcat(datasetdir,sufName,'bg','RnSp.mat');        %Just the datafile name
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
