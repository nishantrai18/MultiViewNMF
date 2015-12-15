clear;
%clc;
addpath(genpath(('../../../../datasets/bbcsport_2v')));
addpath(genpath('../misc/'));

loaddata='../../../../datasets/bbcsport_2v/';
datasetdir='../data/';
sufName='bbcsport';
dataname={'seg1of3','seg2of3','seg3of3'};
delimeter = ' ';
headersInFile = 1;
numClust = 5;

views=cell(1,length(dataname));
revMap=cell(length(dataname),numClust,300);     %300: A large number


for idata=1:length(dataname)  
    dataf = strcat(loaddata,sufName,'_',lower(dataname(idata)),'.mtx'); %Just the datafile name
    cell2mat(dataf(1))
    X1 = importdata(cell2mat(dataf(1)), delimeter, headersInFile);
    X1 = X1.data;
    X1 = readsparse(X1);
    dataf = strcat(loaddata,sufName,'_',lower(dataname(idata)),'.docs');             %Just the datafile name
    cell2mat(dataf(1))
    fileID = fopen(cell2mat(dataf(1)));
    X2 = textscan(fileID,'%d %d');
    fclose(fileID);
    for i=1:length(X2{1})
        revMap{idata,X2{1,1}(i),X2{1,2}(i)} = i;
    end
    views{idata} = X1;
end

X=cell(1,length(dataname));
truth = [];
for i=1:size(revMap,2)
    for j=1:size(revMap,3)
        fg = 1;
        for v=1:length(dataname)
            if(isempty(revMap{v,i,j}))
                fg = 0;
                break;
            end
        end
        if fg==0
            continue;
        end
        for v=1:length(dataname)
            X{v}=horzcat(X{v},views{1,v}(:,revMap{v,i,j}));
        end
        truth=[truth;i];
    end
end

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
