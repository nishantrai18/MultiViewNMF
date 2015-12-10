clear all;
%clc;
addpath(genpath(('../../../../datasets/bbcsport_2v')));
addpath(genpath('../misc/'));

loaddata='../../../../datasets/bbcsport_2v/';
datasetdir='../data/';
sufName='bbcsport';
dataname={'seg1of2','seg2of2'};
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

for v1=1:length(dataname)
    for v2=v1+1:length(dataname)
        X1=[];
        X2=[];
        truth = [];
        for i=1:size(revMap,2)
            for j=1:size(revMap,3)
                if( ~isempty(revMap{v1,i,j}) && ~isempty(revMap{v2,i,j}) )
                    %fprintf('%d %d\n',revMap{v1,i},revMap{v2,i});
                    X1=horzcat(X1,views{1,v1}(:,revMap{v1,i,j}));
                    X2=horzcat(X2,views{1,v2}(:,revMap{v2,i,j}));
                    truth=[truth;i];
                end
            end
        end
        
        folds = [];
        numInst = size(truth);
        numInst = numInst(1);
        numPerms = 30;
        for j=1:numPerms
            folds = [folds;randperm(numInst)];
        end
        finName = [dataname{1,v1}(4) dataname{1,v2}(4)];
        finName = cellstr(finName);
        dataf=strcat(datasetdir,sufName,finName{1},'Folds.mat');        %Just the datafile name
        save(dataf, 'folds');
        dataf=strcat(datasetdir,sufName,finName{1},'RnSp.mat');        %Just the datafile name
        save(dataf, 'X1','X2', 'truth');
    end
end