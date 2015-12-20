%This is  a  sample demo
%Test Digits dataset
clear all;

addpath('../tools/');
addpath('../print/');
addpath('../')
options = [];
options.maxIter = 100;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 30;
options.K=10;
options.WeightMode='Binary';

% options.kmeans means whether to run kmeans on v^* or not
% options alpha is an array of weights for different views

options.varWeight = 0;
options.kmeans = 1;
options.beta = 10;
options.gamma = 2;
options.Gaplpha=1;
options.delta = 0.1;
options.alpha = 0.1;

%% read dataset
%load handwritten.mat
%data{1} = fourier';         %Currently column major
%data{2} = pixel';   

dataset = '../../partialMV/PVC/recreateResults/data/mfeatRnSp.mat';
load(dataset);
data{1} = X1';
data{2} = X2';
gnd = truth;

%Need column major data here
% Create map
numData = size(data{1},2);      %Get the total number of data points
numView = length(data);
map = cell(1,numView);
for i=1:numView
    for j=1:size(data{i},2)
        map{i}(j) = j;
    end
end

%Create invMap
invMap = cell(1,numData);
for i=1:numView
    for j=1:size(map{i},2)
        id = map{i}(j);
        invMap{1,id} = [invMap{1,id};[i,j]];
    end
end

%{
dataset = '../../partialMV/PVC/recreateResults/data/mfeatbigRnSp.mat';
load(dataset);
data = X;
gnd = truth;
%}
K = 10;

%% normalize data matrix
for i = 1:length(data)
    data{i} = data{i};
    data{i} = data{i} / sum(sum(data{i}));
    options.WeightMode='Binary';
    W{i}=constructW_cai(data{i}',options);           %Need row major
end
%%

% run 20 times
U_final = cell(1,3);
V_final = cell(1,3);
V_centroid = cell(1,3);
for i = 1:1
   [U_final{i}, V_final{i}, V_centroid{i}, weights, log] = PartialGNMF(data, K, W, map, invMap, gnd, options);
   %printResult( V_centroid{i}, gnd, K, options.kmeans);
   ComputeStats(V_centroid{i}, gnd, options.K, 2, 1);
   fprintf('\n');
end
