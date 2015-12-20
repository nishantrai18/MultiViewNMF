%This is  a  sample demo
%Test Digits dataset
clear;

%The best results with varWeights are found for Gaplpha = 1, alpha = 0.1, delta = 0.1,
% gamma = 2 (or something else), beta = 10
% In GMultiNMF, don't comment out the optionsPerVIewNMF part (In updating
% the view matrices U,V)

addpath('../tools/');
addpath('../print/');
addpath('../')


options = [];
options.maxIter = 100;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 20;
options.WeightMode='Binary';
options.varWeight = 1;
options.kmeans = 1;

options.Gaplpha=1;                            %Graph regularisation parameter
options.alpha=0.1;
options.delta = 0.1;
options.beta=10;
options.gamma = 2;

%% read dataset
%load handwritten.mat
%data{1} = fourier';
%data{2} = pixel';   

dataset = '../../partialMV/PVC/recreateResults/data/mfeatbigRnSp.mat';
load(dataset);
%data{1} = X1;
%data{2} = X2;
data = X;
gnd = truth;

options.K = 10;

%% normalize data matrix
for i = 1:length(data)
    data{i} = data{i}';
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
   [U_final{i}, V_final{i}, V_centroid{i}, weights, log] = GMultiNMF(data, options.K, W, gnd, options);
   ComputeStats(V_centroid{i}, gnd, options.K);
   fprintf('\n');
end
