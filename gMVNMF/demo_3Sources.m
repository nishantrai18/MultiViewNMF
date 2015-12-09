%This is  a  sample demo
%Test Digits dataset
clear all;

addpath('tools/');
addpath('print/');
options = [];
options.maxIter = 100;
options.error = 1e-6;
options.nRepeat = 10;
options.minIter = 30;
options.meanFitRatio = 0.1;
options.rounds = 30;
options.K=6;
options.Gaplpha=100;                            %Graph regularisation parameter
options.WeightMode='Binary';
options.alpha = 100;

% options.kmeans means whether to run kmeans on v^* or not
% options alpha is an array of weights for different views

options.alphas = [0.01 0.01];
options.kmeans = 1;
options.beta=0;

%% read dataset

load 3sourcesbg.mat
data{1} = X1;
data{2} = X2;   
K = 6;
gnd = truth;

%% normalize data matrix

for i = 1:length(data)  %Number of views
%     dtemp=computeDistMat(data{i},2);
%     W{i}=constructW(dtemp,20);
%     data{i} = data{i} / sum(sum(data{i}));
    options.WeightMode='Binary';
    W{i}=constructW_cai(data{i}',options);                      %Incorrect call to construct weight matrix
    %Weight matrix constructed for each view
    data{i} = data{i} / sum(sum(data{i}));
end
%save('handwrittenW','W');
%%

% run 20 times
U_final = cell(1,3);
V_final = cell(1,3);
V_centroid = cell(1,3);
for i = 1:1
   [U_final{i}, V_final{i}, V_centroid{i}, log] = GMultiNMF(data, K, W,gnd, options);
   printResult( V_centroid{i}, gnd, K, options.kmeans);
   fprintf('\n');
end
