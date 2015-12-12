function [U_final, V_final, nIter_final, objhistory_final] = GNMF(X, k, W, options, U, V)

% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%               options.alpha ... the graph regularization parameter. 
%                                 [default: 100]
%                                 alpha = 0, GNMF boils down to the ordinary NMF. 
%                                 
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Xiaoyun Wu, and Jiawei Han. "Non-negative
% Matrix Factorization on Manifold", Proc. 2008 Int. Conf. on Data Mining
% (ICDM'08), Pisa, Italy, Dec. 2008. 
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
% 8, pp. 1548-1560, 2011.  

%   Written by Deng Cai (dengcai AT gmail.com)
%	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)

%if min(min(X)) < 0                                              %Sanity Checks
%    error('Input should be nonnegative!');
%end

if ~isfield(options,'error')                                    %Default values of the errors
    options.error = 1e-5;
end
if ~isfield(options, 'maxIter')                                 %Check again
    options.maxIter = [];
end

if ~isfield(options,'nRepeat')
    options.nRepeat = 10;
end

if ~isfield(options,'minIter')
    options.minIter = 30;
end

if ~isfield(options,'meanFitRatio')
    options.meanFitRatio = 0.1;
end

if ~isfield(options,'alpha')
    options.alpha = 10;
end

nSmp = size(X,2);                           %Number of data points

if isfield(options,'alpha_nSmp') && options.alpha_nSmp              %Not relevant for now
    options.alpha = options.alpha*nSmp;    
end

if isfield(options,'weight') && strcmpi(options.weight,'NCW')       %Not relevant for now
    feaSum = full(sum(X,2));
    D_half = X'*feaSum;
    X = X*spdiags(D_half.^-.5,0,nSmp,nSmp);
end

if ~isfield(options,'Optimization')                                 %Default value
    options.Optimization = 'Multiplicative';
end

if ~exist('U','var')                                            %Initialise empty U, V if no parameters given
    U = [];
    V = [];
end

switch lower(options.Optimization)
    case {lower('Multiplicative')}                                      %Only multiplicative updates
        [U_final, V_final, nIter_final, objhistory_final] = GNMF_Multi(X, k, W, options, U, V);
    otherwise
        error('optimization method does not exist!');
end


    
        