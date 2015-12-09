function [U_final, V_final, nIter_final, objhistory_final] = GNMF_Multi(X, k, W, options, U, V)
%U, V are probably initilizations, in case they are empty we do random init

% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U*V'

%   Written by Deng Cai (dengcai AT gmail.com)
%	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter                   %Sanity checks and default value initialization
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;                                      %Graph Reg weight

Norm = 2;
NormV = 0;

[mFea,nSmp]=size(X);                                        %Dimensions

if alpha >= 0                                                   %Graph regularisation matrix
    W = alpha*W;
    DCol = full(sum(W,2));
    D = spdiags(DCol,[0],nSmp,nSmp);
    L = D - W;
    if isfield(options,'NormW') && options.NormW                %Weighted GNMF, refer to Cai et al
        D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
        L = D_mhalf*L*D_mhalf;
    end
else
    L = [];
end

selectInit = 1;
if isempty(U)                                           %If empty U i.e. need for random intializations
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;                                        %If not random initialization then do the following loop at least once
end

[U,V] = NormalizeUV(U, V, NormV, Norm);                 %Initial normalization
if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, L);                  %Storing objective history
        meanFit = objhistory*10;                                %Initialize meanFit
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, L);
        end
    end
else
    if isfield(options,'Converge') && options.Converge              %Something
        error('Not implemented!');
    end
end

tryNo = 0;
nIter = 0;
while tryNo < nRepeat   
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
        % ===================== update V ========================
        XU = X'*U;  % mnk or pk (p<<mn)
        UU = U'*U;  % mk^2
        VUU = V*UU; % nk^2
        
        if alpha > 0
            WV = W*V;
            DV = D*V;
            
            XU = XU + WV;
            VUU = VUU + DV;
        end
        
        V = V.*(XU./max(VUU,1e-10));
        
        % ===================== update U ========================
        XV = X*V;   % mnk or pk (p<<mn)
        VV = V'*V;  % nk^2
        UVV = U*VV; % mk^2
        
        U = U.*(XV./max(UVV,1e-10)); % 3mk
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, L);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, L);
                    objhistory = [objhistory newobj]; %#ok
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, L);
                        objhistory = [objhistory newobj]; %#ok
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    
    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           nIter_final = nIter;
           objhistory_final = objhistory;
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea,k));
            V = abs(rand(nSmp,k));
            
            [U,V] = NormalizeUV(U, V, NormV, Norm);
            nIter = 0;
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

[U_final,V_final] = NormalizeUV(U_final, V_final, NormV, Norm);             %Final Normalisation, from Cai et al


%==========================================================================

function [obj, dV] = CalculateObj(X, U, V, L, deltaVU, dVordU)
%Returns graph regularised loss with X = UV, L is the weight matrix (multiplied by its parameter)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    nBlock = ceil(mn/MAXARRAY);

    if mn < MAXARRAY                                    %If complete matrix can be computed in one go
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));                      %Frobenius norm
        if deltaVU                                      %By default, we do not consider this    
            if dVordU
                dV = dX'*U + L*V;
            else
                dV = dX*V;
            end
        end
    else                                                %Computing the matrix in parts
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        PatchSize = ceil(nSmp/nBlock);
        for i = 1:nBlock
            if i*PatchSize > nSmp
                smpIdx = (i-1)*PatchSize+1:nSmp;
            else
                smpIdx = (i-1)*PatchSize+1:i*PatchSize;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V;
            end
        end
    end
    if isempty(L)
        obj_Lap = 0;
    else
        obj_Lap = sum(sum((V'*L).*V'));
    end
    obj = obj_NMF+obj_Lap;                              %Remember that L was already multipled with the paramter to Graph Reg
    
function [U, V] = NormalizeUV(U, V, NormV, Norm)
%Normalisation for U and V, both L1 and L2
    K = size(U,2);
    if Norm == 2                                        %L2 normalization
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    else                                                %L1 Normalisation
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end

        