function [Pc_final, U1_final, U2_final, nIter_final, objhistory_final] = UpdatePcU(X1, X2, P1, Pc, P2, k, W1, W2, options, U1, U2)
% options contain numCom, which tell which part is common amongst the views
%X1 is expected to have the view at the end, while X2 is expected to have
%the common view in the beginning
%Same goes for W1 and W2

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
%	Modified by Nishant Rai (nishantr AT iitk DOT ac DOT in)

if min(min(X1)) < 0                                              %Sanity Checks
    error('Input should be nonnegative!');
end

if min(min(X2)) < 0                                              %Sanity Checks
    error('Input should be nonnegative!');
end

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

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter                   %Sanity checks and default value initialization
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;
alpha = options.alpha;           %Graph Reg weight

Norm = 2;
NormV = 0;

numCom = options.numCom;

%workspace

X1c = X1(:,size(X1,2)-numCom+1:end);
X2c = X2(:,1:numCom);

nSmp=size(X1,2);                                        %Dimensions
W = W1;
DCol = full(sum(W,2));
D = spdiags(DCol,0,nSmp,nSmp);
L1 = D - W;
nSmp=size(X2,2);
W = W2;
DCol = full(sum(W,2));
D = spdiags(DCol,0,nSmp,nSmp);
L2 = D - W;

W1 = W1(end-numCom+1:end,end-numCom+1:end);
W2 = W2(1:numCom,1:numCom);
nSmp=numCom;
W = (W1+W2);
DCol = full(sum(W,2));
D = spdiags(DCol,0,nSmp,nSmp);
L = D - W;

nRepeat = 1;
selectInit = 1;
if isempty(U1)                                           %If empty i.e. need for random intializations
    U1 = abs(rand(size(X1,1),k));
    nRepeat = options.nRepeat;
end
if isempty(U2)                                           %If empty i.e. need for random intializations
    U2 = abs(rand(size(X2,1),k));
    nRepeat = options.nRepeat;
end

%[U1,Pc] = NormalizeUV(U1, Pc, NormV, Norm);                 %Initial normalization
%[U2,Pc] = NormalizeUV(U2, Pc, NormV, Norm);                 %Initial normalization

objhistory = 0;
if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = ObjectivePc(Pc, X1, X2, U1, U2, L);                  %Storing objective history
        meanFit = objhistory*10;                                %Initialize meanFit
    end
end

tryNo = 0;
nIter = 0;

%fprintf('%.10f SCORE 1\n',norm(X1-U1*P1','fro')+ norm(X2-U2*P2','fro')+ sum(sum((P1'*L1)*P1)) + sum(sum((P2'*L2)*P2)));
%fprintf('%.10f SCORE 2\n',ObjectivePc(Pc, X1c, X2c, U1, U2, L));

while tryNo < nRepeat   
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
        % ===================== update Pc ========================
        M = X1c'*U1 + X2c'*U2;
        N = U1'*U1 + U2'*U2;
        PcN = Pc*N;
        if alpha > 0
            WPc = W*Pc;
            DPc = D*Pc;            
            M = M + WPc;
            PcN = PcN + DPc;
        end
        Pc = Pc.*(M./max(PcN,1e-10));
        
        % ===================== update U's ========================
        PP1 = P1'*P1;
        XP1 = X1*P1;
        UPP1 = U1*PP1;
        U1 = U1.*(XP1./max(UPP1,1e-10));
        
        PP2 = P2'*P2;
        XP2 = X2*P2;
        UPP2 = U2*PP2;
        U2 = U2.*(XP2./max(UPP2,1e-10));

        %ObjectivePc(Pc, X1c, X2c, U1, U2, L)
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = ObjectivePc(Pc, X1c, X2c, U1, U2, L);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = ObjectivePc(Pc, X1c, X2c, U1, U2, L);
                    objhistory = [objhistory newobj]; %#ok
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                    end
                end
            end
        end
    end
    
    if tryNo == 1
        U1_final = U1;
        U2_final = U2;
        Pc_final = Pc;
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
            U1_final = U1;
            U2_final = U2;
            Pc_final = Pc;
            nIter_final = nIter;
            objhistory_final = objhistory;
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U1 = abs(rand(size(X1,1),k));
            U2 = abs(rand(size(X2,1),k));
            Pc = abs(rand(nSmp,k));
            %[U1,Pc] = NormalizeUV(U1, Pc, NormV, Norm);                 %Initial normalization
            %[U2,Pc] = NormalizeUV(U2, Pc, NormV, Norm);                 %Initial normalization
            nIter = 0;
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            U1_final = U1;
            U2_final = U2;
            Pc_final = Pc;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

%fprintf('%.10f SCORE 3\n',ObjectivePc(Pc, X1c, X2c, U1, U2, L));
P1 = [P1(1:end-numCom,:);Pc_final];
P2 = [Pc_final;P2(numCom+1:end,:)];
%fprintf('%.10f SCORE 4\n',norm(X1-U1_final*P1','fro')+ norm(X2-U2_final*P2','fro')+ sum(sum((P1'*L1)*P1)) + sum(sum((P2'*L2)*P2)));

%[U_final,V_final] = NormalizeUV(U_final, V_final, NormV, Norm);             %Final Normalisation, from Cai et al

%==========================================================================

function [obj] = ObjectivePc(Pc, X1, X2, U1, U2, L)
    val1 = CalculateObj(X1, U1, Pc, L);
    val2 = CalculateObj(X2, U2, Pc, []);
    obj = val1 + val2;
    
function [obj, dV] = CalculateObj(X, U, V, L, deltaVU, dVordU)
%Returns graph regularised loss with X = UV, L is the weight matrix (multiplied by its parameter)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    nBlock = ceil(mn/MAXARRAY);

    if mn < MAXARRAY                                    %If complete matrix can be computed in one go
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));                      %Frobenius norm
    else                                                %Computing the matrix in parts
        obj_NMF = 0;
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