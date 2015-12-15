function [U_final, V_final, nIter_final, elapse_final, bSuccess, objhistory_final] = PartialGNMF(X, k, bigV, W, options, U, V)
%  Vo is actually the whole matrix of dimensions nMspxk, the begins, ends
%  signify the part which denotes Pc
%                                                                                   (A1, K, centroidPc, W1, optionsPGNMF, Ux, P1)

%
% 	Notation:
% 	X ... (mFea x nSmp) data matrix of one view
%       mFea  ... number of features
%       nSmp  ... number of samples
% 	k ... number of hidden factors
% 	W ... weight matrix of the affinity graph 
% 	Vo... consunsus
% 	options ... Structure holding all settings
% 	U ... initialization for basis matrix 
% 	V ... initialization for coefficient matrix 
%
%	Writen by Deng Cai (dengcai AT gmail.com) Jialu Liu (jliu64@illinois.edu) 
% 	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)
%   Further modified by Nishant Rai (nishantr AT iitk.ac.in)

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIterOrig = options.minIter;
minIter = minIterOrig-1;
meanFitRatio = options.meanFitRatio;

tots = size(X,2);
begins = options.begins;
ends = options.ends;
alpha = options.alpha;
beta=alpha*options.beta;
alphaPriv = options.alphaPriv;

v(1:begins-1) = 0;
v(begins:ends) = alpha;
v(ends+1:tots) = 0;

Vo = bigV(begins:ends,:);

diagLamda = diag(v);

Norm = 1;
NormV = 0;

[mFea,nSmp]=size(X);

bSuccess.bSuccess = 1;

W = beta*W;
DCol = full(sum(W,2));
D = spdiags(DCol,0,nSmp,nSmp);
L = D - W;

selectInit = 1;
if isempty(U)
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end

%fprintf('%.10f\n',CalculateObj(X, U, V, Vo , L, alphaPriv, options));

%[U,V] = Normalize(U, V);
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, Vo , L, alphaPriv, options);  
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, Vo , L, alphaPriv, options);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end

%%Modify all the update rules here
%workspace

%fprintf('%.10f\n',CalculateObj(X, U, V, Vo , L, alphaPriv, options));

tryNo = 0;
while tryNo < nRepeat   
    tmp_T = cputime;
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    nStepTrial = 0;
    %disp a
    while(maxErr > differror)
        % ===================== update V ========================
        XU = X'*U;  % mnk or pk (p<<mn)
        UU = U'*U;  % mk^2
        VUU =V*UU; % nk^2
        if beta > 0
            WV = W*V;
            DV = D*V;            
            XU = XU + WV;
            VUU = VUU + DV;
        end
        XU = XU + diagLamda*bigV;
        VUU = VUU + diagLamda*V;
        
        V = V.*(XU./max(VUU,1e-10));
    
        % ===================== update U ========================
        XV = X*V; 
        Vc = V(begins:ends,:);
        VV = V'*V;
        UVV = U*VV;
        
        VV_ = repmat(diag(Vc'*Vc)' .* sum(U, 1), mFea, 1);
        tmp = sum(Vc.*Vo);
        VVo = repmat(tmp, mFea, 1);
        
        XV = XV + alpha * VVo;
        UVV = UVV + alpha * VV_;

        U = U.*(XV./max(UVV,1e-10)); 
        
        [U,V] = Normalize(U, V);
        nIter = nIter + 1;
        %fprintf('%.10f\n',CalculateObj(X, U, V, Vo , L, alphaPriv, options));
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, Vo , L, alphaPriv, options);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, Vo , L, alphaPriv, options);
                    objhistory = [objhistory newobj]; 
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, Vo , L, alpha, options);
                        objhistory = [objhistory newobj]; 
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
    
    elapse = cputime - tmp_T;

    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        elapse_final = elapse;
        objhistory_final = objhistory;
        bSuccess.nStepTrial = nStepTrial;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           nIter_final = nIter;
           objhistory_final = objhistory;
           bSuccess.nStepTrial = nStepTrial;
           if selectInit
               elapse_final = elapse;
           else
               elapse_final = elapse_final+elapse;
           end
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea,k));
            V = abs(rand(nSmp,k));
            [U,V] = Normalize(U, V);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
            
        end
    end
end
%%

objhistory_final;

nIter_final = nIter_final + minIterOrig;

%fprintf('%.10f\n',CalculateObj(X, U, V, Vo , L, alphaPriv, options));
[U_final, V_final] = Normalize(U_final, V_final);
%fprintf('%.10f\n',CalculateObj(X, U, V, Vo , L, alphaPriv, options));

%==========================================================================

function [obj, dV] = CalculateObj(X, U, V, Vo, L, alphaPriv, options)
    dV = [];
    begins = options.begins;
    ends = options.ends;
    maxM = 62500000;
    [mFea, nSmp] = size(X);
    mn = numel(X);
    nBlock = floor(mn*3/maxM);

    if mn < maxM
        dX = (U*V'-X);
        obj_NMF = sum(sum(dX.^2));
    else
        obj_NMF = 0;
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
        end
    end
    tmp = (V(begins:ends,:)-Vo);
    obj_Vo = sum(sum(tmp.^2));
    obj_Lap=sum(sum((V'*L).*V'));
    dX = (U*V'-X);
    obj_NMF = sum(sum(dX.^2));
    obj = obj_NMF+ (alphaPriv)*obj_Vo+obj_Lap;

function [U, V] = Normalize(U, V)
    [U,V] = NormalizeUV(U, V, 0, 1);

function [U, V] = NormalizeUV(U, V, NormV, Norm)
    nSmp = size(V,1);
    mFea = size(U,1);
    if Norm == 2
        if NormV
            norms = sqrt(sum(V.^2,1));
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sqrt(sum(U.^2,1));
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    else
        if NormV
            norms = sum(abs(V),1);
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sum(abs(U),1);
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = bsxfun(@times, V, norms);
        end
    end

        