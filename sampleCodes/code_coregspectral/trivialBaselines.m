function [P nmi pur] = trivialBaselines(X, num_views, numClust, truth)

    if (min(truth)==0)
        truth = truth + 1;
    end
    
    Xbig = [];
    for i=1:num_views
        if (size(X{i},2) >= numClust)
            [W{i},H{i}] = nnmf(X{i},numClust);
        else
            W{i} = X{i};
        end
        Xbig = [Xbig X{i}];
        W{i} = Normalize(W{i});
    end
    [Wc,Hc] = nnmf(Xbig,numClust);
    Wc = Normalize(Wc);
    Pi = cell(1,num_views);
    nmii = cell(1,num_views);
    pure = cell(1,num_views);
    C = cell(1,num_views);
    
    for i=1:20
        %view1
        for j=1:num_views
            C = kmeans(W{j},numClust,'EmptyAction','drop');
            [~,Pi{j}(i)] = compute_f(truth+1,C);
            [~, nmii{j}(i)] = compute_nmi(truth,C);
            pure{j}(i)=purity(truth,C);     
        end
        %joint
        C = kmeans(Wc, numClust, 'EmptyAction','drop');
        [~,Pic(i)] = compute_f(truth+1,C);
        [~, nmiic(i)] = compute_nmi(truth,C);
        purec(i)=purity(truth,C);     
    end
    for i=1:num_views
        P(i) = mean(Pi{i});
        nmi(i) = mean(nmii{i});
        pur(i) = mean(pure{i});
    end
    pc = mean(Pic);
    nmic = mean(nmiic);
    purc = mean(purec);
    maxp = 0;
    maxnmi = 0;
    maxpur = 0;
    minp = 100;
    minnmi = 100;
    minpur = 100;
    %1: Best View
    %2: Worst View
    for i=1:num_views
        if (P(i) > maxp)
            maxp = P(i);
            maxnmi = nmi(i);
            maxpur = pur(i);
        end
        if (P(i) < minp)
            minp = P(i);
            minnmi = nmi(i);
            minpur = pur(i);
        end
    end
    P = [];
    nmi = [];
    pur = [];
    P(1) = maxp; nmi(1) = maxnmi; pur(1) = maxpur;
    P(2) = minp; nmi(2) = minnmi; pur(2) = minpur;
    P(3) = pc; nmi(3) = nmic; pur(3) = purc;
    fprintf('nmi_1 = %f, nmi_2 = %f, nmi = %f\n', nmi(1), nmi(2), nmi(3));
    fprintf('Pur_1 = %f, Pur_2 = %f, Pur = %f\n', pur(1), pur(2), pur(3));
    %fprintf('F_1 = %f(%f), F_2 = %f(%f), F = %f(%f)\n', F(1), std(F1i),F(2), std(F2i),F(3), std(Fi));
    fprintf('P_1 = %f, P_2 = %f, P = %f\n', P(1), P(2), P(3));
    %fprintf('R_1 = %f(%f), R_2 = %f(%f), R = %f(%f)\n', R(1), std(R1i),R(2), std(R2i),R(3), std(Ri));
    %fprintf('Entropy_1 = %f(%f), Entropy_2 = %f(%f), Entropy = %f(%f)\n', avgent(1), std(avgent1i),avgent(2), std(avgent2i),avgent(3), std(avgenti));
    %fprintf('AR_1 = %f(%f), AR_2 = %f(%f), AR = %f(%f)\n', AR(1), std(AR1i),AR(2), std(AR2i),AR(3), std(ARi));
    
function  [K] = renormalize(K)
    
    mn = min(min(K));
    mx = max(max(K));
    if (mn < 0)
        %K = (K - mn) / (mx-mn);
        %K = (K+K')/2;
        K = K - mn;
    end
    
function [VNorm] = Normalize(V)
    UPI = V;
    norm_mat = repmat(sqrt(sum(UPI.*UPI,2)),1,size(UPI,2));
    %avoid divide by zero
    for i=1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    VNorm= UPI./norm_mat;
