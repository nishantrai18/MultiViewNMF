function [ac, stats] = ComputeStats(X, label, K, kmeansFlag)
    if (~exist('kmeansFlag','var'))
        kmeansFlag = 1;
    end
    fprintf('running k-means...\n');
    kmeansIter = 10;
    for i=1:kmeansIter
        if kmeansFlag == 1
            indic = litekmeans(X, K, 'Replicates',20);
        else
            [~, indic] = max(X, [] ,2);
        end
        result = bestMap(label, indic);
        [ac(i), nmi_value(i), cnt(i)] = CalcMetrics(label, indic);
        [Pri(i)] = purity(label, indic);
    end
    stats = [];
    stats = [stats;ac];
    stats = [stats;nmi_value];
    %stats = [stats;cnt];
    stats = [stats;Pri];
    disp(sprintf('ac: %0.3f,  nmi:%0.3f,  purity:%.3f,  errors: %d/%d\n', ...
                mean(ac), mean(nmi_value), mean(Pri), round(mean(cnt)), length(label)));    
    ac = mean(ac);
end