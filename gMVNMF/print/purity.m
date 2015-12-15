function [pure] = purity(T,H)
      if length(T) ~= length(H)
        size(T)
        size(H)
      end
      if (min(T)==0)
      	T = T+1;
      end    
      if (min(H)==0)
      	H = H+1;
      end    
      num = length(T);
      N = max(unique(T))+2;
      clustMat = cell(N);
      for i=1:N
          for j=1:N
              clustMat{i,j}=0;
          end
      end
      for i=1:num
        clustMat{H(i),T(i)} = clustMat{H(i),T(i)} + 1;
      end
      mat = cell2mat(clustMat);
      pure = 0;
      for i=1:N
        pure = pure + max(mat(i,:));  
      end
      pure = pure/num;
