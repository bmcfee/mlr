function D = setDistanceDODMKL(X, W, Ifrom, Ito)
%
% D = setDistanceDODMKL(X, W, Ifrom, Ito)
%
%   X       = d-by-n data matrix
%   W       = m-by-m-by-n PSD matrix
%   Ifrom   = k-by-1 vector of source points
%   Ito     = j-by-1 vector of destination points
%
%   D = n-by-n matrix of squared euclidean distances from Ifrom to Ito
%       D is sparse, and only the rows corresponding to Ifrom and
%       columns corresponding to Ito are populated.

    [d,n,m]       = size(X);
    L           = W.^0.5;
   
    D = 0;
    for i = 1:m
        for j = i:m
            Vfrom       = bsxfun(@times, squeeze(L(i,j)), X(:,Ifrom,i) + X(:,Ifrom,j));

            if nargin == 4
                Vto     = bsxfun(@times, squeeze(L(i,j)), X(:,Ito,i) + X(:,Ito,j));
            else
                Vto     = bsxfun(@times, squeeze(L(i,j)), X(:,:,i) + X(:,:,j));
                Ito     = 1:n;
            end

            if i == j
                s = 0.5;
            else
                s = 1;
            end

            D = D + s * distToFrom(n, Vto, Vfrom, Ito, Ifrom);
        end
    end

end
