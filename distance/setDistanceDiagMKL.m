function D = setDistanceDiagMKL(X, W, Ifrom, Ito)
%
% D = setDistanceDiagMKL(X, W, Ifrom, Ito)
%
%   X       = d-by-n data matrix
%   W       = d-by-1 PSD matrix
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
        Vfrom       = bsxfun(@times, L(:,i), X(:,Ifrom,i));

        if nargin == 4
            Vto     = bsxfun(@times, L(:,i), X(:,Ito,i));
        else
            Vto     = bsxfun(@times, L(:,i), X(:,:,i));
            Ito     = 1:n;
        end

        D = D + distToFrom(n, Vto, Vfrom, Ito, Ifrom);
    end
end
