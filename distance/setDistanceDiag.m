function D = setDistanceDiag(X, W, Ifrom, Ito)
%
% D = setDistanceDiag(X, W, Ifrom, Ito)
%
%   X       = d-by-n data matrix
%   W       = d-by-1 PSD matrix
%   Ifrom   = k-by-1 vector of source points
%   Ito     = j-by-1 vector of destination points
%
%   D = n-by-n matrix of squared euclidean distances from Ifrom to Ito
%       D is sparse, and only the rows corresponding to Ifrom and
%       columns corresponding to Ito are populated.

    [d,n]       = size(X);
    L           = W.^0.5;
    
    Vfrom       = bsxfun(@times, L, X(:,Ifrom));

    if nargin == 4
        Vto     = bsxfun(@times, L, X(:,Ito));
    else
        Vto     = bsxfun(@times, L, X);
        Ito     = 1:n;
    end

    D = distToFrom(n, Vto, Vfrom, Ito, Ifrom);
end
