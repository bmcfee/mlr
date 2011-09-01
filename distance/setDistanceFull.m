function D = setDistanceFull(X, W, Ifrom, Ito)
%
% D = setDistanceFull(X, W, Ifrom, Ito)
%
%   X       = d-by-n data matrix
%   W       = d-by-d PSD matrix
%   Ifrom   = k-by-1 vector of source points
%   Ito     = j-by-1 vector of destination points
%
%   D = n-by-n matrix of squared euclidean distances from Ifrom to Ito
%       D is sparse, and only the rows corresponding to Ifrom and
%       columns corresponding to Ito are populated.

    [d,n]       = size(X);
    [vecs,vals] = eig(0.5 * (W + W'));
    L           = real(abs(vals)).^0.5 * vecs';

    Vfrom   = L * X(:,Ifrom);

    if nargin == 4
        Vto     = L * X(:,Ito);
    else
        Vto     = L * X;
        Ito     = 1:n;
    end

    D = distToFrom(n, Vto, Vfrom, Ito, Ifrom);
end
