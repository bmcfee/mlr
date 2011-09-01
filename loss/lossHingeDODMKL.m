function Xi = lossHingeDODMKL(W, Psi, M, gradient)
%
%   Xi = lossHingeDODMKL(W, Psi, M, gradient)
%   
%   W:          m*m*d matrix of diagonal metrics
%   Psi:        m*m*d feature matrix
%   M:          the desired margin
%   gradient:   if 0, returns the loss value
%               if 1, returns the gradient of the loss WRT W

    m = size(W,1);

    Xi = max(0, M - sum(sum(sum(W .* Psi))));

    if gradient & Xi > 0
        Xi = -Psi;
    end
end
