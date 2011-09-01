function Xi = lossHingeDiagMKL(W, Psi, M, gradient)
%
%   Xi = lossHingeDiagMKL(W, Psi, M, gradient)
%   
%   W:          d*m matrix of diagonal metrics
%   Psi:        d*m feature matrix
%   M:          the desired margin
%   gradient:   if 0, returns the loss value
%               if 1, returns the gradient of the loss WRT W
    
    Xi = max(0, M - trace(W' * Psi));

    if gradient & Xi > 0
        Xi = -Psi;
    end
end
