function Xi = lossHinge(W, Psi, M, gradient)
%
%   Xi = lossHinge(W, Psi, M, gradient)
%   
%   W:          d*d metric
%   Psi:        d*d feature matrix
%   M:          the desired margin
%   gradient:   if 0, returns the loss value
%               if 1, returns the gradient of the loss WRT W
    
    Xi = max(0, M - sum(sum(W .* Psi)));

    if gradient & Xi > 0
        Xi = -Psi;
    end
end
