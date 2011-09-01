function Xi = lossHingeMKLFull(W, Psi, M, gradient)
%
%   Xi = lossHingeMKLFull(W, Psi, M, gradient)
%   
%   W:          d*d*m metric
%   Psi:        d*d*m feature matrix
%   M:          the desired margin
%   gradient:   if 0, returns the loss value
%               if 1, returns the gradient of the loss WRT W
    
    m = size(W, 3);

    Xi = M;
    for i = 1:m
        Xi = Xi -  sum(sum(W(:,:,i) .* Psi(:,:,i)));
    end
    Xi = max(0, Xi);

    if gradient & Xi > 0
        Xi = -Psi;
    end
end
