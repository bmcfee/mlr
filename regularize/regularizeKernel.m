function r = regularizeKernel(W, X, gradient)
%
% r = regularizeKernel(W, X, gradient)
%
%
    if gradient
        r = X;
    else
        r = sum(sum(W .* X));
    end
end
