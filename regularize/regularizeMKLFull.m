function r = regularizeMKLFull(W, X, gradient)
%
% r = regularizMKL(W, X, gradient)
%
%

    [d,n,m] = size(X);

    if gradient
        r = X;
    else
        r = 0;
        for i = 1:m
            r = r + sum(sum(W(:,:,i) .* X(:,:,i)));
        end
    end
end
