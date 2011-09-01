function r = regularizeMKLDiag(W, X, gradient)
%
% r = regularizeMKL(W, X, gradient)
%
%

    [d,n,m] = size(X);

    if gradient
        r = zeros(d,m);
        for i = 1:m
            r(:,i) = diag(X(:,:,i));
        end
    else
        r = 0;
        for i = 1:m
            r = r + W(:,i)' * diag(X(:,:,i));
        end
    end
end
