function D = distanceDODMKL(W, X)

    [d, n, m] = size(X);

    D = 0;
    for i = 1:m
        D = D + PsdToEdm(X(:,:,i)' * bsxfun(@times, squeeze(W(i,i,:)), X(:,:,i)));
        for j = (i+1):m
            Q = X(:,:,i) + X(:,:,j);
            D = D + PsdToEdm(Q' * bsxfun(@times, squeeze(W(i,j,:)), Q));
        end
    end
end
