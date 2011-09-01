function dPsi = cpGradientDODMKL(X, S, batchsize)

    [d,n,m] = size(X);

    dPsi    = zeros(m,m,d);

    for i = 1:m
        dPsi(i,i,:)     = diag(X(:,:,i) * S * X(:,:,i)');
        for j = (i+1):m
            Q           = X(:,:,i) + X(:,:,j);
            dPsi(i,j,:) = diag( Q * S * Q');
        end
    end
    dPsi = dPsi / batchsize;
end
