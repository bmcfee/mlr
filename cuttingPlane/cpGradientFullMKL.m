function dPsi = cpGradientFullMKL(X, S, batchsize)

    [d,n,m] = size(X);

    dPsi = zeros(d,d,m);
    for i = 1:m
        dPsi(:,:,i)    = X(:,:,i) * S * X(:,:,i)';
    end

    dPsi = dPsi / batchsize;
end
