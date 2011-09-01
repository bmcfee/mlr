function dPsi = cpGradientDiagMKL(X, S, batchsize)

    [d,n,m] = size(X);

    dPsi    = zeros(d,m);

    for i = 1:m
        dPsi(:,i)    = diag(X(:,:,i) * S * X(:,:,i)');
    end
    
    dPsi = dPsi / batchsize;
end
