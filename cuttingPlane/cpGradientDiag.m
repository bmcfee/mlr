function dPsi = cpGradientDiag(X, S, batchsize)

    dPsi    = diag(X * S * X') / batchsize;

end
