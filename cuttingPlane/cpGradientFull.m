function dPsi = cpGradientFull(X, S, batchSize)

    dPsi    = X * S * X' / batchSize;

end
