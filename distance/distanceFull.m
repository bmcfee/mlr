function D = distanceFull(W, X)

    D = PsdToEdm(X' * W * X);
end
