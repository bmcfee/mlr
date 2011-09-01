function D = distanceDiag(W, X)

    D = PsdToEdm(X' * bsxfun(@times, W, X));
end
