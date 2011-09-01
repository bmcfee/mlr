function W = initializeFull(X)

    [d,n,m] = size(X);

    W = zeros(d,d,m);
    for i = 1:m
        W(:,:,i) = diag(1./std(X(:,:,i),1,2));
    end
    W(isinf(W)) = 1;
end
