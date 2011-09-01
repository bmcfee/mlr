function W = initializeDiagMKL(X)
    
    [d,n,m] = size(X);

    W = zeros(d,m);
    for i = 1:m
        W(:,i) = 1./std(X(:,:,i),1,2);
    end
    W(isinf(W)) = 1;
end
