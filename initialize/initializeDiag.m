function W = initializeDiag(X)

    [d,n,m] = size(X);

%     W = ones(d, 1);
    W = 1./std(X,1,2);
    W(isinf(W)) = 1;
end
