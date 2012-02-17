function W = initializeFull(X)
    
    [d,n,m] = size(X);

    W = diag(1./std(X,1,2));
    W(isinf(W)) = 1;
    W(isnan(W)) = 1;

end
