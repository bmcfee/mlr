function W = initializeDODMKL(X)

    [d,n,m] = size(X);

    W = zeros(m,m,d);
    for i = 1:m
        for j = i:m
            W(i,j,:) = ones(d,1);
        end
    end
end
