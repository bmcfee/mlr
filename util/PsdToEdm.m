function D = PsdToEdm(G)

    d = diag(G);
    D = bsxfun(@plus, d, d') - 2 * G;
