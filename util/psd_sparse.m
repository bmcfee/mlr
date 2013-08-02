function W = psd_sparse(A)
    % Compute the projection of a symmmetric, real matrix A onto the PSD cone
    % If A is row/column-sparse, only work in the subspace containing the non-zeros
    %

    % Compute the row sums
    z       = mean(abs(A), 1);

    % find the non-zero columns
    THRESH  = 1e-10;
    idx     = find(z > THRESH);

    % Pull out the non-zero submatrix
    B       = A(idx, idx);
    
    % Compute the projection
    [v, d]  = eig(0.5 * (B + B'));

    % Repopulate
    W            = zeros(size(A));
    W(idx, idx)  = v * bsxfun(@times, max(real(diag(d)), 0), v');
end
