function dPsi = cpGradientRCPFull(X, S, batchSize)

    dPsi_matrix     = X * S * X' / batchSize;

    global LOSS;
    global C;

    if isequal(LOSS, @lossHingeRCPFull)
        d = size(X,1);
        % gradient is going to be:
        %   eye(d) / d - C * dPsi_matrix

        % Project the gradient onto PSD
        [V,D]           = eig(eye(d) / d - C * dPsi_matrix);
        D               = max(0, diag(D));

        % Now factor, so we can sample from N(0, P[subgrad(f)])
        L               = bsxfun(@times, D, V');
    else
        error(sprintf('[MLR-RCP] Unsupported loss function'));
    end

    dPsi = cat(3, dPsi_matrix, L);
end
