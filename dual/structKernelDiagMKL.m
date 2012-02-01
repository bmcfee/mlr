function H = structKernelDiagMKL(Psi1, Psi2, RESHAPE)

    if nargin < 3
        RESHAPE = 0;
    end

    H = 0;

    if RESHAPE
        for i = 1:size(Psi1, 3)
            H = H + Psi1(:,i)' * diag(Psi2(:,:,i));
        end
    else
        for i = 1:size(Psi1, 3)
            H = H + diag(Psi1(:,:,i))' * diag(Psi2(:,:,i));
        end
    end
end
