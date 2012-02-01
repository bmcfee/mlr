function H = structKernelMKL(Psi1, Psi2, pass)

    H = 0;
    for i = 1:size(Psi1, 3)
        H = H + Psi1(:,:,i)' * Psi2(:,:,i);
    end

end
