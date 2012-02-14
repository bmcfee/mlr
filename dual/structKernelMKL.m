function H = structKernelMKL(Psi1, Psi2)

    H = Psi1(:)' * Psi2(:);

end
