function H = structKernelLinear(Psi1, Psi2, pass)

    H = Psi1(:)' * Psi2(:);

end
