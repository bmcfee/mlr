function H = structKernelDiag(Psi1, Psi2, pass)

    H = diag(Psi1)' * diag(Psi2);

end
