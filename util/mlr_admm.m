function [W, Xi, Diagnostics] = mlr_admm(C, K, Delta, H, Q, RHO)
% [W, Xi, D] = mlr_admm(C, Delta, W, X)
%
%   C       >= 0    Slack trade-off parameter
%   K       =       data matrix (or kernel)
%   Delta   =       array of mean margin values
%   H       =       structural kernel matrix
%   Q       =       kernel-structure interaction vector
%   RHO     > 0     augmented lagrangian term
%
%   W (output)  =   the learned metric
%   Xi          =   1-slack
%   D           =   diagnostics

    global DEBUG REG FEASIBLE LOSS INIT STRUCTKERNEL DUALW;

    %%%
    % Initialize the gradient directions for each constraint
    %
    global PsiR;

    global ADMM_Z ADMM_U;

    numConstraints = length(PsiR);

    Diagnostics = struct(   'f',                [], ...
                            'num_steps',        [], ...
                            'stop_criteria',    []);


    % Convergence settings
    MAX_ITER    = 50;
    ABSTOL      = 1e-4 * sqrt(numel(ADMM_Z));
    RELTOL      = 1e-2;
    stopcriteria= 'MAX STEPS';

    % Objective function
    F           = zeros(1,MAX_ITER);

    % how many constraints

    alpha           = zeros(numConstraints,  1);
    Gamma           = zeros(numConstraints,  1);

    for step = 1:MAX_ITER
        % do a w-update
        % dubstep needs:
        %   C       <-- static
        %   RHO     <-- static
        %   H       <-- static
        %   Q       <-- static
        %   Delta   <-- static
        %   Gamma   <-- this one's dynamic

        for i = 1:numConstraints
            Gamma(i) = STRUCTKERNEL(ADMM_Z-ADMM_U, PsiR{i}, 1);
        end
        alpha   = mlr_dual(C, RHO, H, Q, Delta, Gamma, alpha);

        %%%
        % 3) convert back to W
        %
        W = DUALW(alpha, ADMM_Z, ADMM_U, RHO, K);

%         figure(1), imagesc(W), drawnow;
        % Update Z
        Zold    = ADMM_Z;
        ADMM_Z  = FEASIBLE(W + ADMM_U);

        % Update residuals
        ADMM_U  = ADMM_U + W - ADMM_Z;

        % Compute objective
        %   slack term
        Xi      = 0;
        for R = numConstraints:-1:1
            Xi = max(Xi, LOSS(W, PsiR{R}, Delta(R), 0));
        end
        N1          = norm(W(:)-ADMM_Z(:));
        F(step)     = C * Xi + REG(W, K, 0) + RHO / 2 * N1^2;
        
%         figure(2), loglog(1:step, F(1:step)), xlim([0, MAX_ITER]), drawnow;
        % Test for convergence

        N2 = RHO * norm(Zold(:) - ADMM_Z(:));

        eps_primal = ABSTOL + RELTOL * max(norm(W(:)), norm(ADMM_Z(:)));
        eps_dual   = ABSTOL + RELTOL * RHO * norm(ADMM_U(:));
        if N1 < eps_primal && N2 < eps_dual
            stopcriteria = 'CONVERGENCE';
            break;
        end
    end

    %%%
    % Ensure feasibility
    %
    W = FEASIBLE(W);
    

    %%%
    % Compute the slack
    %
    Xi = 0;
    for R = numConstraints:-1:1
        Xi  = max(Xi, LOSS(W, PsiR{R}, Delta(R), 0));
    end

    %%% 
    % Update diagnostics
    %

    Diagnostics.f               = F(1:step)';
    Diagnostics.stop_criteria   = stopcriteria;
    Diagnostics.num_steps       = step;

    dbprint(1, '\t%s after %d steps.\n', stopcriteria, step);
end

function alpha = mlr_dual(C, RHO, H, Q, Delta, Gamma, alpha)

    global PsiClock;

    m = length(Delta);

    if nargin < 7
        alpha = zeros(m,1);
    end

    %%%
    % 1) construct the QP parameters
    %
    b = RHO * (Gamma - Delta) - Q;

    %%%
    % 2) solve the QP
    %
    alpha = qplcprog(H, b, ones(1, m), C, [], [], 0, [], 'tolpiv', 1e-5);

    %%%
    % 3) update the Psi clock
    %
    PsiClock(alpha > 0)  = 0;

end
