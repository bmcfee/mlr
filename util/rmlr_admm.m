function [W, Xi, Diagnostics] = rmlr_admm(C, K, Delta, H, Q, lam)
% [W, Xi, D] = mlr_admm(C, Delta, W, X)
%
%   C       >= 0    Slack trade-off parameter
%   K       =       data matrix (or kernel)
%   Delta   =       array of mean margin values
%   H       =       structural kernel matrix
%   Q       =       kernel-structure interaction vector
%
%   W (output)  =   the learned metric
%   Xi          =   1-slack
%   D           =   diagnostics

global DEBUG REG FEASIBLE LOSS INIT STRUCTKERNEL DUALW THRESH;

%%%
% Initialize the gradient directions for each constraint
%
global PsiR;

global ADMM_Z ADMM_V ADMM_UW ADMM_UV;

global ADMM_STEPS;

global RHO;

numConstraints = length(PsiR);

Diagnostics = struct(   'f',                [], ...
    'num_steps',        [], ...
    'stop_criteria',    []);


% Convergence settings
if ~isempty(ADMM_STEPS)
    MAX_ITER = ADMM_STEPS;
else
    MAX_ITER = 10;
end
ABSTOL      = 1e-4 * sqrt(numel(ADMM_Z));
RELTOL      = 1e-3;
SCALE_THRESH    = 10;
RHO_RESCALE     = 2;
stopcriteria= 'MAX STEPS';

% Objective function
F           = zeros(1,MAX_ITER);

% how many constraints

alpha           = zeros(numConstraints,  1);
Gamma           = zeros(numConstraints,  1);

ln1 = 0;
ln2 = 0;

% figure(2)
% hold off
% plot(0)
% delete(abc)
% delete(abc2)
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
        Gamma(i) = STRUCTKERNEL(ADMM_Z-ADMM_UW, PsiR{i});
    end
%     d = length(K);
    alpha = mlr_dual(C, RHO, H, Q, Delta, Gamma, alpha);
    
    %%%
    % 3) convert back to W
    %
    W = DUALW(alpha, ADMM_Z, ADMM_UW, RHO, K);
    
                figure(1), imagesc(W), drawnow;
    
    % Update V
    ADMM_V = THRESH(ADMM_Z - ADMM_UV, lam/RHO);
    
    % Update Z
    Zold    = ADMM_Z;
    ADMM_Z  = FEASIBLE(0.5* (W + ADMM_V + ADMM_UW + ADMM_UV));
    
    % Update residuals
    ADMM_UW  = ADMM_UW + W - ADMM_Z;
    ADMM_UV  = ADMM_UV + ADMM_V - ADMM_Z;
    
    % Compute primal objective
    %   slack term
    Xi      = 0;
    for R = numConstraints:-1:1
        Xi = max(Xi, LOSS(ADMM_Z, PsiR{R}, Delta(R), 0));
    end
    F(step)     = C * Xi + REG(W, K, 0) + lam * sum(sqrt(sum(W.^2)));
    
%     figure(2), loglog(1:step, F(1:step)), xlim([0, MAX_ITER]), drawnow;
    % Test for convergence
    
    %WIP
    N1          = norm(ADMM_V(:) + W(:) - 2* ADMM_Z(:));
    N2          = RHO * norm(2* (Zold(:) - ADMM_Z(:)));
    
    eps_primal = ABSTOL + RELTOL * max(norm(W(:)), norm(ADMM_Z(:)));
    eps_dual   = ABSTOL + RELTOL * RHO * norm(ADMM_UW(:));
    %end WIP
    
    
%            figure(2), loglog(step + (-1:0), [ln1, N1/eps_primal], 'b'), xlim([0, MAX_ITER]), hold('on');
%            figure(2), loglog(step + (-1:0), [ln2, N2/eps_dual], 'r-'), xlim([0, MAX_ITER]), hold('on'), drawnow;
%           ln1 = N1/eps_primal;
%           ln2 = N2/eps_dual;
    
    if N1 < eps_primal && N2 < eps_dual
        stopcriteria = 'CONVERGENCE';
        break;
    end
    
    if N1 > SCALE_THRESH * N2
         dbprint(3, sprintf('RHO: %.2e UP %.2e', RHO, RHO * RHO_RESCALE));
        RHO = RHO * RHO_RESCALE;
        ADMM_UW  = ADMM_UW / RHO_RESCALE;
    elseif N2 > SCALE_THRESH * N1
         dbprint(3, sprintf('RHO: %.2e DN %.2e', RHO, RHO / RHO_RESCALE));
        RHO = RHO / RHO_RESCALE;
        ADMM_UW  = ADMM_UW * RHO_RESCALE;
    end
end
%     figure(2), hold('off');

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
alpha = qplcprog(H, b, ones(1, m), C, [], [], 0, []);

%%%
% 3) update the Psi clock
%
PsiClock(alpha > 0)  = 0;

end
