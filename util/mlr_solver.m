function [W, Xi, Diagnostics] = mlr_solver(C, Margins, W, K)
% [W, Xi, D] = mlr_solver(C, Margins, W, X)
%
%   C       >= 0    Slack trade-off parameter
%   Margins =       array of mean margin values
%   W       =       initial value for W 
%   X       =       data matrix (or kernel)
%
%   W (output)  =   the learned metric
%   Xi          =   1-slack
%   D           =   diagnostics

    global DEBUG REG FEASIBLE LOSS;

    %%%
    % Initialize the gradient directions for each constraint
    %
    global PsiR;
    global PsiClock;

    numConstraints = length(PsiR);

    %%% 
    % Some optimization details
    
    % Armijo rule number
    armijo      = 1e-5;
    
    % Initial learning rate
    lambda0     = 1e-6;

    % Increase/decrease after each iteration
    lambdaup    = ((1+sqrt(5))/2)^(1/3);
    lambdadown  = ((1+sqrt(5))/2)^(-1);

    % Maximum steps to take
    maxsteps    = 1e4;

    % Size of convergence window
    frame       = 10;
    
    % Convergence threshold
    convthresh  = 1e-5;

    % Maximum number of backtracks
    maxbackcount = 100;


    Diagnostics = struct(   'f',                [], ...
                            'num_steps',        [], ...
                            'stop_criteria',    []);

    % Repeat until convergence:
    % 1) Calculate f
    % 2) Take a gradient step
    % 3) Project W back onto PSD

    %%%
    % Initialze
    %

    f       = inf;
    dfdW    = zeros(size(W));
    lambda  = lambda0;
    F       = Inf * ones(1,maxsteps+1);
    XiR     = zeros(numConstraints,1);


    stepcount   = -1;
    backcount   = 0;
    done        = 0;


    while 1
        fold = f;
        Wold = W;

        %%%
        % Count constraint violations and build the gradient
        dbprint(3, 'Computing gradient');

        %%%
        % Calculate constraint violations
        %
        XiR(:) = 0;
        for R = numConstraints:-1:1
            XiR(R)  = LOSS(W, PsiR{R}, Margins(R), 0);
        end

        %%%
        % Find the most active constraint
        %
        [Xi, mgrad] = max(XiR);
        Xi          = max(Xi, 0);
        
        PsiClock(mgrad) = 0;

        %%%
        % Evaluate f
        %

        f           = C     * max(Xi, 0) ...
                            + REG(W, K, 0);

        %%%
        % Test for convergence
        %
        objDiff        = fold - f;

        if objDiff > armijo * lambda * (dfdW(:)' * dfdW(:))

            stepcount = stepcount + 1;

            F(stepcount+1) = f;

            sdiff = inf;
            if stepcount >= frame;
                sdiff = log(F(stepcount+1-frame) / f);
            end

            if stepcount >= maxsteps
                done = 1; 
                stopcriteria = 'MAXSTEPS';
            elseif sdiff <= convthresh
                done = 1;
                stopcriteria = 'CONVERGENCE';
            else
                %%%
                % If it's positive, add the corresponding gradient
                dfdW    = C     * LOSS(W, PsiR{mgrad}, Margins(mgrad), 1) ...
                                + REG(W, K, 1);
            end

            dbprint(3, 'Lambda up!');
            Wold        = W;
            lambda      = lambdaup * lambda;
            backcount   = 0;

        else
            % Backtracking time, drop the learning rate
            if backcount >= maxbackcount
                W       = Wold;
                f       = fold;
                done    = 1;

                stopcriteria = 'BACKTRACK';
            else
                dbprint(3, 'Lambda down!');
                lambda      = lambdadown * lambda;
                backcount   = backcount+1;
            end
        end
        
        %%%
        % Take a gradient step
        %
        W   = W - lambda * dfdW;

        %%%
        % Project back onto the feasible set
        %

        dbprint(3, 'Projecting onto feasible set');
        W   = FEASIBLE(W);
        if done
            break;
        end; 

    end

    Diagnostics.f               = F(2:(stepcount+1))';
    Diagnostics.stop_criteria   = stopcriteria;
    Diagnostics.num_steps       = stepcount;

    dbprint(1, '\t%s after %d steps.\n', stopcriteria, stepcount);
end

