function S = xplus_lp_series(y, p, lambda, tol, maxN)
% XPLUS_LP_SERIES  Compute S(y) = x_+(y) series for the ℓ_p prox (0<p<1).
%   y      : vector/array of inputs (can be any real; magnitude is used)
%   p      : scalar in (0,1)
%   lambda : positive scalar
%   tol    : (optional) relative tolerance, default 1e-12
%   maxN   : (optional) hard cap on series terms, default 1000
%
% Returns:
%   S : same shape as y; NaN where |y| <= tau_{p,lambda} (series not valid)

    if nargin < 4 || isempty(tol),   tol = 1e-12; end
    if nargin < 5 || isempty(maxN),  maxN = 1000; end
    if ~(p>0 && p<1), error('p must be in (0,1)'); end
    if ~(lambda>0),   error('lambda must be > 0'); end

    % thresholds (scalar)
    rho  = (2*lambda*(1-p))^(1/(2-p));
    tau  = ((2-p)/(2*(1-p))) * rho;

    ya   = abs(y);
    mask = ya > tau;              % series is valid only here
    S    = nan(size(y));          % prefill with NaN outside domain

    if ~any(mask)
        return; % nothing to do
    end

    ym   = ya(mask);              % active entries
    epsv = (lambda*p) * ym.^(p-2);

    % Initialize S = y*(1 - eps)
    Sact = ym .* (1 - epsv);

    % We'll accumulate terms: term_n = y * (eps^n / n!) * A_n,
    % where A_n = Gamma(n(2-p)-1)/Gamma(n(1-p)) depends only on n,p.
    % Track eps powers vectorized across y; start at n=2 => eps.^2
    epsPow = epsv.^2;

    % Convergence tracker: we stop when the current term is small for all entries
    done = false(size(ym));

    for n = 2:maxN
        % Scalar coefficient using gammaln for stability
        logA  = gammaln(n*(2-p) - 1) - gammaln(n*(1-p));
        lognF = gammaln(n+1);
        cn    = exp(logA - lognF);     % equals A_n / n!

        term  = ym .* (cn .* epsPow);  % vectorized

        % Update S
        Sact  = Sact - term;

        % Check elementwise convergence
        done  = done | (abs(term) <= tol .* max(1, abs(Sact)));

        % If all active entries converged, stop
        if all(done), break; end

        % Prepare next power: eps^(n+1)
        epsPow = epsPow .* epsv;
    end

    % Write back to full output
    S(mask) = Sact;
end
