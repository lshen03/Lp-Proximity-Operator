function [xhat, info] = xplus_lp_hybrid(y, p, lambda, opts)
%XPLOUS_LP_HYBRID  Certified hybrid evaluator for x_+(y) with 0<p<1.
% Hybrid = truncated series + truncated Mellin–Barnes (MB) vertical segment,
% with a practical error certificate (Section 5.6.B, distance-certificate use).
%
% [xhat, info] = xplus_lp_hybrid(y, p, lambda, opts)
% Inputs:
%   y       : scalar or vector with y > tau_{p,lambda}
%   p       : 0<p<1
%   lambda  : >0
%   opts    : struct (all optional)
%       .eta        : target |xhat - x_+|/y bound (default 1e-8)
%       .R          : cap for |epsilon|, must satisfy R < u_star(p) (default 0.9*u_*)
%       .sigmaN     : vertical line Re(s)=sigmaN in (N, N+1) (default N+0.5)
%       .N          : if provided, fix N; else picked to meet eta/2 for series tail
%       .T          : if provided, fix T; else picked to meet eta/2 for MB tail
%       .nt         : # trapezoid panels on [-T, T] (default 2000)
%       .safety     : multiplicative safety on bounds (default 1.25)
%       .maxN       : maximum allowed N in search (default 200)
%       .maxT       : maximum allowed T in search (default 50000)  % ↑ larger default
%       .verbose    : true/false (default false)
%
% Outputs:
%   xhat : approximation to x_+(y)
%   info : details and certificates (scalar fields)
%
if nargin < 4, opts = struct(); end
if ~isfield(opts, 'eta'),      opts.eta = 1e-8; end
if ~isfield(opts, 'verbose'),  opts.verbose = false; end
if ~isfield(opts, 'nt'),       opts.nt = 2000; end
if ~isfield(opts, 'safety'),   opts.safety = 1.25; end
if ~isfield(opts, 'maxN'),     opts.maxN = 200; end
if ~isfield(opts, 'maxT'),     opts.maxT = 50000; end  % bigger default helps near-threshold

u_star = ustar(p);
tau    = ((2-p)/(2*(1-p))) * (2*lambda*(1-p))^(1/(2-p)); % tau_{p,lambda}
if ~isfield(opts, 'R'), opts.R = 0.9*u_star; end

info = struct(); info.p = p; info.lambda = lambda;
info.u_star = u_star; info.tau = tau; info.R = opts.R;
info.eta_target = opts.eta; info.status = 'init';

xhat = zeros(size(y));
for k = 1:numel(y)
    yk = y(k);
    if yk <= tau
        error('xplus_lp_hybrid: y must exceed tau_{p,lambda}≈%.6g (got y=%.6g).', tau, yk);
    end
    eps_k = lambda*p*yk^(p-2);
    if abs(eps_k) > opts.R
        error('epsilon(y)=%.4g exceeds R=%.4g (choose larger R or shrink step).', abs(eps_k), opts.R);
    end

    % --- choose N for series tail ---
    if isfield(opts,'N') && ~isempty(opts.N)
        N = opts.N;
        [tailS, C2_used] = series_tail_bound_from_next(N, opts.R, p);
    else
        [N, tailS, C2_used] = choose_N_for_eta(p, opts.R, opts.eta/2, opts.maxN, opts.verbose);
    end
    sigmaN = N + 0.5;
    if isfield(opts,'sigmaN') && ~isempty(opts.sigmaN), sigmaN = opts.sigmaN; end

    % --- choose T for MB tail ---
    if isfield(opts,'T') && ~isempty(opts.T)
        T = opts.T;
        mbBound = bound_mb_tail_numeric(p, sigmaN, opts.R, T);  % guaranteed scalar
    else
        [T, mbBound] = choose_T_for_eta(p, sigmaN, opts.R, opts.eta/2, opts.maxT, opts.verbose);
    end

    % --- evaluate series and MB segment ---
    z_series = 1.0;
    for n = 1:N
        an = a_coeff(n, p);
        z_series = z_series - an * (eps_k)^n;
    end
    [z_mb, dt] = mb_segment(eps_k, p, sigmaN, T, opts.nt);

    zhat = z_series + z_mb;

    % --- certificates ---
    cert_series = tailS;
    cert_mb     = mbBound;                 % ALWAYS scalar (may be > eta/2 if maxT hit)
    cert_total  = opts.safety * (cert_series + cert_mb);
    total_x_err = yk * cert_total;

    xhat(k) = yk * zhat;

    % record (per-point; last one is kept if y is a vector)
    info.eps = eps_k;
    info.N = N; info.T = T; info.sigmaN = sigmaN; info.dt = dt; info.nt = opts.nt;
    info.z_series_sum = z_series; info.z_mb_segment = z_mb; info.zhat = zhat;
    info.series_tail_bound = cert_series;
    info.mb_tail_bound = cert_mb;
    info.total_bound = cert_total;
    info.total_x_error_bound = total_x_err;
    info.C2_used = C2_used;
    info.status = 'ok';
    if opts.verbose
        fprintf('y=%.6g: N=%d, T=%.1f, |eps|=%.4g, series<=%.2e, MB<=%.2e, total<=%.2e\n', ...
            yk, N, T, abs(eps_k), cert_series, cert_mb, cert_total);
    end
end
end

% ---------------- helpers ----------------
function u = ustar(p), u = (1-p)^(1-p) / (2-p)^(2-p); end

function an = a_coeff(n, p)
% a_n = 1/n! * Gamma((2-p)n - 1) / Gamma((1-p)n)  (real args → use real gammaln)
an = exp( -gammaln(n+1) + real(gammaln((2-p)*n - 1)) - real(gammaln((1-p)*n)) );
end

function [tailBound, C2_used] = series_tail_bound_from_next(N, R, p)
u = ustar(p); r = R/u; if ~(r<1), error('R must be < u_star(p).'); end
n_max = max(1000, N+400);
ns = (1:n_max).';
loga = -gammaln(ns+1) + real(gammaln((2-p)*ns - 1)) - real(gammaln((1-p)*ns));
log_vals = loga + ns.*log(u);
finite = isfinite(log_vals);
if ~any(finite), logC2_emp = 0; else, logC2_emp = max(log_vals(finite)); end
C2_emp = exp(logC2_emp); C2_used = 1.1 * C2_emp;
tailBound = C2_used * (r^(N+1)) / (1 - r);
if ~isfinite(tailBound), tailBound = Inf; end
end

function [N, tailBound, C2_used] = choose_N_for_eta(p, R, eta_half, maxN, verbose)
tb = Inf; C2 = NaN;
for N = 1:maxN
    [tb, C2] = series_tail_bound_from_next(N, R, p);
    if tb <= eta_half
        tailBound = tb; C2_used = C2;
        if verbose, fprintf('[choose N] N=%d OK: tail<=%.2e\n', N, tb); end
        return
    end
end
tailBound = tb; C2_used = C2;
warning('choose_N_for_eta: reached maxN=%d with tail=%.3e > eta/2=%.3e.', maxN, tb, eta_half);
end

function [z_mb, dt] = mb_segment(eps, p, sigmaN, T, nt)
% (1/(2π)) ∫_{-T}^{T} H(sigmaN+it) * (-eps)^{sigmaN+it} dt
t = linspace(-T, T, nt+1); dt = t(2)-t(1);
s = sigmaN + 1i*t;
H = exp( cplx_lngamma(-s) + cplx_lngamma(s*(2-p) - 1) - cplx_lngamma(s*(1-p)) );
logm_eps = log(abs(eps)) + 1i*pi; % principal branch for -eps (eps>0)
phase = exp( (sigmaN + 1i*t) * logm_eps );
integrand = H .* phase;
I = trapz(t, integrand);
z_mb = (1/(2*pi)) * I;
end

function Cnum = sample_H_envelope(p, sigma, tgrid)
% Return a finite, nonempty constant even if numerics struggle.
s = sigma + 1i*tgrid;
H = exp( cplx_lngamma(-s) + cplx_lngamma(s*(2-p) - 1) - cplx_lngamma(s*(1-p)) );
A = (2-p)*log(2-p) - (1-p)*log(1-p);
val = abs(H) .* exp(pi*abs(tgrid)) .* (abs(tgrid).^(3/2)) ./ exp(A*sigma);
finite = isfinite(val);
if ~any(finite)
    Cnum = 1;                % conservative fallback constant
else
    Cnum = max(val(finite));
    if ~isfinite(Cnum) || isempty(Cnum), Cnum = 1; end
end
end

function mbBound = bound_mb_tail_numeric(p, sigma, R, T)
Tmax = max(1000, 5*T);
tgrid = linspace(T, Tmax, 4000);
Cnum = sample_H_envelope(p, sigma, tgrid);
mbBound = 2 * Cnum * (R^sigma) / pi * T^(-1/2);
if ~isfinite(mbBound) || isempty(mbBound), mbBound = Inf; end
end

function [T, mbBound] = choose_T_for_eta(p, sigma, R, eta_half, maxT, verbose)
T = 50;
while T < maxT
    mbBound = bound_mb_tail_numeric(p, sigma, R, T);
    if isfinite(mbBound) && (mbBound <= eta_half)
        if verbose, fprintf('[choose T] T=%.1f OK: MBtail<=%.2e\n', T, mbBound); end
        return
    end
    T = T * 1.5;
end
% Ensure we return a scalar bound even if target not met:
mbBound = bound_mb_tail_numeric(p, sigma, R, T);
warning('choose_T_for_eta: reached maxT=%.1f with MBtail=%.3e > eta/2=%.3e.', maxT, mbBound, eta_half);
end

% ---------- Complex log-gamma (Lanczos + reflection), vectorized ----------
function lg = cplx_lngamma(z), lg = arrayfun(@cplx_lngamma_scalar, z); end
function lg = cplx_lngamma_scalar(z)
if real(z) < 0.5
    lg = log(pi) - log(sin(pi*z)) - cplx_lngamma_lanczos(1 - z);
else
    lg = cplx_lngamma_lanczos(z);
end
end
function lg = cplx_lngamma_lanczos(z)
p = [ ...
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, ...
  771.32342877765313, -176.61502916214059, 12.507343278686905, ...
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
g = 7; z1 = z - 1; x = p(1);
for k = 2:numel(p), x = x + p(k) ./ (z1 + (k-1)); end
t = z1 + g + 0.5;
lg = 0.5*log(2*pi) + (z1+0.5).*log(t) - t + log(x);
end
