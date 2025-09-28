
% RUN_PARTB_DEMO  Minimal demo reproducing Section 5.6.B (near-threshold)
% Example: p=2/3, lambda=0.8, y=1.26 (just above tau), target eta = 1e-8
clearvars;  % avoid name conflicts (e.g., a variable named 'gamma')

p = 2/3; lambda = 0.8; y = 1.5;
opts = struct();
opts.eta     = 1e-8;
opts.R       = 0.40;   % must be < u_star(p); smaller R => easier certification
opts.verbose = true;
opts.nt      = 3000;   % more panels if you want extra safety
opts.safety  = 1.25;

Sstar = xplus_lp_series(y, p, lambda, 1e-14, 5000);
[xhat, info] = xplus_lp_hybrid(y, p, lambda, opts);

fprintf('y=%.6f, xhat=%.12f\n', y, xhat);
fprintf('Certificates: series<=%.3e, MB<=%.3e, total<=%.3e -> |xhat-x_+|<= y*total<= %.3e\n', ...
    info.series_tail_bound, info.mb_tail_bound, info.total_bound, info.total_x_error_bound);
