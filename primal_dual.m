function [alpha, cost, error] = primal_dual(K, y, lambda, mu, delta, tol, max_iters)
    n = size(K, 1);
    alpha = zeros(n, 1);
    % v = alpha;
    z = zeros(n, 1);
    
    eta = 1;
    % theta = 1;
    cost = zeros(max_iters, 1);
    error = zeros(max_iters, 1);
    
    for i = 1:max_iters
        alpha_new = (eye(n) + eta * lambda * K) \ (alpha - eta * K' * z);
        
        theta_new = sqrt(1 / (1 + 2 * mu * delta));
        eta = eta * theta_new;
        delta = delta * theta_new;
        
        z_new = projPr(z + delta * (K * alpha_new - y));
        
        v = alpha_new + theta_new * (alpha_new - alpha);
        alpha = alpha_new;
        z = z_new;
        
        err = norm(K * v - y, 2)^2 / norm(y, 2)^2;
        
        cost(i) = err.^2/max_iters;
        error(i) = err;

        if err < tol
            disp(['Converged after ', num2str(i), ' iterations.'])
            break;
        end
    end
end

function projected = projPr(u)
    max_norm = max(norm(u, inf), 1);
    projected = u / max_norm;
end
