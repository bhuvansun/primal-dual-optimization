function [x, cost, error, x_values, x_history] = gradient_descent(f, grad_f, x0, max_iter, alpha)
    x = x0;
    y = x0;
    t = 1;
    cost = zeros(max_iter, 1);
    error = zeros(max_iter, 1);
    x_history = zeros(max_iter, 1);

    for i = 1:max_iter
        x_prev = x;
        grad = grad_f(y);

        x = y - alpha * grad;
        t_prev = t;
        t = (1 + sqrt(1 + 4 * t^2)) / 2;
        y = x + ((t_prev - 1) / t) * (x - x_prev);

        x_history(i) = x;

        cost(i) = f(x).^2/max_iter;
        error(i) = f(x);
    end

    x_values = x_history(x_history ~= 0);

end
