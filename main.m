set(0, 'DefaultLineLineWidth', 2);

%% Primal - Dual (Chambolle and Pock)

n = 100;
left = -1;
right = 1;

x = linspace(left, right, n);
% y = -50 + (50-(-50)) .* rand(n,1);
% y = floor(x);
% y = x - floor(x);
% y = sign(x);
y = abs(x);
% y = rectangularPulse(x);
y = y';

% K = randn(n, n);
K = kernelGenerator(n, x);

lambda = 1e-3;
mu = 0.1;
delta = 0.1;
tol = 1e-6;
max_iters = 1e4;

tic
[alpha, cost_pd, error_pd] = primal_dual(K, y, lambda, mu, delta, tol, max_iters);
toc

% disp('Estimated Solution set (alpha):');
% disp(alpha);

% loss = (1/(2*n)) * norm(K*alpha - y, 2)^2 + (lambda/2) * norm(alpha, 2)^2;

F = K * alpha;

% disp('Predicted values (F):');
% disp(F);
% 
% disp('Actual values (y):');
% disp(y);

% disp('MAE:');
% disp(sum(abs(F-y)));
% 
% disp('Loss function error:');
% disp(loss);

figure;
plot(y, 'b');
xlabel('Data Points');
ylabel('Values');
hold on;
plot(F, 'r');
title("Primal Dual");
legend('Original curve', 'Approximation Curve', 'Location', 'best');
grid on;
hold off;


figure;
subplot(2, 1, 1);
semilogy(cost_pd);
title('Cost vs Iterations');
xlabel('Iterations');
ylabel('Cost');

subplot(2, 1, 2);
semilogy(error_pd);
title('Error vs Iterations');
xlabel('Iterations');
ylabel('Error');

sgtitle("Primal Dual");

%% Gradient Descent (Nesterov Accelerated)

x_ = linspace(-1, 1, 100);
f_ = @(x_) abs(x_);

grad_f_ = gradient(f_(x_), x_);
max_iters = 1e4;

approx_curve = zeros(size(x_));
tic
for j = 1:max_iters
    for i = 2:length(x_)
        approx_curve(i) = approx_curve(i-1) + grad_f_(i-1) * (x_(i) - x_(i-1));
    end
end
toc

approx_curve = approx_curve + 1;
f = @(x) abs(x);
grad_f = @(x) sign(x);

x0 = 5;
max_iter = 1e4;
alpha = 0.1;

% [x, cost_gd, error_gd, x_values, x_history] = gradient_descent(f, grad_f, x0, max_iter, alpha);
[~, cost_gd, error_gd, ~, x_history] = gradient_descent(f, grad_f, x0, max_iter, alpha);

figure;
plot(approx_curve);
hold on;
plot(f_(x_));
hold off;
title('Gradient Descent');
xlabel('x');
ylabel('f(x)');
legend('Approximation', 'Original f(x)', 'Location', 'best');

figure;
subplot(2, 1, 1);
semilogy(1:max_iter, cost_gd);
title('Cost vs Iterations');
xlabel('Iterations');
ylabel('Cost');

subplot(2, 1, 2);
semilogy(1:max_iter, error_gd);
title('Error vs Iterations');
xlabel('Iterations');
ylabel('Error');

sgtitle("Gradient Descent");

%% Stochastic Gradient Descent








%% Adam Boost

% [optimal_x, min_loss, approximated_function, x_values, cost_values, error_values, y_values_original, y_values_approximated] = adam_optimization();
[~, ~, ~, ~, cost_values_adam, error_values_adam, ~, ~] = adam_optimization();

num_iterations = 1e4;

tic
[~, ~, ~, x_values_, y_values_original_, y_values_approximated_adam] = nonSmoothApproximation();
toc

figure;
plot(y_values_original_);
hold on;
plot(y_values_approximated_adam);
title("Adam Boost");
legend(["Original" "Approximation"]);
hold off;

% Plotting
% figure;
% plot(x_values, y_values_original, 'r-', 'LineWidth', 2); % Original function
% hold on;
% plot(x_values, y_values_approximated, 'b--', 'LineWidth', 2); % Approximated function
% xlabel('x');
% ylabel('y');
% title('Original Function vs. Approximated Function');
% legend('Original Function', 'Approximated Function');
% grid on;

% Plotting cost and error per iteration

figure;
subplot(2, 1, 1);
plot(1:num_iterations, cost_values_adam, 'b-', 'LineWidth', 2);
xlabel('Iterations');
ylabel('Cost');
title('Cost vs Iterations');
grid on;

subplot(2, 1, 2);
plot(1:num_iterations, error_values_adam, 'r-', 'LineWidth', 2);
xlabel('Iterations');
ylabel('Error');
title('Error vs Iterations');
grid on;

sgtitle("Adam Boost");

% Display final cost/error
fprintf('Final Cost: %.4f\n', cost_values_adam(num_iterations));
fprintf('Final Error: %.4f\n', error_values_adam(num_iterations));

%% Comparision of Algorithms

figure;
plot(y, 'b');
hold on;
plot(F, 'r');
plot(approx_curve);
plot(y_values_approximated_adam);
hold off;
title('Approximation Curves');
xlabel('Data Points');
ylabel('Values');
legend('Original', 'Primal-Dual', 'GD', 'Adam');
grid on;
hold off;

figure;
semilogy(cost_pd);
hold on;
semilogy(cost_gd);
semilogy(cost_values_adam);
hold off;
title('Cost vs Iterations');
xlabel('Iterations');
ylabel('Cost');
legend('Primal-Dual', 'GD', 'Adam');

figure;
semilogy(error_pd);
hold on;
semilogy(error_gd);
semilogy(error_values_adam);
hold off;
title('Error vs Iterations');
xlabel('Iterations');
ylabel('Error');
legend('Primal-Dual', 'GD', 'Adam');
