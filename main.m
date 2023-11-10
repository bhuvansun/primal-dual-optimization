n = 50;
left = -1;
right = 1;

x = linspace(left, right, n);
% y = -1 + (1-(-1)) .* rand(n,1);
% y = floor(x);
% y = x - floor(x);
% y = sign(x);
y = abs(x);
y = y';

% K = randn(n, n);
K = kernelGenerator(n, x);

lambda = 1e-3;
mu = 0.1;
delta = 0.1;
tol = 1e-6;
max_iters = 1e5;

alpha = primal_dual_tikhonov(K, y, lambda, mu, delta, tol, max_iters);

disp('Estimated Solution set (alpha):');
disp(alpha);

loss = (1/(2*n)) * norm(K*alpha - y, 2)^2 + (lambda/2) * norm(alpha, 2)^2;

F = K * alpha;

disp('Predicted values (F):');
disp(F);

disp('Actual values (y):');
disp(y);

% disp('MAE:');
% disp(sum(abs(F-y)));
% 
% disp('Loss function error:');
% disp(loss);

figure;
plot(y, 'b', 'LineWidth', 2);
title('Non-Smooth vs Smooth Curve');
xlabel('Data Points');
ylabel('Values');
hold on;
plot(F, 'r', 'LineWidth', 2);
legend('Original curve', 'Approximation Curve');
grid on;
hold off;
