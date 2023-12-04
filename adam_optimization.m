% Adam optimization algorithm within the interval [-1, 1]
function [optimal_x, min_loss, approximated_function, x_values, cost_values, error_values, y_values_original, y_values_approximated] = adam_optimization()

    % Initialize variables
    alpha = 0.01; % Learning rate
    beta1 = 0.9;  % Exponential decay rates for moment estimates
    beta2 = 0.999;
    epsilon = 1e-8; % Small value to prevent division by zero
    
    % Number of iterations
    num_iterations = 1e4;
    
    % Initial value within the interval [-1, 1]
    x = 0.5; % Initial guess
    
    % Arrays to store values for plotting and tracking cost/error
    x_values = [];
    y_values_original = [];
    y_values_approximated = [];
    cost_values = zeros(1, num_iterations);
    error_values = zeros(1, num_iterations);
    
    % Adam optimization loop
    for i = 1:num_iterations
        
        % Compute gradient of the non-smooth function at x
        gradient = nonSmoothGradient(@nonSmoothLoss, x, epsilon);
        
        % Adam algorithm steps (as in previous examples)
        m = 0;
        v = 0;
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * (gradient.^2);
        
        m_hat = m / (1 - beta1^i);
        v_hat = v / (1 - beta2^i);
        
        x = x - alpha * m_hat ./ (sqrt(v_hat) + epsilon);
        % Ensure x remains within the interval [-1, 1]
        x = max(-1, min(1, x));
        
        % Store values for plotting
        x_values = [x_values, x];
        y_values_original = [y_values_original, nonSmoothLoss(x)];
        y_values_approximated = [y_values_approximated, nonSmoothLoss(x)];
        
        % Calculate cost/error for current iteration
        cost_values(i) = sum(y_values_approximated.^2) / i;
        error_values(i) = sqrt(cost_values(i));
    end
    
    % Output optimal x, minimum loss, approximated function, and tracked values
    optimal_x = x;
    min_loss = nonSmoothLoss(x);
    approximated_function = y_values_approximated;
    
    
end

% Define your non-smooth loss function within the interval [-1, 1]
function loss = nonSmoothLoss(x)
    % Ensure x remains within the interval [-1, 1]
    x = max(-1, min(1, x));
    % Absolute value function within the interval [-1, 1]
    loss = abs(x);
end

% Gradient of the non-smooth loss function
function gradient = nonSmoothGradient(f, x, epsilon)
    % Gradient calculation for the non-smooth function at point x
    f_plus = f(x + epsilon);
    f_minus = f(x - epsilon);
    gradient = (f_plus - f_minus) / (2 * epsilon);
end
