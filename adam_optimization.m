function [optimal_x, min_loss, approximated_function, x_values, cost_values, error_values, y_values_original, y_values_approximated] = adam_optimization()

    alpha = 0.01;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    
    num_iterations = 1e4;
    
    x = 0.5;
    
    x_values = [];
    y_values_original = [];
    y_values_approximated = [];
    cost_values = zeros(1, num_iterations);
    error_values = zeros(1, num_iterations);
    
    for i = 1:num_iterations
        
        gradient = nonSmoothGradient(@nonSmoothLoss, x, epsilon);
        
        m = 0;
        v = 0;
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * (gradient.^2);
        
        m_hat = m / (1 - beta1^i);
        v_hat = v / (1 - beta2^i);
        
        x = x - alpha * m_hat ./ (sqrt(v_hat) + epsilon);
        
        x_values = [x_values, x];
        y_values_original = [y_values_original, nonSmoothLoss(x)];
        y_values_approximated = [y_values_approximated, nonSmoothLoss(x)];
        
        cost_values(i) = sum(y_values_approximated.^2) / i;
        error_values(i) = sqrt(cost_values(i));
        
    end
    
    optimal_x = x;
    min_loss = nonSmoothLoss(x);
    approximated_function = y_values_approximated; 
    
end

function loss = nonSmoothLoss(x)
    loss = abs(x);
end

function gradient = nonSmoothGradient(f, x, epsilon)
    f_plus = f(x + epsilon);
    f_minus = f(x - epsilon);
    gradient = (f_plus - f_minus) / (2 * epsilon);
end
