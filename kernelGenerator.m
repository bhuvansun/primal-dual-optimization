function K = kernelGenerator(n, x)
    sigma = 1.0;
    K = zeros(n, n);
    
    for i = 1:n
        for j = 1:n
            K(i, j) = exp(-norm(x(i) - x(j))^2 / (2 * sigma^2));
        end
    end
end
