function [Theta1, Theta2, Omega] = stoch_grad(Theta1, Theta2, Omega, ...
    X, y, epsilon, alpha, phi)

m = size(X, 1);
Jo = inf;

epoch = 0;
Error = [];

while Jo > epsilon && epoch < 40000
    epoch = epoch + 1;
    Jo = 0;
    for sample=1:m
        % Feedforward the neural network and calculate the cost in J
        a1 = [1 X(sample,:)];
        z2 = a1 * Theta1';
        
        % calculate lateral connection
        lat_con = z2 .* [Omega 0];
        z2 = z2 + [0 lat_con(1:end-1)];
        
        a2 = [1 sigmoid(z2)];
        z3 = a2 * Theta2';
        a3 = sigmoid(z3);
        
        e = 0.5 * (y(sample) - a3)^2;
        Jo = Jo + e;
        
        % Vectorized BackPropagation algoritm
        delta_3 = (y(sample) - a3) * a3 * (1 - a3);
        delta_2 = (delta_3 * Theta2) .* ( a2 .* ( 1 - a2 ));
        Theta2_delta = delta_3 * a2;
        Theta1_delta = delta_2' * a1;
        Theta1_delta = Theta1_delta(2:end,:);
        Omega_delta = delta_2(3:end) .* Omega .*  a2(2:end-1) .* (1 - a2(2:end-1));
        
        Theta2 = Theta2 + alpha * Theta2_delta;
        Theta1 = Theta1 + alpha * Theta1_delta;
        Omega = Omega + alpha * Omega_delta;
        
        % updating ThetaH and OmegaH using Jh
        Theta1_delta = (a2 .* a2 .* (1 - a2))' * a1;
        Theta1_delta = Theta1_delta(2:end,:);
        Omega_delta = a2(3:end) .* a2(3:end) .* (1 - a2(3:end)) .* a2(2:end-1);
        Theta1 = Theta1 - phi * Theta1_delta;
        Omega = Omega - phi * Omega_delta;
    end
    
    Error = [Error Jo];
end

fprintf('Error: %.5f \n', Jo);
if (epoch > 40000)
    fprintf('Training terminated after %i epochs\n', epoch);
else
    fprintf('Network converged after %i epochs\n', epoch);
end

plot(1:epoch, Error(1:epoch));

end
