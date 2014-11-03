function [nn_params, Jo, epoch] = stoch_grad(nn_params, input_layer_size, ...
    hidden_layer_size, num_labels, ...
    X, y, epsilon, alpha, phi)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
nn_params = nn_params(numel(Theta1)+1:end);

Theta2 = reshape(nn_params(1:(hidden_layer_size + 1) * num_labels), ...
    num_labels, (hidden_layer_size + 1));
nn_params = nn_params(numel(Theta2)+1:end);

Omega = nn_params';

m = size(X, 1);
Jo = inf;

epoch = 0;
Error = [];
while Jo > epsilon
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
    
    fprintf('Error: %.5f epoch: %i \n', Jo, epoch);
    Error = [Error Jo];
end

plot(1:length(Error), Error);
nn_params = [Theta1(:); Theta2(:); Omega(:)];

end
