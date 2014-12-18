function [nn_params, j, epoch] = stoch_grad(nn_params, input_layer_size, ...
    hidden_layer_size, num_labels, ...
    X, y, epsilon, alpha, phi)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
nn_params = nn_params(numel(Theta1)+1:end);
Theta2 = reshape(nn_params(1:(hidden_layer_size + 1) * num_labels), ...
    num_labels, (hidden_layer_size + 1));
nn_params = nn_params(numel(Theta2)+1:end);
Omega = reshape(nn_params,hidden_layer_size,hidden_layer_size);


m = size(X, 1);
j = inf;

epoch = 0;
Error = [];
while j > epsilon & epoch < 1000,
    epoch = epoch + 1;
    j = 0;
    for sample= 1: m,
        % Forward Propagation
        a1 = [1 X(sample,:)];
        z2 = a1 * Theta1';
        % calculate lateral connection
        lat_con = (Omega * z2')';
        z2 = z2 + lat_con;
        a2 = [1 sigmoid(z2)];
        z3 = a2 * Theta2';
        a3 = sigmoid(z3);

        Jo = sum((y(sample) - a3).^2);
        j = j + Jo;

        % BackPropagation algorithm
        delta_3 = (y(sample) - a3) .* a3 .* (1 - a3);
        delta_2 = (delta_3 * Theta2) .* ( a2 .* ( 1 - a2 ));
        Theta2_delta = delta_3' * a2;
        Theta1_delta = delta_2' * a1;
        Theta1_delta = Theta1_delta(2:end,:);
%        this is the first line
        Omega_delta = Omega * (delta_2(2:end))' *  (a2(2:end) .* (1 - a2(2:end)));
%         Omega_delta = [zeros(1, size(Omega, 2)) ; a1' * delta_2(2:end)];
        Omega_delta = tril(Omega_delta,-1);
        Theta2 = Theta2 + alpha * Theta2_delta;
        Theta1 = Theta1 + alpha * Theta1_delta;
        Omega = Omega + alpha * Omega_delta;
        % updating ThetaH and OmegaH using Jh
        Theta1_delta = (a2 .* a2 .* (1 - a2))' * a1;
        Theta1_delta = Theta1_delta(2:end,:);
%        this is the second line
        Omega_delta = a2(2:end)' * ((a2(2:end) .* (1 - a2(2:end))) .* a2(2:end));
        Theta1 = Theta1 - phi * Theta1_delta;
        Omega_delta = tril(Omega_delta,-1);
        Omega = Omega - phi * Omega_delta;
    end
    fprintf('Error: %.5f epoch: %i \n', j, epoch);
    Error = [Error j/m];
end

figure('name','data')
plot(1:length(Error), Error);
nn_params = [Theta1(:); Theta2(:); Omega(:)];

end


%         Omega_delta  = delta_2' * lat_con
%         Omega_delta = Omega_delta(2:end,:)
%         for i = 1: hidden_layer_size,
%           for j =1: i,
%             Omega_delta(j,i) = 0;
%           end
%         end
%         Omega_delta 