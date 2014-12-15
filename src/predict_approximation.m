function p = predict_approximation(nn_params, input_layer_size, ...
    hidden_layer_size, num_labels, X)
% Useful values
m = size(X, 1);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
nn_params = nn_params(numel(Theta1)+1:end);

Theta2 = reshape(nn_params(1:(hidden_layer_size + 1) * num_labels), ...
    num_labels, (hidden_layer_size + 1));
nn_params = nn_params(numel(Theta2)+1:end);

Omega = nn_params';
Omega = Omega(ones(m,1), :); 

z1 = [ones(m, 1) X ] * Theta1';
lat_con = z1 .* [Omega zeros(m,1)];
z1 = z1 + [zeros(m,1) lat_con(:, 1:end-1)];
h1 = sigmoid(z1);
p = [ones(m, 1) h1] * Theta2';

end
