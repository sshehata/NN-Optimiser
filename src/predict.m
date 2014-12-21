function p = predict(nn_params, input_layer_size, ...
    hidden_layer_size, num_labels, X)
% Useful values
m = size(X, 1);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
nn_params = nn_params(numel(Theta1)+1:end);

Theta2 = reshape(nn_params(1:(hidden_layer_size + 1) * num_labels), ...
    num_labels, (hidden_layer_size + 1));
nn_params = nn_params(numel(Theta2)+1:end);

Omega = reshape(nn_params, [hidden_layer_size,hidden_layer_size]);

z1 = [ones(m, 1) X ] * Theta1';
lat_con = Omega * z1';
z1 = z1 + lat_con';
h1 = sigmoid(z1);
h2 = sigmoid([ones(m, 1) h1] * Theta2');

if(size(h2,2) > 1),
  [~, p] = max(h2, [], 2);
else
  p = h2 >= 0.5;
end
end
