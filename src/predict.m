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
Omega = reshape(nn_params,hidden_layer_size,hidden_layer_size);
 
a3 = zeros(m,1);
if(num_labels > 2),
  a3 = zeros(m,num_labels);
end
for sample= 1: m,
    % Forward Propagation
    a1 = [1 X(sample,:)];
    z2 = a1 * Theta1';
    % calculate lateral connection
    lat_con = z2 * Omega;
    lat_con = [0 lat_con(1:end-1)];
    z2 = z2 + lat_con;
    a2 = [1 sigmoid(z2)];
    z3 = a2 * Theta2';
    a3(sample,:) = sigmoid(z3);
end

size(a3)
if(size(a3,2) > 1),
  [~, p] = max(a3, [], 2);
else
  p = a3 >= 0.5;
end
end
