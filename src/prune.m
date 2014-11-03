function [ nn_params, hidden_layer_size ] = prune( nn_params, input_layer_size, ...
    hidden_layer_size, num_labels, ...
    X, y)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
nn_params = nn_params(numel(Theta1)+1:end);

Theta2 = reshape(nn_params(1:(hidden_layer_size + 1) * num_labels), ...
    num_labels, (hidden_layer_size + 1));
nn_params = nn_params(numel(Theta2)+1:end);

Omega = nn_params';

while hidden_layer_size > input_layer_size
    
    [Theta1, Theta2, Omega, ah, oh] = stoch_grad(Theta1, Theta2, Omega, ...
        X, y, 10^-2, 1, 0.3);

    p = zeros(1,hidden_layer_size);
    for i=1:hidden_layer_size
        
        varh = var(ah(i,:));
        varo = var(oh);
        covar = varh + varo;
        p(i) = covar / sqrt(varh * varo);
    end
    
end


end

