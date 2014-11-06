function [ nn_params, hidden_layer_size ] = prune( nn_params, input_layer_size, ...
    hidden_layer_size, num_labels, ...
    X, y)

m = size(X,1);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));
nn_params = nn_params(numel(Theta1)+1:end);

Theta2 = reshape(nn_params(1:(hidden_layer_size + 1) * num_labels), ...
    num_labels, (hidden_layer_size + 1));
nn_params = nn_params(numel(Theta2)+1:end);

Omega = nn_params';

while hidden_layer_size > input_layer_size + 1
    
    [Theta1, Theta2, Omega] = stoch_grad(Theta1, Theta2, Omega, ...
        X, y, 10^-2, 1, 0.3);
    
    ah = zeros(m, hidden_layer_size);
    ao = zeros(m, num_labels);
    
    for sample=1:m
        a1 = [1 X(sample,:)];
        z2 = a1 * Theta1';
        
        % calculate lateral connection
        lat_con = z2 .* [Omega 0];
        z2 = z2 + [0 lat_con(1:end-1)];
        
        a2 = [1 sigmoid(z2)];
        z3 = a2 * Theta2';
        a3 = sigmoid(z3);
        
        ah(sample, :) = a2(2:end);
        ao(sample, :) = a3;
        
    end
    
    p = zeros(hidden_layer_size, 1);
    for i=1:hidden_layer_size
        covar = cov(ah(:,i), ao);
        varh = var(ah(:,i));
        varo = var(ao);
        if (varh == 0 || varo == 0)
            p(i) = 0;
        else
            p(i) = abs(covar(2)) / sqrt(varh * varo);
        end
    end
    
    [n, nidx] = min(p);
    % remove node
    if (nidx < hidden_layer_size)
        Theta1 = [Theta1(1:nidx -1, :) ; Theta1(nidx + 1: end, :)];
        Theta2 = [Theta2(:, 1:nidx)  Theta2(:, nidx + 2:end)];
        
    else
        Theta1 = Theta1(1:nidx -1, :);
        Theta2 = Theta2(:, 1:nidx);
        
    end
    
    if (nidx == hidden_layer_size)
        Omega = Omega(:, 1:end - 1) ;
    elseif (nidx == hidden_layer_size - 1)
        Omega = Omega(:, 1:nidx - 1) ;
    else
        Omega = [Omega(:, 1:nidx - 1)  Omega(:, nidx+1)];
    end
    hidden_layer_size = hidden_layer_size - 1;
end

[Theta1, Theta2, Omega] = stoch_grad(Theta1, Theta2, Omega, ...
        X, y, 10^-2, 1, 0.3);
    
nn_params = [Theta1(:); Theta2(:); Omega(:)];


end

