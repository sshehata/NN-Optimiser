function p = predict(Theta1, Theta2, X)
% Useful values
m = size(X, 1);

h1 = sigmoid([ones(m, 1) X ] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

p = h2 >= 0.5;

end
