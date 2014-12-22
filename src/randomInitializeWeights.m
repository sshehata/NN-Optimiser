function [W] = randomInitializeWeights(in, out)
%   epsilon is used so that the range of weights at the begining is going
%   to be -epsilon till epsilon
  epsilon = 0.2;
  W = rand(out, in+1) * 2 * epsilon - epsilon;
end