function [effR, M] = updateRewardMotivation(X, Y, M0, Mmax)
%UPDATEREWARDMOTIVATION Updates the reward and motivation in the task
%
%   Parameters
%   ==========
%   X            - number (X-coordinate of the agent)
%   Y            - number (Y-coordinate of the agent)
%   M0           - vector (components of motivation to each resourse)
%   Mmax         - vector (maximum values of motivation to each resourse)
%   effR         - double (subjective value of the reward)
%   M            - vector (updated motivation to each resourse)
%
%   Author
%   ======
%   Sergey Shuvaev, 2018-2021. sshuvaev@cshl.edu

% Define the locations where each of the resources is available

R = zeros(6, 6, 4);
R(1 : 3, 1 : 3, 1) = 1; %Food
R(1 : 3, 4 : 6, 2) = 1; %Water
R(4 : 6, 1 : 3, 3) = 1; %Sleep
R(4 : 6, 4 : 6, 4) = 1; %Play

% Compute the subjective value of the reward

effR = M0 * squeeze(R(X, Y, :));

% Update the value of motivation

M = M0 + 1; %Growth
M = M .* (squeeze(R(X, Y, :))' .* M0 == 0); %Reset
M = min(M, Mmax); %Threshold
