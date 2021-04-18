function [X, Y] = updatePosition(X0, Y0, action)
%UPDATEPOSITION Updates the agent's new position in the task
%
%   Parameters
%   ==========
%   X0           - number (X-coordinate of the agent)
%   Y0           - number (Y-coordinate of the agent)
%   action       - number (action)
%   X            - number (new X-coordinate of the agent)
%   Y            - number (new Y-coordinate of the agent)
%
%   Author
%   ======
%   Sergey Shuvaev, 2018-2021. sshuvaev@cshl.edu

%Define possible actions: 1=left, 2=right, 3=up, 4=down, 5=stay
%For every action, define the coordinates where the action is available.

A = ones(6, 6, 5);
A(:, :, 1) = [...
    0 1 1 0 1 1
    0 1 1 1 1 1
    0 1 1 0 1 1
    0 1 1 0 1 1
    0 1 1 1 1 1
    0 1 1 0 1 1];
A(:, :, 2) = flip(A(:, :, 1), 2);
A(:, :, 3) = A(:, :, 1)';
A(:, :, 4) = flip(A(:, :, 3), 1);

%Update the agent's position

X = X0;
Y = Y0;
        
switch action
    case 1 %Left
        Y = Y0 - A(X0, Y0, 1);
    case 2 %Right
        Y = Y0 + A(X0, Y0, 2);
    case 3 %Up
        X = X0 - A(X0, Y0, 3);
    case 4 %Down
        X = X0 + A(X0, Y0, 4);
end
