function R_AVG = testAgent(MODEL_FILE, MMAX, DISPLAY, EPSILON)
%TESTAGENT tests a model on a Four Demands task
%
%   Parameters
%   ==========
%   MODEL_FILE   - string (e.g. 'tmp.mat')
%   MMAX         - double (maximum allowed motivation)
%   DISPLAY      - boolean (1 = display and write a GIF, 0 = don't)
%   EPSILON      - number (epsilon-greegy action choice)
%
%   Author
%   ======
%   Sergey Shuvaev, 2018-2021. sshuvaev@cshl.edu

addpath(genpath('Scripts'));

if nargin < 1
    MODEL_FILE = 'curriculum';
end
if nargin < 2
    MMAX = 10 * ones(1, 4);
end
if nargin < 3
    DISPLAY = 1;
end
if nargin < 4
    EPSILON = 1e-2;
end

TRIALS = 2e3;

load(fullfile('Models', MODEL_FILE), 'Mnet', 'MMAX_GLOB');

M = zeros(TRIALS, 4);
X = zeros(TRIALS, 1);
Y = zeros(TRIALS, 1);
A = zeros(TRIALS, 1);
R = zeros(TRIALS, 1);

%Initialize the agent
X(1) = randi(6);
Y(1) = randi(6);

%Run the model
for i = 2 : TRIALS
    A(i) = actionAgent(X(i - 1), Y(i - 1), M(i - 1, :), Mnet, MMAX,...
        MMAX_GLOB, EPSILON);
    [X(i), Y(i)] = updatePosition(X(i - 1), Y(i - 1), A(i));
    [R(i), M(i, :)] = updateRewardMotivation(X(i), Y(i), M(i - 1, :), MMAX);
end

R_AVG = mean(R(round(end / 2) : end));

if DISPLAY
    showTrace(X(end - 99 : end), Y(end - 99 : end), ...
        M(end - 99 : end, :), A(end - 99 : end));
end
end
