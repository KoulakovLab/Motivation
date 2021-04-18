%PHASEDIAGRAM evaluates model's behavior under a range of parameters
%
%   Parameters
%   ==========
%   MMAX         - double (maximum allowed motivation)
%   N_WORKERS    - number (of agents evaluated per each set of parameters)
%
%   Author
%   ======
%   Sergey Shuvaev, 2018-2021. sshuvaev@cshl.edu

close all
clear
clc

N_WORKERS = 10;

%Set maximum allowed motivations
M = (10 .^ (log10(1) : 0.05 : log10(100)));
R_MOTIVATED = zeros(length(M), N_WORKERS);
R_UNMOTIVATED = zeros(length(M), N_WORKERS);
R_RANDOMWALK = zeros(length(M), N_WORKERS);

%Evaluate the model
for i = 1 : length(M)
    fprintf('Iteration %d\n', i)
    parfor j = 1 : N_WORKERS
        MMAX = @(x) M(i) * ones(1, 4);
        [~, R_MOTIVATED(i, j)] = trainAgent(MMAX, 1, 0.9, 1); %Motivated
        [~, R_UNMOTIVATED(i, j)] = trainAgent(MMAX, 0, 0.9, 1); %Unmotivated
        R_RANDOMWALK(i, j) = testAgent('curriculum', M(i) * ones (1, 4), 0, 1); %Random
        close all
    end
end

%Plot the average reward values for each model
figure, scatter(repmat(M, 1, N_WORKERS), R_MOTIVATED(:), 70, 'filled', 'markerfacealpha', 0.2);
axis([1 100 0.25 3.5]);
set(gca, 'xscale', 'log')
hold on
scatter(repmat(M, 1, N_WORKERS), R_RANDOMWALK(:), 70, 'filled', 'markerfacealpha', 0.3);
scatter(repmat(M, 1, N_WORKERS), R_UNMOTIVATED(:), 70, 'filled', 'markerfacealpha', 0.3);
box on

%Plot theoretical predictions
line([1 100], [10/3, 10/3], 'color', 'black', 'linestyle', '--');
line([1 100], [4/3, 4/3], 'color', 'black', 'linestyle', '--');
line([1 100], [1, 1], 'color', 'black', 'linestyle', '--');
X = (3 : 0.01 : 9);
Y = 1/3 + X / 3;
plot(X, Y, 'k--');
X = (3/2 : 0.01 : 2);
Y = 2 * X / 3;
plot(X, Y, 'k--');

%Display captions
text(1.1, 0.9, 'binging');
text(2.1, 1.2, 'delayed binging');
text(6.5, 2.2, 'cycling');

xlabel('Maximum allowed motivation');
ylabel('Average reward intake')
title('Learned behaviors')
