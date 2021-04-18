function [Mnet, R_AVG] = trainAgent(MMAX, ISMOTIVATED, GAMMA, BATCH_SIZE)
%TRAINAGENT trains a deep Q-model for the Four Demands task
%
%   Parameters
%   ==========
%   MMAX         - function (maximum allowed motivation)
%   GAMMA        - double (temporal discount factor)
%   ISMOTIVATED  - boolean (1 = motivated, 0 = unmotivated)
%   TRIALS       - number (of training iterations)
%   BATCH_SIZE   - number (of parallel tasks)
%   LRATE        - function (learning rate schedule)
%   EPSILON      - function (epsilon-greegy action choice)
%   N            - number (of network inputs)
%   SZ           - number (of units in each hidden layer)
%   R_AVG        - number (average reward at the end of training)
%
%   Author
%   ======
%   Sergey Shuvaev, 2018-2021. sshuvaev@cshl.edu

addpath(genpath('Scripts'));

if nargin < 1
    MMAX = @(agent) agent * ones(1, 4);
end
if nargin < 2
    ISMOTIVATED = 1;
end
if nargin < 3
    GAMMA = 0.9;
end
if nargin < 4
    BATCH_SIZE = 10;
end
TRIALS = 4e5;
LRATE = @(i) 3e-4 * 1e-2 ^ (i / TRIALS);
EPSILON = @(i) 0.5 * 0.02 ^ (i / TRIALS);
N = 41;
SZ = 100;

%Define the network structure
%    #                 TYPE      INDIM     OUTIM     WDIM     NLTYPE     OS
Mnet(1)=DLNetworkLayer('input',	 [N 1 1],  [N 1 1],  [],      [],        []);
Mnet(2)=DLNetworkLayer('full',	 [N 1 1],  [SZ 1 1], [SZ N],  'sigmoid', []);
Mnet(3)=DLNetworkLayer('full',	 [SZ 1 1], [SZ 1 1], [SZ SZ], 'sigmoid', []);
Mnet(4)=DLNetworkLayer('full',	 [SZ 1 1], [SZ 1 1], [SZ SZ], 'sigmoid', []);
Mnet(5)=DLNetworkLayer('full',	 [SZ 1 1], [5 1 1],	 [5 SZ],  'linear',  []);
Mnet(6)=DLNetworkLayer('target', [5 1 1],  [5 1 1],	 [],      [],        []);

len = length(Mnet);

%Initialize the agents
for k = 1 : BATCH_SIZE
    ag{k}.MMAX = MMAX(k);
    ag{k}.M = [0 0 0 0];
    ag{k}.x = randi(6);
    ag{k}.y = randi(6);
end

%Train the network
R = zeros(TRIALS, BATCH_SIZE) * NaN; %Effective reward
TD_ERROR = zeros(TRIALS, BATCH_SIZE) * NaN; %Delta (Reward presiction error)

MMAX_GLOB = 0;
for i = 1 : BATCH_SIZE
    MMAX_GLOB = max(MMAX_GLOB, max(ag{i}.MMAX(:)));
end
tic

for i = 2 : TRIALS
    for k = 1 : BATCH_SIZE

        %Take an action
        [action, Q0] = actionAgent(ag{k}.x, ag{k}.y, ...
            ag{k}.M * ISMOTIVATED, Mnet, ag{k}.MMAX, MMAX_GLOB, EPSILON(i));
        [Xnew, Ynew] = updatePosition(ag{k}.x, ag{k}.y, action);
        [R(i, k), Mnew] = updateRewardMotivation(Xnew, Ynew, ag{k}.M, ...
            ag{k}.MMAX);
        [~, Q] = actionAgent(Xnew, Ynew, Mnew * ISMOTIVATED, Mnet, ...
            ag{k}.MMAX, MMAX_GLOB, EPSILON(i));
        actionAgent(ag{k}.x, ag{k}.y, ag{k}.M * ISMOTIVATED, Mnet, ...
            ag{k}.MMAX, MMAX_GLOB, EPSILON(i));
        ag{k}.M = Mnew; ag{k}.x = Xnew; ag{k}.y = Ynew;

        %Perform the TD update
        TD_ERROR(i, k) = R(i, k) + GAMMA * max(squeeze(Q)) - Q0(action);
        delta_vector = zeros(5, 1);
        delta_vector(action) = -TD_ERROR(i, k);
        Mnet(len - 1).delta = delta_vector;
        for j = len - 1 : - 1 : 2
            stepBackward(Mnet, j, LRATE(i), 0.0, 0.0, (k == BATCH_SIZE));
        end
    end

    %Plot the progress
    if ~mod(i,1000)
        subplot(2, 1, 1);
        loglog(conv2(abs(TD_ERROR), ones(1000, 1)) / 1000), grid
        title('TD error')
        subplot(2, 1, 2);
        semilogx(conv2(abs(R), ones(1000, 1)) / 1000), grid
        line([1e3 i], [10/3, 10/3], 'color', 'black');
        line([1e3 i], [3.5, 3.5], 'color', 'black');
        title('Effective reward')
        xlabel('Trial number')
        drawnow
    end
end

t = toc;
fprintf('\nTraining time: %d min %d sec.\n', ...
    floor(t / 60), round(t - floor(t / 60) * 60));

%Remove training-related variables and save the model
for i = 2 : len - 1
    Mnet(i).strip();
end
save(fullfile('Models', 'tmp.mat'), 'Mnet', 'MMAX_GLOB')

R_AVG = mean(R(end - 999 : end, :));
end
