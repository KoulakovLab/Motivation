function showTrace(x, y, m, a)
%SHOWTRACE displays a model's behavior
%
%   Parameters
%   ==========
%   X            - number (X-coordinate of the agent)
%   Y            - number (Y-coordinate of the agent)
%   M            - vector (updated motivation to each resourse)
%   A            - number (next action generated by the model)
%   TAIL_LENGTH  - number (of previous steps to be displayed)
%   CMP          - array (colormap to display motivation)
%
%   Author
%   ======
%   Sergey Shuvaev, 2018-2021. sshuvaev@cshl.edu

TAIL_LENGTH = 6;
CMP = flip(summer, 1) .^ 4;

figure, set(gcf,'color','w');
MMAX = max(m(:));
m = m / MMAX;
IM = ones(700, 600, 3);

for k = TAIL_LENGTH + 2 : length(x)
    
    %Background color in four rooms
    IM(51 : 350, 1 : 300, 1) = CMP(round(1 + m(k, 1) * 63), 1);
    IM(51 : 350, 1 : 300, 2) = CMP(round(1 + m(k, 1) * 63), 2);
    IM(51 : 350, 1 : 300, 3) = CMP(round(1 + m(k, 1) * 63), 3);
    IM(51 : 350, 301 : 600, 1) = CMP(round(1 + m(k, 2) * 63), 1);
    IM(51 : 350, 301 : 600, 2) = CMP(round(1 + m(k, 2) * 63), 2);
    IM(51 : 350, 301 : 600, 3) = CMP(round(1 + m(k, 2) * 63), 3);
    IM(351 : 650, 1 : 300, 1) = CMP(round(1 + m(k, 3) * 63), 1);
    IM(351 : 650, 1 : 300, 2) = CMP(round(1 + m(k, 3) * 63), 2);
    IM(351 : 650, 1 : 300, 3) = CMP(round(1 + m(k, 3) * 63), 3);
    IM(351 : 650, 301 : 600, 1) = CMP(round(1 + m(k, 4) * 63), 1);
    IM(351 : 650, 301 : 600, 2) = CMP(round(1 + m(k, 4) * 63), 2);
    IM(351 : 650, 301 : 600, 3) = CMP(round(1 + m(k, 4) * 63), 3);

    %Colorbars for each motivation
    IM(1 : 30, 11 : 290, 1) = repmat(CMP(round(1 : 63/279 : 64), 1), [1 30])';
    IM(1 : 30, 11 : 290, 2) = repmat(CMP(round(1 : 63/279 : 64), 2), [1 30])';
    IM(1 : 30, 11 : 290, 3) = repmat(CMP(round(1 : 63/279 : 64), 3), [1 30])';
    IM(1 : 30, 311 : 590, 1) = repmat(CMP(round(1 : 63/279 : 64), 1), [1 30])';
    IM(1 : 30, 311 : 590, 2) = repmat(CMP(round(1 : 63/279 : 64), 2), [1 30])';
    IM(1 : 30, 311 : 590, 3) = repmat(CMP(round(1 : 63/279 : 64), 3), [1 30])';
    IM(671 : 700, 11 : 290, 1) = repmat(CMP(round(1 : 63/279 : 64), 1), [1 30])';
    IM(671 : 700, 11 : 290, 2) = repmat(CMP(round(1 : 63/279 : 64), 2), [1 30])';
    IM(671 : 700, 11 : 290, 3) = repmat(CMP(round(1 : 63/279 : 64), 3), [1 30])';
    IM(671 : 700, 311 : 590, 1) = repmat(CMP(round(1 : 63/279 : 64), 1), [1 30])';
    IM(671 : 700, 311 : 590, 2) = repmat(CMP(round(1 : 63/279 : 64), 2), [1 30])';
    IM(671 : 700, 311 : 590, 3) = repmat(CMP(round(1 : 63/279 : 64), 3), [1 30])';

    image(IM); axis image; hold on;

    %Room borders
    line([300 300], [050 150], 'color', 'black', 'linewidth', 6);
    line([300 300], [250 450], 'color', 'black', 'linewidth', 6);
    line([300 300], [550 650], 'color', 'black', 'linewidth', 6);
    line([000 100], [350 350], 'color', 'black', 'linewidth', 6);
    line([200 400], [350 350], 'color', 'black', 'linewidth', 6);
    line([500 600], [350 350], 'color', 'black', 'linewidth', 6);
    rectangle('Position', [0 50 600 600], 'linewidth', 6)
    
    %State borders
    for i = 0 : 100 : 600
        line([i, i], [50, 650], 'color', 'black', 'linewidth', .5,'LineStyle','--')
        line([0, 600], [i, i] + 50, 'color', 'black', 'linewidth', .5,'LineStyle','--')
    end
    
    %Agent's (fading) trace
    lines = arrayfun(@(x) line(NaN,NaN, 'linewidth', 4), 1 : TAIL_LENGTH, 'uni', 0);
    recolor = @(lines) arrayfun(@(x) set(lines{x}, ...
        'Color', [zeros(1, 3), 1 - (x / TAIL_LENGTH)]), 1 : TAIL_LENGTH);
    for j = 1 : TAIL_LENGTH
        set(lines{j}, 'XData', -50 + y((k : k + 1) - j) * 100, ...
            'YData', x((k : k + 1) - j) * 100);
    end
    recolor(lines);
    
    %Agent's current position
    if a(k) ~= 5
        plot(-50 + y(k) * 100, x(k) * 100, 'k.', 'markersize', 40)
    else
        plot(-50 + y(k) * 100, x(k) * 100, 'k.', 'markersize', 60)
        plot(-50 + y(k) * 100, x(k) * 100, 'w.', 'markersize', 30)
    end
    
    %Colorbar borders
    rectangle('Position', [9.5 -0.5 281 31])
    rectangle('Position', [9.5 669.5 281 31])
    rectangle('Position', [309.5 -0.5 281 31])
    rectangle('Position', [309.5 669.5 281 31])
    
    %Agent's current motivation on colorbars
    line(10 + m(k, 1) * [280 280], [000 030], 'color', 'black', 'linewidth', 4);
    line(10 + m(k, 2) * [280 280] + 300, [000 030], 'color', 'black', 'linewidth', 4);
    line(10 + m(k, 3) * [280 280], [670 700], 'color', 'black', 'linewidth', 4);
    line(10 + m(k, 4) * [280 280] + 300, [670 700], 'color', 'black', 'linewidth', 4);
    
    %Colorbar labels
    xticks([10 + (0 : 0.2 : 1) * 280, 310 + (0.2 : 0.2 : 1) * 280])
    xticklabels([(0 : 0.2 : 1), (0.2 : 0.2 : 1)] * MMAX)
    yticks([])
    
    %Drawing the image
    gf = gca;
    gf.YRuler.Axle.LineStyle = 'none';
    gf.XRuler.Axle.LineStyle = 'none';
    drawnow
    hold off
    pause(0.1);
    
    %Writing the image to a GIF file
    frame = getframe;
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if k == TAIL_LENGTH + 2
        imwrite(imind, cm, 'evaluation.gif', 'gif', ...
            'Loopcount', inf, 'delaytime', .2);
    else
        imwrite(imind, cm, 'evaluation.gif', 'gif', ...
            'WriteMode', 'append', 'delaytime', .2);
    end
end
