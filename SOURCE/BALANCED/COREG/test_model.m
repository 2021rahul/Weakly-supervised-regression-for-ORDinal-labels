function [y] = pred_coreg(x, x_s_1, y_s_1, x_s_2, y_s_2)

k1 = 3; % euclidean
k2 = 5; % minkowski


idx1 = knnsearch(x_s_1, x,'K', k1, 'Distance','euclidean');
y_pred_1 = mean(y_s_1(idx1),2);

idx2 = knnsearch(x_s_1, x ,'K', k2, 'Distance','minkowski');
y_pred_2 = mean(y_s_2(idx2),2);

y = (y_pred_1 + y_pred_2)/2;