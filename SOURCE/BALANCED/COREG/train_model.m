function [x_s_1, y_s_1, x_s_2, y_s_2] = learn_model_coreg(x_s, y_s, x)

k1 = 3; % euclidean
k2 = 5; % mahalanobis

sample_size = 800;

ind_pool = (1:size(x,1))';
x_s_1 = x_s; y_s_1 = y_s;
x_s_2 = x_s; y_s_2 = y_s;

flag_iter = 1; % if anything changes
ind_iter = randsample(ind_pool, sample_size);

num_iter = 500
for iter = 1:num_iter
  iter
  if(flag_iter == 0)
    return;
  end
  
  flag_iter = 0;
  flag_iter1 = 0;
  flag_iter2 = 0;
  
  % Euclidean
  idx1 = knnsearch(x_s_1,x(ind_iter,:),'K', k1, 'Distance','euclidean');
  y_pred_1 = mean(y_s_1(idx1),2);
  err_iter = zeros(length(ind_iter),1);
  for i = 1:length(ind_iter)
    idx_i = knnsearch([x_s_1;x(ind_iter(i),:)], x_s_1(idx1(i,:),:),'K', k1, 'Distance','euclidean');
    tmp = [y_s_1; y_pred_1(i)];
    new_err = y_s_1(idx1(i,:)) - mean(tmp(idx_i),2);
    idx_i_original = knnsearch(x_s_1, x_s_1(idx1(i,:),:),'K', k1, 'Distance','euclidean');
    old_err = y_s_1(idx1(i,:)) - mean(y_s_1(idx_i_original),2);
    err_iter(i) = old_err'*old_err - new_err'*new_err;
  end
  
  if(sum(err_iter>0) >0)
    [~, ix_max_1] = max(err_iter);
    x_max_1 = x(ind_iter(ix_max_1),:);
    y_max_1 = y_pred_1(ix_max_1);
    ind_iter = setdiff(ind_iter, ix_max_1);
    flag_iter1 = 1;
  end
    
    
  % mahalanobis
  idx2 = knnsearch(x_s_2, x(ind_iter,:),'K', k2, 'Distance','minkowski');
  y_pred_2 = mean(y_s_2(idx2),2);
  err_iter = zeros(length(ind_iter),1);
  for i = 1:length(ind_iter)
    idx_i = knnsearch([x_s_2;x(ind_iter(i),:)], x_s_2(idx2(i,:),:),'K', k2, 'Distance','minkowski');
    tmp = [y_s_2; y_pred_2(i)];
    new_err = y_s_2(idx2(i,:)) - mean(tmp(idx_i),2);
    %old_err = y_s_2(idx2(i,:)) - y_pred_2(idx2(i,:));
    idx_i_original = knnsearch(x_s_2, x_s_2(idx2(i,:),:),'K', k2, 'Distance','minkowski');
    old_err = y_s_2(idx2(i,:)) - mean(y_s_2(idx_i_original),2);
    err_iter(i) = old_err'*old_err - new_err'*new_err;
  end
  
  if(sum(err_iter>0) >0)
    [~, ix_max_2] = max(err_iter);
    x_max_2 = x(ind_iter(ix_max_2),:);
    y_max_2 = y_pred_2(ix_max_2);
    ind_iter = setdiff(ind_iter, ix_max_2);
    flag_iter2 = 1;
  end
  
  
  % set flags
  flag_iter = flag_iter1 + flag_iter2;
  
  if(flag_iter1 == 1)
    x_s_2 = [x_s_2; x_max_1];
    y_s_2 = [y_s_2; y_max_1];
  end
  
  if(flag_iter2 == 1)
    x_s_1 = [x_s_1; x_max_2];
    y_s_1 = [y_s_1; y_max_2];
  end
  
  if(flag_iter >0)
    ind_iter = [ind_iter; randsample(ind_pool, sample_size - length(ind_iter))];
  end
  
end

