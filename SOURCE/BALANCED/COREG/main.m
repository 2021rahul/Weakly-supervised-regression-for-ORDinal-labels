load('../../../DATA/D6/NUMPY/strong_data.mat')
load('../../../DATA/D6/NUMPY/weak_data.mat')
%%
x_s = strong_data(:,1:end-2);
y_s = strong_data(:,end-1);
x = weak_data(:,1:end-2);

[x_s_1, y_s_1, x_s_2, y_s_2] = train_model(x_s, y_s, x);
[y] = test_model(x, x_s_1, y_s_1, x_s_2, y_s_2);

%%

save('../../../DATA/D6/RESULT/IMBALANCED/COREG/Y.mat' ,'y')