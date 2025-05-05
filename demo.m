clc; clear; 
addpath(genpath('.\'))

load('Scene_data.mat')

[num_train, num_label] = size(Y_train); [num_test, num_feature] = size(X_test);

pca_remained =50 ;

all = [X_train; X_test]; 
ave = mean(all);
all = (all'-concur(ave', num_train + num_test))';

covar = cov(all); covar = full(covar);

[u,s,v] = svd(covar);

t_matrix = u(:, 1:pca_remained)';
all = (t_matrix * all')';

X_train = all(1:num_train,:); X_test = all((num_train + 1):(num_train + num_test),:);
    
para.alpha = 1; para.beta = 1; para.gamma = 100; para.k = num_label - 1;

t0 = clock;
[ W, obj ] = FLFS( X_train, Y_train, para );
time = etime(clock, t0);

[dumb idx] = sort(sum(W.*W,2),'descend'); 
feature_idx = idx(1:pca_remained);

Num = 10;Smooth = 1;  

for i = 1:pca_remained
    fprintf('Running the program with the selected features - %d/%d \n',i,pca_remained);
    
    f=feature_idx(1:i);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(X_train(:,f),Y_train',Num,Smooth);
    [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
        MLKNN_test(X_train(:,f),Y_train',X_test(:,f),Y_test',Num,Prior,PriorN,Cond,CondN);
    
    HL_FLFS(i)=HammingLoss;
    RL_FLFS(i)=RankingLoss;
    CV_FLFS(i)=Coverage;
    AP_FLFS(i)=Average_Precision;
    MA_FLFS(i)=macrof1;
    MI_FLFS(i)=microf1;
end

save('Scene_FLFS.mat','HL_FLFS','RL_FLFS','CV_FLFS'...
    ,'AP_FLFS','MA_FLFS' ,'MI_FLFS','feature_idx','time');

