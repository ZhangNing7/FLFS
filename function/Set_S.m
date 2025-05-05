S = [];
 para.mu     = 0.1;
    for i = 1:num_feature
        cvx_begin
        variable V2(num_feature);
        minimize(norm(X_train(:,i) - X_train*V2) + para.mu * norm(V2,1));
        subject to
            V2(i)==0;
        cvx_end
        S=[S V2];    
    end
    
%     S = rec_feature(X_train);
    for i = 1:num_feature
        S(i,i) = 1; % keep original features
    end