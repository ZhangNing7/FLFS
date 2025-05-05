function [ W, obj ] = FLFS( X_train, Y_train, para )

[num_train, num_feature] = size(X_train); num_label = size(Y_train, 2);

%L, L0, and H
L = Laplacian_GK(X_train', para); 
L0 = Laplacian_GK(Y_train, para);
H = eye(num_train) - 1 / num_train * ones(num_train, 1) * ones(num_train, 1)';

%Initialize W
W = rand(num_feature, num_label); 
 S = [];
    para.mu  = 0.1;
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
        S(i,i) = 1; 
    end

iter = 1; obji = 1;
while 1
    %Update F--------------------------------------------------------------
    A = para.alpha *L +  para.alpha *H ;
    B = para.alpha * L0;
    C = - para.alpha * H * X_train * W ;
    F = lyap(A, B, C);  
    
    %Update W--------------------------------------------------------------
    d = 0.5./sqrt(sum(W.*W, 2) + eps);
    D = diag(d);
    W = (para.alpha * X_train' * H * X_train + para.gamma * D + para.beta*((eye(size(S)))-S)' *((eye(size(S)))-S) + eps*eye(num_feature)) \ (para.alpha * X_train' * H * F); 
    
    obj(iter) =  para.alpha*(norm((H*X_train*W - H*F), 'fro'))^2 +  ...
        + para.alpha*(trace(F'*L*F) +trace(F*L0*F')) + para.beta*(norm((W - S*W), 'fro'))^2 + para.gamma * sum(sqrt(sum(W.*W,2)+eps));

    cver = abs((obj(iter) - obji)/obji);
    obji = obj(iter);
    iter = iter + 1;
    if (cver < 10^-3 && iter > 2) , break, end
end

end

