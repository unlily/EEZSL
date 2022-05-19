clear; clc;
load('dataroot\aPY\attributes.mat');
load('dataroot\aPY\feature.mat');
for i=1:length(train_label)
    for b=1:length(trainClassLabels)
        if(train_label(i)==trainClassLabels(1,b))
            S_tr(i,:)= S_tr_gt(b,:);
        end
    end
end

X_tr    = NormalizeFea(X_tr')';
X_te    = NormalizeFea(X_te')';
S_tr    = NormalizeFea(S_tr')';
dimFeature=size(X_tr,2);
dimSemantic = size(S_te_gt,2);


%% train 
H=zeros(length(train_label),length(trainClassLabels));
for i=1:length(train_label)
    for b=1:length(trainClassLabels)
        if(train_label(i)==trainClassLabels(b))
            H(i,b)=1; 
        end
    end
end
W1      = initializeParameters(dimSemantic,dimFeature);
W2      = initializeParameters(dimFeature,dimFeature);
lambda1  =15;
lambda2  =10;

bata=1;
eta=0.2;
k=1;
for i=1:20
    %df1=2*(W1*X_tr'*X_tr-W2*S_tr'*X_tr+W2*S_tr_gt'*S_tr_gt*W2'*W1*X_tr'*X_tr-W2*S_tr_gt'*H'*X_tr+lambda1*W1);
    %W1=W1-eta*df1/norm(df1); 
    W1=SAE6(X_tr, S_tr_gt, S_tr, H, lambda1,W2,k,bata);
    %df2=2*(W2*S_tr'*S_tr-W1*X_tr'*S_tr+W1*X_tr'*X_tr*W1'*W2*S_tr_gt'*S_tr_gt-W1*X_tr'*H*S_tr_gt+lambda2*W2-lambda2*eye(dimSemantic));  
    %W2=W2-eta*df2/norm(df2); 
    W2=SAE66(X_tr, S_tr_gt, S_tr,H,lambda2,W1,bata);
    loss(i)=norm(W1*S_tr'-W2*X_tr')+bata*norm(X_tr*W2'*W1*S_tr_gt'-H)+lambda1*norm(W1)+lambda2*norm(W2);
end
plot(loss)

W1      = NormalizeFea(W1')';
W2      = NormalizeFea(W2')';
n=0;
D=W1'*W2*S_te_gt';
B=pdist2(D',X_te, 'cosine');
B=B';
for i=1:length(test_label) 
    [~, I] =sort(B(i,:),'ascend');
    testClassLabels=testClassLabels';
    Y_hit5(i,:) =testClassLabels(I(1:1));
end   
for i=1:length(test_label)
    if (test_label(i)==Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/length(test_label);
disp(['APY Mean class accuracy=' num2str(zsl_accuracy)]); 