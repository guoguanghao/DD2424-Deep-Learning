function [grad_W,grad_b,grad_Wn,grad_bn]=Assignment3()
    addpath Datasets/cifar-10-batches-mat/;
    clc
    rng(400)
    [Train,Val,Test] = GetData(0); %Use GetFullData for all available examples
    
    m=[50]';
    d=3072;
    k=10;
    
    [W,b]=Init(m,d,k,0.001);
    
    n_batch=100;
    n_epochs=10;
    lambda=0;
    eta=0.01; 
    Run(Train,Val,Test,W,b,n_batch,eta,n_epochs,lambda)
    
end

function [W,b]=Init(m,d,K,sigma)   
    h=size(m,1)+1; %total number of layer

    W=cell(1,h);
    W{1}=randn([m(1) d])*sigma;
    for i=1:(h-2)
        W{i+1}=randn([m(i+1) m(i)])*sigma;
    end
    W{h}=randn([K m(end)])*sigma;

    b=cell(1,h);
    for i=1:(h-1)
        b{i}=randn([m(i) 1])*sigma;
    end
    b{h}=randn([K 1])*sigma;
end

function [X,Y,y] = LoadBatch(fileName) 
%LoadBatch Loads the data from a given file
%Returns X - the inputs
%        Y - one hot representation of y
%        y - labels

    A = load(fileName);
    X=(double(A.data)./255)';       
    X=X-repmat(mean(X,2),[1,size(X,2)]); %Subtract the average
    y=A.labels+uint8(ones(10000,1));
    Y = (bsxfun(@eq, y(:), 1:max(y)))';
end

function [Train,Val,Test] = GetData(full)
	%GetData Gets all the required data for training, validation and testing
		if full
			[Xtrain1,Ytrain1,ytrain1]=LoadBatch('data_batch_1.mat');
			[Xtrain2,Ytrain2,ytrain2]=LoadBatch('data_batch_2.mat');
			[Xtrain3,Ytrain3,ytrain3]=LoadBatch('data_batch_3.mat');
			[Xtrain4,Ytrain4,ytrain4]=LoadBatch('data_batch_4.mat');
			[Xtrain5,Ytrain5,ytrain5]=LoadBatch('data_batch_5.mat');

			%Combine all training data
			XtrainAll=[Xtrain1 Xtrain2 Xtrain3 Xtrain4 Xtrain5];
			YtrainAll=[Ytrain1 Ytrain2 Ytrain3 Ytrain4 Ytrain5];      
			ytrainAll=[ytrain1; ytrain2; ytrain3; ytrain4; ytrain5];

			n=size(XtrainAll,2);
			k=1000; %number of data points in validation set
			
			%Shuffle all data with the same permutation
			XtrainShuffled = reshape(XtrainAll,3072,size(XtrainAll,2));
			YtrainShuffled = reshape(YtrainAll,10,size(YtrainAll,2));
			ytrainShuffled = reshape(ytrainAll,size(ytrainAll,1),1);
			
			Xtrain = XtrainShuffled(1:3072,1:(n-k));
			Ytrain = YtrainShuffled(1:10,1:(n-k));
			yTrain = ytrainShuffled(1:(n-k),1);

			%Let the last k entries be for the validation set
			Xval = XtrainShuffled(1:3072,(n-k+1):n);
			Yval = YtrainShuffled(1:10,(n-k+1):n);
			yVal = ytrainShuffled((n-k+1):n,1);
		else 
			[Xtrain,Ytrain,yTrain]=LoadBatch('data_batch_1.mat'); 
			[Xval,Yval,yVal]=LoadBatch('data_batch_2.mat');
		end
		[Xtest,Ytest,ytest]=LoadBatch('test_batch.mat');
		Train={Xtrain,Ytrain,yTrain};
		Val={Xval,Yval,yVal};
		Test={Xtest,Ytest,ytest};
end

function [acc] = Run(Train,Val,Test,W,b,n_batch,eta,n_epochs,lambda)
    [Wstar,bstar]=MiniBatchGD(Train,Val,W,b,n_batch,eta,n_epochs,lambda);
    [acc] = ComputeAccuracyTest(Test,Wstar,bstar);
end

function [acc] = ComputeAccuracyTest(Test,W,b)
    X=Test{1};
    y=Test{3};
    n=size(X,2);
    numCorrect=0;
    P=EvaluateClassifier(X,W,b);
    for i=1:n
        p=P(:,i);
        correctLabel=y(i,:);
        [~,index]=max(p);
        if correctLabel==index
           numCorrect=numCorrect+1;
        end
    end
    acc=numCorrect/n;
end

function [acc] = ComputeAccuracy(Val,W,b)
    X=Val{1};
    y=Val{3};
    n=size(X,2);
    numCorrect=0;
    P=EvaluateClassifier(X,W,b);
    for i=1:n
        p=P(:,i);
        correctLabel=y(i,:);
        [~,index]=max(p);
        if correctLabel==index
           numCorrect=numCorrect+1;
        end
    end
    acc=numCorrect/n;
end

%Finds value for forward pass
function [J] = ComputeCost(X,Y,W,b,lambda)
    n=size(X,2);
    P=EvaluateClassifier(X,W,b);
    loss=0;
    for i=1:n
        y=Y(:,i);
        p=P(:,i);
        loss=loss-log(y'*p);
    end
    reg=lambda*(sumsqr(W{1})+sumsqr(W{2}));
    J=1/n*loss+reg;
end

%Given a vector x, return a vector y such that
%y(i)=1 if x(i)>0 and y(i)=0 otherwise
function y = Ind(x)
    y=x;
    idx=sign(x);
    y(idx>0)=1;
    y(idx<=0)=0;
end

%Finds P, needed for classification
function [P,S,Mu,V] = EvaluateClassifier(X,W,b)
    k=size(W,2);
    n=size(X,2); %num of data examples
    
    S=cell(1,k); %Has K elements, each is an mxn matrix
    Mu=cell(1,k-1);    
    V=cell(1,k-1);
    
       
    for layer=1:(k-1)
        S{layer}=W{layer}*X+repmat(b{layer},[1 n]);
        Mu{layer}=sum(S{layer},2)/n;  
        V{layer}=var(S{layer},0,2);
        V{layer}=V{layer}*(n-1)/n;     
%         S{layer}=BatchNormalize(S{layer},Mu{layer},V{layer});
        X=max(0,S{layer});
    end
    S{k}=W{k}*X+repmat(b{k},[1 n]);

    applySoftmax=@(s)(softmax(s));
    P=applySoftmax(S{end});
end

function Sbn=BatchNormalize(S,mu,v)
    n=size(S,2);
    e=0.00000000001;
    Sbn=(diag(v+e))^(-1/2)*(S-repmat(mu,[1 n]));
end

function [grad_W,grad_b]=InitGrads(W,b,k)
    %Initialise gradient matrices, 1 for each layer
    grad_W=cell(1,k);
    grad_b=cell(1,k);
    
    for i=1:k
        grad_W{i}=zeros(size(W{i}));
        grad_b{i}=zeros(size(b{i}));
    end
end

function[G]=InitG(P,Y,n)
    G=zeros(n,10); 
    for i=1:n
        p=P(:,i);
        yt=Y(:,i)';
        G(i,:)=-yt/(yt*p)*(diag(p)-p*p');
    end
end

%Check if correct
function [grad_W,grad_b] = ComputeGradients(X,Y,W,b,lambda)
    [P,S,Mu,V] = EvaluateClassifier(X,W,b);
    n=size(X,2);
    k=size(W,2);
    
    [grad_W,grad_b]=InitGrads(W,b,k);    
    G=InitG(P,Y,n);
    
    grad_b{end}=sum(G,1)/n;
    grad_W{end}=CalcGradW(W,G,S,k,n,lambda,X);
    
    G=G*W{end};
    G=G.*(Ind(max(0,S{k-1})))';
    
    for layer=(k-1):-1:1    
       G=BatchNormBackPass(G,S{layer},Mu{layer},V{layer});
        
        grad_b{layer}=sum(G,1)/n;   
        grad_W{layer}=CalcGradW(W,G,S,layer,n,lambda,X);

        if layer>1
            G=G*W{layer};
            G=G.*(Ind(max(0,S{layer-1})))';
        end
    end

end

function Gbn=BatchNormBackPass(G,S,mu,v)
    n=size(S,2);
    
    DjDv=FindDjDv(G,S,mu,v,n);
    DjDmu=FindDjDmu(G,v,n);
    e=0.00000000001;
    Gbn=zeros(size(G));
    
    Vb=(diag(v+e))^(-1/2);
    for i=1:n       
        g=G(i,:);
        s=S(:,i);        
        term1=g*Vb;
        term2=2/n*DjDv*diag(s-mu);
        term3=DjDmu/n;
        Gbn(i,:)=term1+term2+term3;
    end
end

function DjDv = FindDjDv(G,S,mu,v,n) 
    e=0.00000000001;
    Vb=(diag(v+e))^(-3/2);
    DjDv=0;
    for i=1:n
        g=G(i,:);
        s=S(:,i);
        DjDv=DjDv+g*Vb*diag(s-mu);
    end
    DjDv=DjDv*(-1/2);
end

function DjDmu = FindDjDmu(G,v,n)
    e=0.000000001;
    Vb=(diag(v+e))^(-1/2);
    DjDmu=0;
    for i=1:n
        g=G(i,:);
        DjDmu=DjDmu+g*Vb;
    end
    DjDmu=DjDmu*-1;
end

function [grad_W]=CalcGradW(W,G,S,layer,n,lambda,X)
    if layer>1
        X=max(0,S{layer-1});
    end 
    grad_W=(G'*X')/n+2*lambda*W{layer};
end

function [Wstar,bstar]=MiniBatchGD(Train,Val,W,b,n_batch,eta,n_epochs,lambda)
    Wstar=W;
    bstar=b;
    
    X=Train{1};
    Y=Train{2};
    
    
    totLayers=size(W,2);       
    n=size(X,2);
    decay_rate=0.95;    
    rho=0.95;
    
    vW=cell(1,totLayers);
    vb=cell(1,totLayers);
    
    for i=1:totLayers
        vW{i}=zeros(size(W{i}));
        vb{i}=zeros(size(b{i}));
    end
    
    
    acc=zeros(1,n_epochs);
    accVal=zeros(1,n_epochs);
    costs=zeros(1,n_epochs);
    
    for e=1:n_epochs
        for j=1:n/n_batch        
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);    

            [grad_W,grad_b]=ComputeGradients(Xbatch,Ybatch,Wstar,bstar,lambda);    
            
            for i=1:totLayers
                Wstar{i}=Wstar{i}-eta*grad_W{i};
                bstar{i}=bstar{i}-eta*grad_b{i}';
                
                vW{i}=rho*vW{i}+eta*grad_W{i};
                vb{i}=rho*vb{i}+eta*grad_b{i}';
                
                Wstar{i}=Wstar{i}-vW{i};
                bstar{i}=bstar{i}-vb{i};
            end
        end

        costs(:,e)=ComputeCost(X,Y,Wstar,bstar,lambda);        
        acc(:,e)=ComputeAccuracy(Train,Wstar,bstar);
        accVal(:,e)=ComputeAccuracy(Val,Wstar,bstar);
        eta=eta*decay_rate;
    end 
    figure()
    plot(costs)
    title('Cost per epoch');
    figure()
    plot(acc);
    hold on
    plot(accVal);
    title('Accuracy on training/validation data per epoch');
end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

c = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)

    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)   
    grad_W{j} = zeros(size(W{j}));
    for i=1:numel(W{j}) 
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})

            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);

            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);

            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})

            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);

            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);

            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end