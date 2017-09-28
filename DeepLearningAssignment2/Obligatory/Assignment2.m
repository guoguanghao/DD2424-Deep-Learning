function Assignment2()
    addpath Datasets/cifar-10-batches-mat/;
    clc
    rng(400)
    [Train,Val,Test] = GetData(); %Use GetFullData for all available examples
    m=50;
    d=3072;
    k=10;
    
    [W,b]=Init(m,d,k);

    n_batch=100;
    n_epochs=10;
    lambda=0;
    eta=0.01;

    acc=Run(Train,Val,Test,W,b,n_batch,eta,n_epochs,lambda)    
	Search(Train,Val,Test,W,b)
end

%For coarse/fine search
function Search(Train,Val,Test,W,b)  
    n_epochs=3;
    for i=1:20
        min = 0.0001;
        max = 0.001;
        lambda = (max-min).*rand(1,1) + min;
        
        etaMin=0.02;
        etaMax=0.06;
        eta = etaMin + (etaMax - etaMin)*rand(1, 1);
        
        acc=Run(Train,Val,Test,W,b,n_batch,eta,n_epochs,lambda);
        Save(eta,lambda,acc);
    end
end

function Save(eta,lambda,acc)    
    fileID = fopen('Best.txt','a+');
    fprintf(fileID,'%f %f %f \n',eta,lambda,acc);
    fclose(fileID);
end

function TestGradients()
    [Train,~,~,] = GetData();    
    m=10;
    d=3072; 
    k=10; 
    [W,b]=Init(m,d,k);   
    lambda=0;
    X=Train{1}(:,1);
    Y=Train{2}(:,1);
    [grad_W,grad_b] = ComputeGradients(X,Y,W,b,lambda);
    [grad_bN, grad_WN] = ComputeGradsNum(X, Y, W, b, lambda, 1e-5);
    
    norm(grad_WN{1}-grad_W{1})/max(norm(grad_W{1}),norm(grad_WN{1}))
    norm(grad_bN{1}-grad_b{1})/max(norm(grad_b{1}),norm(grad_bN{1}))  
    
    norm(grad_WN{2}-grad_W{2})/max(norm(grad_W{2}),norm(grad_WN{2}))
    norm(grad_bN{2}-grad_b{2})/max(norm(grad_b{2}),norm(grad_bN{2}))    
end

function [W,b]=Init(m,d,K)
%Init Initializes the network paramters
%Returns Cell arrays W,b which contain 2
%weight matrices and 2 bias vectors respectively

    sigma=0.001;
    W1=randn([m d])*sigma;
    b1=zeros(m,1);    
    W2=randn([K m])*sigma;
    b2=zeros(K, 1);
    
    W={W1,W2};
    b={b1,b2};
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

function [Train,Val,Test] = GetData()
%GetData Gets all the required data for training, validation and testing
    [Xtrain,Ytrain,yTrain]=LoadBatch('data_batch_1.mat'); 
    [Xval,Yval,yVal]=LoadBatch('data_batch_2.mat');
    [Xtest,~,ytest]=LoadBatch('test_batch.mat');
    
    Train={Xtrain,Ytrain,yTrain};
    Val={Xval,Yval,yVal};
    Test={Xtest,ytest};
end

function [Train,Val,Test] = GetFullData()
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

    [Xtest,~,ytest]=LoadBatch('test_batch.mat');
    
    Train={Xtrain,Ytrain,yTrain};
    Val={Xval,Yval,yVal};
    Test={Xtest,ytest};
end



function [acc] = Run(Train,Val,Test,W,b,n_batch,eta,n_epochs,lambda)
    [Wstar,bstar]=MiniBatchGD(Train,Val,W,b,n_batch,eta,n_epochs,lambda);
    [acc] = ComputeAccuracyTest(Test,Wstar,bstar);
end

function [acc] = ComputeAccuracyTest(Test,W,b)
    X=Test{1};
    y=Test{2};
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
function [P] = EvaluateClassifier(X,W,b)
    n=size(X,2);
    S1=W{1}*X+repmat(b{1},[1 n]);        
    h=max(0,S1); 
    S=W{2}*h+repmat(b{2},[1 n]);    
    applySoftmax=@(s)(softmax(s));
    P=applySoftmax(S);
end

%Check if correct
function [grad_W,grad_b] = ComputeGradients(X,Y,W,b,lambda)
    P = EvaluateClassifier(X,W,b);
    [~,n]=size(X);

    grad_W{1}=zeros(size(W{1}));
    grad_b{1}=zeros(size(b{1}));

    grad_W{2}=zeros(size(W{2}));
    grad_b{2}=zeros(size(b{2}));

    S1=W{1}*X+repmat(b{1},[1 n]);
    H=max(S1,0);
    for i = 1:n
        p=P(:,i);
        x=X(:,i);
        yt=Y(:,i)';

        h=H(:,i);
        
        g=-yt/(yt*p)*(diag(p)-p*p');

        grad_b{2}=grad_b{2}+g';
        grad_W{2}=grad_W{2}+g'*h';
        
        g=g*W{2};
        g=g.*(Ind(h))'; %faster than doing g=g*diag(Ind(h));

        grad_b{1}=grad_b{1}+g';   
        grad_W{1}=grad_W{1}+g'*x'; 
    end

    grad_W{1}=grad_W{1}/n+2*lambda*W{1};
    grad_b{1}=grad_b{1}/n;  
    
    grad_W{2}=grad_W{2}/n+2*lambda*W{2};
    grad_b{2}=grad_b{2}/n;  
end


function [Wstar,bstar]=MiniBatchGD(Train,Val,W,b,n_batch,eta,n_epochs,lambda)
    Wstar=W;
    bstar=b;
    
    X=Train{1};
    Y=Train{2};
    
    n=size(X,2);
    decay_rate=0.95;
    
    rho=0.95;
   
    vW1=zeros(size(W{1}));
    vb1=zeros(size(b{1}));
    vW2=zeros(size(W{2}));
    vb2=zeros(size(b{2}));
    
    acc=zeros(1,n_epochs);
    accVal=zeros(1,n_epochs);
    costs=zeros(1,n_epochs);
    
    for e=1:n_epochs
        e
        for j=1:n/n_batch        
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);    

            [grad_W,grad_b]=ComputeGradients(Xbatch,Ybatch,Wstar,bstar,lambda);
  
            %Update
            Wstar{1}=Wstar{1}-eta*grad_W{1};
            Wstar{2}=Wstar{2}-eta*grad_W{2};
            bstar{1}=bstar{1}-eta*grad_b{1};     
            bstar{2}=bstar{2}-eta*grad_b{2};  
            
            vW1=rho*vW1+eta*grad_W{1};
            vb1=rho*vb1+eta*grad_b{1};
            vW2=rho*vW2+eta*grad_W{2};
            vb2=rho*vb2+eta*grad_b{2};
            
            Wstar{1}=Wstar{1}-vW1;
            bstar{1}=bstar{1}-vb1;
            Wstar{2}=Wstar{2}-vW2;
            bstar{2}=bstar{2}-vb2;
        end

        costs(:,e)=ComputeCost(X,Y,Wstar,bstar,lambda);        
        acc(:,e)=ComputeAccuracy(Train,Wstar,bstar);
        accVal(:,e)=ComputeAccuracy(Val,Wstar,bstar);
        eta=eta*decay_rate;
        
         if e>1
            if abs(costs(:,e)-costs(:,e-1))<1e-6
                disp('Final epoch')
                e
                disp('-----------')
                break
            end
        end
    end 
    figure()
    plot(costs)
    title('Cost per epoch');
    figure()
    plot(acc);
    hold on
    plot(accVal);
    title('Accuracy on training/validation data per epoch');
    costs(end)
    acc(end)
    accVal(end)
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