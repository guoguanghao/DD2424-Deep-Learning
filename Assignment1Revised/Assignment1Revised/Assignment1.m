function Assignment1()
    clc
    addpath Datasets/cifar-10-batches-mat/;
    [Train,Val,Test] = GetData();
    [K,~]=size(Train.Y);
    [d,~]=size(Train.X);
    sigma=0.01;
    [W,b]=Init(K,d,sigma);

    Params.eta=0.01;
    Params.lambda=1;
    Params.n_batch=100;
    Params.n_epochs=40;
    Params.experiment=4;
%     [norm_W,norm_b]=TestGradients(Train,W,b,Params);
    Run(Train,Val,Test,Params,W,b)
end

function [norm_W,norm_b]=TestGradients(Train,W,b,Params)
    [grad_W,grad_b] = ComputeGradients(Train.X(:,1),Train.Y(:,1),W,b,Params);
    [ngrad_b, ngrad_W] = ComputeGradsNum(Train.X(:,1),Train.Y(:,1), W, b,Params.lambda,1e-5);
    norm_W=norm(grad_W-ngrad_W)/max(norm(grad_W),norm(ngrad_W));
    norm_b=norm(grad_b-ngrad_b)/max(norm(grad_b),norm(ngrad_b));
end

function [X,Y,y] = LoadBatch(fileName)    
    A = load(fileName);
    X=(double(A.data)./255)';
    y=A.labels+uint8(ones(10000,1));
    Y = (bsxfun(@eq, y(:), 1:max(y)))';
end

function [Train,Val,Test] = GetData()
    [Train.X,Train.Y,Train.y]=LoadBatch('data_batch_1.mat'); 
    [Val.X,Val.Y,Val.y]=LoadBatch('data_batch_2.mat');
    [Test.X,Test.Y,Test.y]=LoadBatch('test_batch.mat');
end

function [acc] = Run(Train,Val,Test,Params,W,b)
    [Wstar,bstar] = MiniBatchGD(Train,Val,Test,Params,W,b);
    acc=ComputeAccuracy(Test,Wstar,bstar);
    VisualiseWeights(Wstar);
end

function VisualiseWeights(W)
    for i=1:10
       im=reshape(W(i,:),32,32,3);
       s_im{i}=(im-min(im(:)))/(max(im(:))-min(im(:)));
       s_im{i}=permute(s_im{i},[2 1 3]);   
    end    
%     figure()
%     montage(s_im);
%     title('Visual representation of trained weights)')
end

function Visualize(A)
    I=reshape(A.data',32,32,3,10000);
    I=permute(I,[2,1,3,4]);
    montage(I(:,:,:,1:500),'Size',[5,5]);
end

function [W,b]=Init(K,d,sigma)
    W=randn([K d])*sigma;
    b=randn([K 1])*sigma;
end

function [Wstar,bstar] = MiniBatchGD(Train,Val,Test,Params,W,b)
    Wstar=W;
    bstar=b;
    n=size(Train.X,2);
    
%     accTrain=zeros(1,Params.n_epochs);
    accVal=zeros(1,Params.n_epochs);
    accTest=zeros(1,Params.n_epochs);
    
%     costTrain=zeros(1,Params.n_epochs);
    costVal=zeros(1,Params.n_epochs);
    costTest=zeros(1,Params.n_epochs);
    
    
    
    
    for i=1:Params.n_epochs
        i
        for j=1:n/Params.n_batch            
            j_start = (j-1)*Params.n_batch + 1;
            j_end = j*Params.n_batch;
            Xbatch = Train.X(:, j_start:j_end);
            Ybatch = Train.Y(:, j_start:j_end);    
            
            [grad_W,grad_b] = ComputeGradients(Xbatch,Ybatch,Wstar,bstar,Params);
            Wstar=Wstar-Params.eta*grad_W;
            bstar=bstar-Params.eta*grad_b;
        end
        
%         costTrain(1,i)=ComputeCost(Xbatch,Ybatch,Wstar,bstar,Params.lambda);
        costVal(1,i)=ComputeCost(Val.X,Val.Y,Wstar,bstar,Params.lambda);
        costTest(1,i)=ComputeCost(Test.X,Test.Y,Wstar,bstar,Params.lambda);
        
%         accTrain(1,i)=ComputeAccuracy(Train,Wstar,bstar);
        accVal(1,i)=ComputeAccuracy(Val,Wstar,bstar);
        accTest(1,i)=ComputeAccuracy(Test,Wstar,bstar);        
    end
    
    figure()
    subplot(1,2,1)
    plot(costVal)
    hold on
    plot(costTest)
    xlabel('Epoch')
    ylabel('Cost')
    legend('Validation','Test')  
    
    
    str=sprintf('Experiment %i: Loss After Every Epoch',Params.experiment);
    
    title(str)
    

    subplot(1,2,2)
    plot(accVal)
    hold on
    plot(accTest)
    xlabel('Epoch')
    ylabel('Accuracy')
    str=sprintf('Experiment %i: Accuracy After Every Epoch',Params.experiment);
    title(str)
    legend('Validation','Test')

end

function [P] = EvaluateClassifier(X,W,b)
    n=size(X,2);
    S=W*X+repmat(b,[1 n]);
    applySoftmax=@(s)(softmax(s));
    P=applySoftmax(S);
end

function [grad_W,grad_b] = ComputeGradients(X,Y,W,b,Params)
    n=size(X,2);
    grad_W=zeros(size(W));
    grad_b=zeros(size(b));
    
    P=EvaluateClassifier(X,W,b);
    G=-(Y-P)';
    grad_b=(grad_b+sum(G,1)')/n;
    grad_W=(grad_W+G'*X')/n+2*Params.lambda*W;
end

function [J] = ComputeCost(X,Y,W,b,lambda)
    n=size(X,2);
%     P=EvaluateClassifier(X,W,b);
%     J=1/n*(-log(Y'*P))+lambda*sumsqr(W);
%     
%     
    P=EvaluateClassifier(X,W,b);
    loss=0;
    for i=1:n
        y=Y(:,i);
        p=P(:,i);
        loss=loss-log(y'*p);
    end
    reg=lambda*sumsqr(W);
    J=1/n*loss+reg;    
end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    c = ComputeCost(X, Y, W, b, lambda);

    for i=1:length(b)

        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end

    for i=1:numel(W)
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        grad_W(i) = (c2-c) / h;
    end
end

function [acc] = ComputeAccuracy(Test,W,b)
    n=size(Test.X,2);
    numCorrect=0;
    P=EvaluateClassifier(Test.X,W,b);
    for i=1:n
        p=P(:,i);
        correctLabel=Test.y(i,:);
        [~,index]=max(p);
        if correctLabel==index
           numCorrect=numCorrect+1;
        end
    end
    acc=numCorrect/n;
end