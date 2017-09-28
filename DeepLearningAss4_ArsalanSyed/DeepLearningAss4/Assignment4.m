function [RNN,M]=Assignment4()
	clc
    book_data = ReadData();    
    book_chars=unique(book_data);
    K=size(book_chars,2);    
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');    
    
    %Set up maps
    for i=1:K
        char=book_chars(:,i);
        char_to_ind(char)=i;        
        ind_to_char(i)=char;
    end
    
    %Hyperparameters
    m=100; 
    eta=0.04; %0.04 was pretty good Try 0.003 to 0.040, else 0.1
    seq_length=25;
    sig=0.01;   
          
    %Bias vectors
    RNN.b=zeros(m,1);
    RNN.c=zeros(K,1);
    
    %Weight matrices
    RNN.U = randn(m, K)*sig;
    RNN.W = randn(m, m)*sig;
    RNN.V = randn(K, m)*sig;
    
    %For testing gradients
    h0=zeros(m,1);
    X_chars = book_data(1:seq_length);
    Y_chars = book_data(2:seq_length+1);
    X=SeqToOneHot(X_chars,K,char_to_ind);
    Y=SeqToOneHot(Y_chars,K,char_to_ind);
    [P,H,A] = ForwardPass(RNN,X,h0);
    grads = ComputeGradients(RNN,X,Y,m,P,H,A);
    num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);   
    norm(grads.W-num_grads.W)/max(norm(grads.W),norm(num_grads.W))
    norm(grads.U-num_grads.U)/max(norm(grads.U),norm(num_grads.U))
    norm(grads.V-num_grads.V)/max(norm(grads.V),norm(num_grads.V))
    norm(grads.b-num_grads.b)/max(norm(grads.b),norm(num_grads.b))
    norm(grads.c-num_grads.c)/max(norm(grads.c),norm(num_grads.c))

    %Training
    [RNN,M] = Train(book_data,RNN,K,m,char_to_ind,eta,seq_length,ind_to_char);
    
    %Generate 1000 char passage
    e=50;
    X_chars = book_data(e:e+seq_length-1);
    X=SeqToOneHot(X_chars,K,char_to_ind);
    [Y] = Synthesize(RNN,zeros(m,1),X(:,1),1000); %char_to_ind('.') OneHot(char_to_ind(X(:,1)),K)
    text=FormSequence(Y,ind_to_char)
end

function Save(cost,text)    
    fileID = fopen('LatestText.txt','a+');
    fprintf(fileID,'%f \n',cost);
    fprintf(fileID,text);
    fprintf(fileID,'\n');
    fclose(fileID);
end

function [RNN,M] = Train(book_data,RNN,K,m,char_to_ind,eta,seq_length,ind_to_char)    
    M=cell(1,5);
    for i=1:5
        M{i}=0;
    end
    
    smooth_losses=zeros(1,700);
    
    epoch=10;
    totalEpochs=1;
    iter=1;
    e=1;

    while(epoch<=totalEpochs)
        X_chars = book_data(e:e+seq_length-1);
        Y_chars = book_data(e+1:e+seq_length);
        X=SeqToOneHot(X_chars,K,char_to_ind);
        Y=SeqToOneHot(Y_chars,K,char_to_ind);
        
        if e==1
            hprev=zeros(m,1);
        end        

        [P,H,A] = ForwardPass(RNN,X,hprev);

        L=ComputeLossPY(P,Y);
        if iter==1
            smooth_loss=L;
        end
        smooth_loss =.999*smooth_loss+.001*L;
       
        smooth_losses(:,iter)=smooth_loss;    
        hprev=H(:,end);

        grads = ComputeGradients(RNN,X,Y,m,P,H,A);
        
        i=1;
        epsilon=0.0000000001;
  
        %Adagrad update
        for f = fieldnames(RNN)'
            g=grads.(f{1});
            M{i}=M{i}+g.^2;         
            RNN.(f{1}) = RNN.(f{1}) - eta*g./sqrt(M{i}+epsilon);
            i=i+1;
        end
        
        %Every 500 steps, synthesize
        if mod(iter,500)==0
            [Y] = Synthesize(RNN,zeros(m,1),X(:,1),200);
            text=FormSequence(Y,ind_to_char)
            Save(smooth_loss,text);
        end

        e=e+seq_length;
        iter=iter+1;
        if(e>length(book_data)-seq_length-1)
           epoch=epoch+1; 
           e=1;
           save('Optimal','RNN','M');
        end
    end   
    plot(smooth_losses)
end

%Converts a sequence to a one hot matrix
function X = SeqToOneHot(seq,K,char_to_ind)
    seqLen = size(seq,2);
    X=zeros(K,seqLen);
    for i=1:seqLen
        char=seq(:,i);
        index=char_to_ind(char);
        X(:,i)=OneHot(index,K);
    end
end

function [book_data] = ReadData()
    book_fname = 'data/goblet_book.txt';
    fid = fopen(book_fname,'r');
    book_data = fscanf(fid,'%c');
    fclose(fid);
end

%Converts an index to onehot
function xVec = OneHot(idx,K)
    xVec=zeros(K,1);
    xVec(idx)=1;
end

%OneHot vector to character
function char = ToChar(onehot,ind_to_char)
    char=ind_to_char(find(onehot));    
end

%Given a matrix Y, forms char seq
function seq = FormSequence(Y,ind_to_char)
    seq='';    
    for i=1:size(Y,2)
       y=Y(:,i);
       seq=[seq ToChar(y,ind_to_char)];
    end
end

%Generates n characters, not used for training, only uses one x
function [Y] = Synthesize(RNN,h0,x0,n)
    x=x0;
    h=h0;
    K=size(RNN.c,1);
    Y=zeros(K,n);

    for t=1:n   
        %Generate probabilities
        a=RNN.W*h+RNN.U*x+RNN.b;
        h=tanh(a);
        o=RNN.V*h+RNN.c;
        p=softmax(o);
       
        %Choose a character
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        
        %Add character to Y
        x=OneHot(ii,K);
        Y(:,t)=x;        
    end
end

function L = ComputeLoss(X,Y,RNN,h0)
    P=ForwardPass(RNN,X,h0);
    n=size(X,2);
    sum=0;
    for i=1:n
       p=P(:,i);
       y=Y(:,i);
       sum=sum+log(y'*p);
    end
    L=-sum;
end

function L = ComputeLossPY(P,Y)
%     sum=0;
%     for i=1:n
%        p=P(:,i);
%        y=Y(:,i);
%        sum=sum+log(y'*p);
%     end
%    
%     L=-sum
    L=-trace(log(Y'*P));
end

function [P,H,A] = ForwardPass(RNN,X,h0)
    h=h0;
    m=size(h,1);
    n=size(X,2);
    K=size(RNN.c,1);

    H=zeros(m,n); 
    A=zeros(m,n);
    O=zeros(K,n);
    for t=1:n 
        x=X(:,t);
        a=RNN.W*h+RNN.U*x+RNN.b;
        h=tanh(a);
        o=RNN.V*h+RNN.c;
        H(:,t)=h;
        A(:,t)=a;
        O(:,t)=o;
    end    
    applySoftmax=@(s)(softmax(s));
    P=applySoftmax(O);
end

function grads = ComputeGradients(RNN,X,Y,m,P,H,A)

    n=size(X,2);
    grads.V=zeros(size(RNN.V));
    grads.W=zeros(size(RNN.W));
    grads.U=zeros(size(RNN.U));
    
    DO=-(Y-P)';
     
    %Calculate Dc
    grads.c=(sum(DO,1))';
    
    %Calculate gradients for last h,a
    do_tau=DO(end,:); %Get last row
    a_tau=A(:,end);    
    
    dh_tau=do_tau*RNN.V;    
    da_tau=dh_tau*diag(1-(tanh(a_tau).^2));
    
    da=da_tau;
    
    DA=zeros(n,m);
    DA(end,:)=da_tau;
    
    %Find dh,da for all other t
    for t=(n-1):-1:1
        do=DO(t,:);
        dh=do*RNN.V+da*RNN.W;
        a=A(:,t);
        da=dh*diag(1-(tanh(a).^2));
        DA(t,:)=da;
    end
    
    %Calculate DV    
  
%     cad=grads.V+DO'*H';
    
%     for i=1:n
%         g=DO(i,:);
%         h=H(:,i);
%         grads.V=grads.V+g'*h';
%     end
     
    grads.V=grads.V+DO'*H';
        
    %Calculate DW
%     for i=1:n
%         g=DA(i,:);
%         if i==1
%             h_prev=zeros(100,1);
%         else
%             h_prev=H(:,(i-1));
%         end
%         grads.W=grads.W+g'*h_prev';
%     end
    
    
   H(:,end)=[];
   H=[zeros(100,1) H];
   grads.W=grads.W+DA'*H';
 
% 
%     H(:,(1))=zeros(100,1);
%     grads.W=grads.W+DA'*H';
    
  
%     %Calculate DU
%     for i=1:n
%         g=DA(i,:);
%         x=X(:,i);
%         grads.U=grads.U+g'*x';
%     end
    grads.U=grads.U+DA'*X';
    
    %Calculate Db
    grads.b=(sum(DA,1))';

    %Clip gradients
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end

end

function num_grads = ComputeGradsNum(X, Y, RNN, h)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end