% Assignment 1 Deep Learning
% Arsalan Syed 961117-6331
% Course DD2424
% 8th April 2017

%% Main Function
function [] = Assignment1()
    addpath Datasets/cifar-10-batches-mat/;
    clear
    rng(400)
    setGlobalx(0) %set to 1 to use average preprocessing 
    setGlobalSVM(0) %set to 1 to use SVM gradient 
    
    acc1=LearnDataset(0,40,100,0.1,1,0,0,1,0)
    acc2=LearnDataset(0,40,100,0.01,1,0,0,2,0)
    acc3=LearnDataset(0.1,40,100,0.01,1,0,0,3,0)
    acc4=LearnDataset(1,40,100,0.1,1,0,0,4,0)
    
    %Takes a lot of time to complete
    acc5=LearnDataset(0,40,50,0.1,0.5,1,0.1,5,0.01)
end

%% Used for global variables
function setGlobalx(val)
    global x
    x = val;
end

function r = getGlobalx()
    global x
    r = x;
end

function setGlobalSVM(val)
    global svm
    svm = val;
end

function r = getGlobalSVM()
    global svm
    r = svm;
end

%% Helper function
function [K,d,n] = GetSizes(X,W)
    [d,n]=size(X);
    [K,~]=size(W);
end

%% Used for handling data
function [X,Y,y] = LoadBatch(fileName)    
    A = load(fileName);
    X=(double(A.data)./255)';
    if getGlobalx()
        X=X-repmat(mean(X,2),[1 10000]); %Subtract the average
    end
    y=A.labels+uint8(ones(10000,1));
    Y = (bsxfun(@eq, y(:), 1:max(y)))';
end

function [Xtrain,Ytrain,yTrain,Xval,Yval,yVal,Xtest,ytest] = GetFullData()
    [Xtrain1,Ytrain1,ytrain1]=LoadBatch('data_batch_1.mat');
    [Xtrain2,Ytrain2,ytrain2]=LoadBatch('data_batch_2.mat');
    [Xtrain3,Ytrain3,ytrain3]=LoadBatch('data_batch_3.mat');
    [Xtrain4,Ytrain4,ytrain4]=LoadBatch('data_batch_4.mat');
    [Xtrain5,Ytrain5,ytrain5]=LoadBatch('data_batch_5.mat');

    %Combine all training data
    XtrainAll=[Xtrain1 Xtrain2 Xtrain3 Xtrain4 Xtrain5];
    XtrainAll = [XtrainAll XtrainAll]; 

    YtrainAll=[Ytrain1 Ytrain2 Ytrain3 Ytrain4 Ytrain5];
    YtrainAll = [YtrainAll YtrainAll];
        
    ytrainAll=[ytrain1; ytrain2; ytrain3; ytrain4; ytrain5];
    ytrainAll = [ytrainAll; ytrainAll];

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
end

function [Xtrain,Ytrain,yTrain,Xval,Yval,yVal,Xtest,ytest] = GetData()
    [Xtrain,Ytrain,yTrain]=LoadBatch('data_batch_1.mat'); 
    [Xval,Yval,yVal]=LoadBatch('data_batch_2.mat');
    [Xtest,~,ytest]=LoadBatch('test_batch.mat');
end

%% Computes cost, accuracy, gradients
function [J] = ComputeCost(X,Y,W,b,lambda)
    [~,~,n]=GetSizes(X,W);
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

function [acc] = ComputeAccuracy(X,y,W,b)
    [~,~,n]=GetSizes(X,W);
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

function [grad_W,grad_b] = ComputeGradients(X,Y,P,W,lambda)
    [K,d,n]=GetSizes(X,W);
    grad_W=zeros(K,d);
    grad_b=zeros(K,1);
    
    for i = 1:n
       p=P(:,i); 
       dpds=diag(p)-p*p';
       x=X(:,i);
       yt=Y(:,i)';
       g=-yt/(yt*p)*dpds;       
       gtxt=g'*x';
       grad_b=grad_b+g';
       grad_W=grad_W+gtxt;
    end
    
    grad_W=grad_W/n+2*lambda*W;
    grad_b=grad_b/n;     
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

%Used for SVM gradient computation
function [dlds] = FindDlDs(s,yVec)
    dlds=zeros(1,10);
    y=find(yVec);
    for i=1:10
       if i==y
          dlds(i)=0;
       else 
           if s(i)-s(y)+1>0
              dlds(i)=1; 
           end
       end        
    end
end

function [grad_W,grad_b] = ComputeGradientsSVM(X,Y,W,b,lambda)
    [K,d,n]=GetSizes(X,W);
    grad_W=zeros(K,d);
    grad_b=zeros(K,1);
    
    for i = 1:n
        x=X(:,i);
        y=Y(:,i);
        dlds = FindDlDs(W*x+b,y);       
        grad_b=grad_b+dlds';
        grad_W=grad_W+dlds'*x';        
    end
    
    grad_W=grad_W/n+2*lambda*W;
    grad_b=grad_b/n;     
end

function [P] = EvaluateClassifier(X,W,b)
    [~,~,n]=GetSizes(X,W);    
    S=W*X+repmat(b,[1 n]);
    applySoftmax=@(s)(softmax(s));
    P=applySoftmax(S);
end


%% Main algorithm
function [Wstar,bstar,costTr,totCostTr,costVal,totCostVal] = MiniBatchGD(X,Y,n_batch,eta,n_epochs,W,b,lambda,decay,Xval,Yval,momentum,yTrain,yVal,noiseSigma)
    Wstar=W;
    bstar=b;

    [~,d,n]=GetSizes(X,W);

    costTr=zeros(1,n_epochs);
    totCostTr=zeros(1,n_epochs);
    costVal=zeros(1,n_epochs);
    totCostVal=zeros(1,n_epochs);  
    
    accurTr=zeros(1,n_epochs);    
    accurVal=zeros(1,n_epochs);
    
    deltaW=0;
    deltaB=0;
    for i=1:n_epochs
        for j=1:n/n_batch            
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);    
            
            Xbatch=Xbatch+randn(d,n_batch)*noiseSigma; %random noise
            
            %(Forward pass) Find probabilities
            P=EvaluateClassifier(Xbatch,Wstar,bstar);                       
          
            %(Backward pass) Find gradients
            if getGlobalSVM()
                [gw,gb]=ComputeGradientsSVM(Xbatch,Ybatch,Wstar,bstar,lambda);
            else
                [gw,gb]=ComputeGradients(Xbatch,Ybatch,P,Wstar,lambda);
            end
                  
            prevW=Wstar;
            prevB=bstar;
            
            %Update
            Wstar=Wstar-eta*gw+momentum*deltaW;
            bstar=bstar-eta*gb+momentum*deltaB;   
                        
            deltaW=Wstar-prevW;
            deltaB=bstar-prevB;
        end
        
        costTr(1,i)= ComputeCost(Xbatch,Ybatch,Wstar,bstar,lambda);
        costVal(1,i)= ComputeCost(Xval,Yval,Wstar,bstar,lambda);
        
        if i>1
            totCostTr(1,i)=totCostTr(1,i-1)+costTr(1,i);
            totCostVal(1,i)=totCostVal(1,i-1)+costVal(1,i);
        else 
            totCostTr(1,i)=costTr(1,i);
            totCostVal(1,i)=costVal(1,i);
        end        
        eta=eta*decay; 
        
        accurTr(1,i)=ComputeAccuracy(X,yTrain,Wstar,bstar);
        accurVal(1,i)=ComputeAccuracy(Xval,yVal,Wstar,bstar);
    end
end

function [acc] = LearnDataset(lambda,n_epochs,n_batch,eta,decay,useAllData,momentum,i,noiseSigma)
    if useAllData==1
        [Xtrain,Ytrain,yTrain,Xval,Yval,yVal,Xtest,ytest] = GetFullData();
    else
        [Xtrain,Ytrain,yTrain,Xval,Yval,yVal,Xtest,ytest] = GetData();
    end

    [K,~]=size(Ytrain);
    [d,~]=size(Xtrain);

    %Initialise W,b with Gaussian distribution (Mean = 0, Std = 0.01)
    sigma=0.01;
    W=randn([K d])*sigma;
    b=randn([K 1])*sigma;
    
    %Train
    [Wstar,bstar,costTr,totCostTr,costVal,totCostVal]=MiniBatchGD(Xtrain,Ytrain,n_batch,eta,n_epochs,W,b,lambda,decay,Xval,Yval,momentum,yTrain,yVal,noiseSigma);
    
    %Test
    acc=ComputeAccuracy(Xtest,ytest,Wstar,bstar);
           
    %Plot
    PlotFigures(costTr,totCostTr,costVal,totCostVal,i);
    
    %Visualize weights
    VisualiseWeights(Wstar)
end

%% Draws graphs
function [] = PlotFigures(costTr,totCostTr,costVal,totCostVal,i)   
    str1=sprintf('Experiment %i: Loss After Every Epoch (Training)',i);
    str2=sprintf('Experiment %i: Total Loss After Every Epoch (Training)',i);
    str3=sprintf('Experiment %i: Loss After Every Epoch (Validation)',i);
    str4=sprintf('Experiment %i: Total Loss After Every Epoch (Validation)',i);
       
    subplot(2,2,1)    
    plot(costTr)
    title(str1)
    xlabel('Epoch') % x-axis label
    ylabel('Loss') % y-axis label
    
    subplot(2,2,2)   
    plot(totCostTr)
    title(str2)
    xlabel('Epoch') % x-axis label
    ylabel('Total Loss') % y-axis label
    
    subplot(2,2,3)   
    plot(costVal)
    title(str3)
    xlabel('Epoch') % x-axis label
    ylabel('Loss') % y-axis label
    
    subplot(2,2,4)   
    plot(totCostVal)
    title(str4)
    xlabel('Epoch') % x-axis label
    ylabel('Total Loss') % y-axis label
end

function VisualiseWeights(W)
    figure()
    for i=1:10
       im=reshape(W(i,:),32,32,3);
       s_im{i}=(im-min(im(:)))/(max(im(:))-min(im(:)));
       s_im{i}=permute(s_im{i},[2 1 3]);             
    end    
    montage(s_im);
    title('Visual representation of trained weights)')
end

function TestGradientCompute()
    [X,Y,~]=LoadBatch('data_batch_1.mat');
    [K,~]=size(Y);
    [d,~]=size(X);
    lambda=0;
    sigma=0.01;
    W=randn([K d])*sigma;
    b=randn([K 1])*sigma;
    P=EvaluateClassifier(X,W,b);
    [grad_W,grad_b] = ComputeGradients(X(:,1),Y(:,1),P(:,1),W,lambda);
    [ngrad_b,ngrad_W] =ComputeGradsNum(X(:,1), Y(:,1), W, b, lambda, 1e-6);
    relativeErrorW=norm(grad_W-ngrad_W)/max(norm(grad_W),norm(ngrad_W))
    relativeErrorB=norm(grad_b-ngrad_b)/max(norm(grad_b),norm(ngrad_b))
end

%% Montage functions
function state = montage(I, varargin)
%MONTAGE  Display multiple images as a montage of subplots
%
% Examples:
%   montage
%   montage(I)
%   montage(I, map)
%   montage(..., param1, value1, param2, value2, ...)
%
% This function displays multiple images, in a stack or cell array or
% defined by filenames in a cell array or simply in the current directory,
% in a grid of subplots in the current figure.
%
% The size of grid is calculated or user defined. Images not fitting in the
% grid can be scrolled through using the scroll keys. This allows fast
% scrolling through a movie or image stack, e.g.
%    montage(imstack, 'Size', 1)
% creates a single frame montage with images scrolled through using arrow
% keys.
%
% This function is designed to replace the MONTAGE function provided with
% the Image Processing Toolbox (IPT). It is syntax compatible, but operates
% differently, and while results are generally comparable given identical
% syntaxes, this is not guaranteed.
%
% Differences from the IPT version are:
%    - The IPT is not required!
%    - Images are placed in subplots, so can be zoomed separately.
%    - Small images are properly enlarged on screen.
%    - Gaps can be placed between images.
%    - Images can be viewed on a grid smaller than the number of images.
%    - Surplus images can be viewed by scrolling through pages.
%    - A directory of images can be viewed easily.
%    - It cannot return an image handle (as there are multiple images)
%
% Keys:
%    Up - Back a row.
%    Down - Forward a row.
%    Left - Back a page (or column if there is only one row).
%    Right - Forward a page (or column if there is only one row).
%    Shift - 2 x speed.
%    Ctrl - 4 x speed.
%    Shift + Ctrl - 8 x speed.
%
% IN:
%   I - MxNxCxP array of images, or 1xP cell array. C is 1 for indexed
%       images or 3 for RGB images. P is the number of images. If I is a
%       cell array then each cell must contain an image or image filename.
%       If I is empty then all the images in the current directory are
%       used. Default: [].
%   map - Kx3 colormap to be used with indexed images. Default: gray(256).
%   Optional parameters - name, value parameter pairs for the following:
%      'Size' - [H W] size of grid to display image on. If only H is given
%               then W = H. If either H or W is NaN then the number of rows
%               or columns is chosen such that all images fit. If both H
%               and W are NaN or the array is empty then the size of grid
%               is chosen to fit all images in as large as possible.
%               Default: [].
%      'Indices' - 1xL list of indices of images to display. Default: 1:P.
%      'Border' - [B R] borders to give each image top and bottom (B) and
%                 left and right (R), to space out images. Borders are
%                 normalized to the subplot size, i.e. B = 0.01 gives a border
%                 1% of the height of each subplot. If only B is given, R =
%                 B. Default: 0.01.
%      'DisplayRange' - [LOW HIGH] display range for indexed images.
%                       Default: [min(I(:)) max(I(:))].
%      'Map' - Kx3 colormap or (additionally from above) name of MATLAB
%              colormap, for use with indexed images. Default: gray(256).

% $Id: montage.m,v 1.7 2009/02/25 16:39:01 ojw Exp $

% Parse inputs
[map layout gap indices lims] = parse_inputs(varargin);

if nargin == 0 || isempty(I)
    % Read in all the images in the directory
    I = get_im_names;
    if isempty(I)
        % No images found
        return
    end
end

if ischar(I)
  disp('Have a directory name here');
  I = get_im_names2(I);
  if isempty(I)
    % No images found
    return
  end  
end

if isnumeric(I)
    [y x c n] = size(I);
    if isempty(lims)
        lims = [min(reshape(I, numel(I), 1)) max(reshape(I, numel(I), 1))];
    elseif isequal(0, lims)
        lims = default_limits(I);
    end
    if isfloat(I) && c == 3
        I = uint8(I * 256 - 0.5);
        lims = round(lims * 256 - 0.5);
    end
    I = squeeze(num2cell(I, [1 2 3]));
elseif iscell(I)
    A = I{1};
    if ischar(A)
        A = imread_rgb(A);
        I{1} = A;
    end
    n = numel(I);
    % Assume all images are the same size and type as the first
    [y x c] = size(A);
    if isempty(lims) || isequal(0, lims)
        lims = default_limits(A);
    end
else
    error('I not of recognized type.');
end

% Select indexed images
if ~isequal(indices, -1)
    I = I(indices);
    n = numel(I);
end

% Compute a good layout
layout = choose_layout(n, y, x, layout);

% Create a data structure to store the data in
num = prod(layout);
state.n = num * ceil(n / num);
state.h = zeros(layout);
I = [I(:); cell(state.n-n, 1)];

% Get and clear the figure
fig = gcf;
clf(fig);

% Set the figure size well
MonSz = get(0, 'ScreenSize');
MaxSz = MonSz(3:4) - [20 120];
ImSz = layout([2 1]) .* [x y] ./ (1 - 2 * gap([end 1]));
RescaleFactor = min(MaxSz ./ ImSz);
if RescaleFactor > 1
    % Integer scale for enlarging, but don't make too big
    MaxSz = min(MaxSz, [1000 680]);
    RescaleFactor = max(floor(min(MaxSz ./ ImSz)), 1);
end
figPosNew = ceil(ImSz * RescaleFactor);
% Don't move the figure if the size isn't changing
figPosCur = get(fig, 'Position');
if ~isequal(figPosCur(3:4), figPosNew)
    % Keep the centre of the figure stationary
    figPosNew = [max(1, floor(figPosCur(1:2)+(figPosCur(3:4)-figPosNew)/2)) figPosNew];
    % Ensure the figure bar is in bounds
    figPosNew(1:2) = min(figPosNew(1:2), MonSz(1:2)+MonSz(3:4)-[6 101]-figPosNew(3:4));
    set(fig, 'Position', figPosNew);
end

% Set the colourmap
colormap(map);

% Set the first lot of images
index = mod(0:num-1, state.n) + 1;
hw = 1 ./ layout;
gap = gap ./ layout;
dims = hw - 2 * gap;
dims = dims([2 1]);
for a = 1:layout(1)
    for b = 1:layout(2)
        c = index(b + (layout(1) - a) * layout(2));
        A = I{c};
        if ischar(A)
            A = imread_rgb(A);
            I{c} = A;
        end
        subplot('Position', [(b-1)*hw(2)+gap(2) (a-1)*hw(1)+gap(1) dims]);
        if isempty(A)
            state.h(a,b) = imagesc(zeros(1, 1, 3), lims);
            axis image off;
            set(state.h(a,b), 'CData', []);
        else
            state.h(a,b) = imagesc(A, lims);
            axis image off;
        end
    end
end
drawnow;
figure(fig); % Bring the montage into view

% Check if we need to be able to scroll through images
if n > num
    % Intialize rest of data structure
    state.index = 1;
    state.layout = layout;
    state.I = I;
    % Set the callback for image navigation, and save the image data in the figure
    set(fig, 'KeyPressFcn', @keypress_callback, 'Interruptible', 'off', 'UserData', state);
end
end

%% Keypress callback
% The function which does all the display stuff
function keypress_callback(fig, event_data)
% Check what key was pressed and update the image index as necessary
switch event_data.Character
    case 28 % Left
        up = -1; % Back a page
    case 29 % Right
        up = 1; % Forward a page
    case 30 % Up
        up = -0.1; % Back a row
    case 31 % Down
        up = 0.1; % Forward a row
    otherwise
        % Another key was pressed - ignore it
        return
end
% Use control and shift for faster scrolling
if ~isempty(event_data.Modifier)
    up = up * (2 ^ (strcmpi(event_data.Modifier, {'shift', 'control'}) * [1; 2]));
end
% Get the state data, if not given
state = get(fig, 'UserData');
% Get the current index
index = state.index;
% Get number of images
n = prod(state.layout);
% Generate 12 valid indices
if abs(up) < 1
    % Increment by row
    index = index + state.layout(2) * (up * 10) - 1;
else
    if state.layout(1) == 1
        % Increment by column
        index = index + up - 1;
    else
        % Increment by page
        index = index + n * up - 1;
    end
end
index = mod(index:index+n, state.n) + 1;
% Plot the images
figure(fig);
for a = 1:state.layout(1)
    for b = 1:state.layout(2)
        c = index(b + (state.layout(1) - a) * state.layout(2));
        A = state.I{c};
        if ischar(A)
            A = imread_rgb(A);
            state.I{c} = A;
        end
        set(state.h(a,b), 'CData', A);
    end
end
drawnow;
% Save the current index
state.index = index(1);
set(fig, 'UserData', state);
end

%% Choose a good layout for the images
function layout = choose_layout(n, y, x, layout)
layout = reshape(layout, 1, min(numel(layout), 2));
v = numel(layout);
N = isnan(layout);
if v == 0 || all(N)
    sz = get(0, 'ScreenSize');
    sz = sz(3:4) ./ [x y];
    layout = ceil(sz([2 1]) ./ sqrt(prod(sz) / n));
    switch ([prod(layout - [1 0]) prod(layout - [0 1])] >= n) * [2; 1]
        case 0
        case 1
            layout = layout - [0 1];
        case 2
            layout = layout - [1 0];
        case 3
            if min(sz .* (layout - [0 1])) > min(sz .* (layout - [1 0]))
                layout = layout - [0 1];
            else
                layout = layout - [1 0];
            end
    end
elseif v == 1
    layout = layout([1 1]);
elseif any(N)
    layout(N) = ceil(n / layout(~N));
end
end

%% Read image to uint8 rgb array
function A = imread_rgb(name)
try
    [A map] = imread(name);
catch
    % Format not recognized by imread, so create a red cross (along diagonals)
    A = eye(101) | diag(ones(100, 1), 1) | diag(ones(100, 1), -1);
    A = uint8(255 * (1 - (A | flipud(A))));
    A = cat(3, zeros(size(A), 'uint8')+uint8(255), A, A);
    return
end
if ~isempty(map)
    map = uint8(map * 256 - 0.5);
    %%JS change
    %A = reshape(map(A,:), [size(A) size(map, 2)]);
    A = reshape(map(A+1,:), [size(A) size(map, 2)]);
elseif size(A, 3) == 4
    ll = lower(name(end));
    if ll == 'f'
        % TIFF in CMYK colourspace - convert to RGB
        error('CMYK image files not yet supported - please fix.');
    elseif ll == 's'
        % RAS in RGBA colourspace - convert to RGB
        error('RGBA image files not yet supported - please fix.');
    end
end
end

%% Get the names of all images in a directory
function L = get_im_names
D = dir;
n = 0;
L = cell(size(D));
% Go through the directory list
for a = 1:numel(D)
    % Check if file is a supported image type
    if numel(D(a).name) > 4 && ~D(a).isdir && (any(strcmpi(D(a).name(end-3:end), {'.png', '.tif', '.jpg', '.bmp', '.ppm', '.pgm', '.pbm', '.gif', '.ras'})) || any(strcmpi(D(a).name(end-4:end), {'.tiff', '.jpeg'})))
        n = n + 1;
        L{n} = D(a).name;
    end
end
L = L(1:n);
end
%% Get the names of all images in a directory
function L = get_im_names2(dname)
D = dir(dname);
n = 0;
L = cell(size(D));
% Go through the directory list
for a = 1:numel(D)
    % Check if file is a supported image type
    if numel(D(a).name) > 4 && ~D(a).isdir && (any(strcmpi(D(a).name(end-3:end), {'.png', '.tif', '.jpg', '.bmp', '.ppm', '.pgm', '.pbm', '.gif', '.ras'})) || any(strcmpi(D(a).name(end-4:end), {'.tiff', '.jpeg'})))
        n = n + 1;
        L{n} = [dname, '/', D(a).name];
    end
end
L = L(1:n);
end


%% Parse inputs
function [map layout gap indices lims] = parse_inputs(inputs)

% Set defaults
map = gray(256);
layout = [];
gap = 0.01;
indices = -1;
lims = 0;

% Check for map
if numel(inputs) && isnumeric(inputs{1}) && size(inputs{1}, 2) == 3
    map = inputs{1};
    inputs = inputs(2:end);
end

% Go through option pairs
for a = 1:2:numel(inputs)
    switch lower(inputs{a})
        case 'map'
            map = inputs{a+1};
            if ischar(map)
                map = eval([map '(256)']);
            end
        case {'size', 'grid'}
            layout = inputs{a+1};
        case {'gap', 'border'}
            gap = inputs{a+1};
        case 'indices'
            indices = inputs{a+1};
        case {'lims', 'displayrange'}
            lims = inputs{a+1};
        otherwise
            error('Input option %s not recognized', inputs{a});
    end
end
end

%% Return default limits for the image type
function lims = default_limits(A)
if size(A, 3) == 1
    lims = [min(reshape(A, numel(A), 1)) max(reshape(A, numel(A), 1))];
else
    lims = [0 1];
    if ~isfloat(A)
        lims = lims * double(intmax(class(A)));
    end
end
end