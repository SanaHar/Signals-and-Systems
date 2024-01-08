clear;
clc;
%extracting train and test datas :
DATA = load('SubjectData6.mat');
data1 = DATA.train;
test1 = DATA.test;
%filtering : 
Subject1 = extract(data1,1);
Subject_test1 = extract(test1,0);
%index extraction of train data :
tempSubject = IndexExtraction(Subject1,1);
Subject1.time = tempSubject.time;
Subject1.index = tempSubject.index;


%%
%epoching train and test datas after filtering :
row10 = Subject1.DataSet(10, :);
n = 0;
for i=2:length(row10)
if(row10(1, i-1)==0 && row10(1, i)~=0)    
n = n + 1;      
stimuliOnset(1, n) = i;
end    
end

row10_test = Subject_test1.DataSet(10, :);
n = 0;
for i=2:length(row10_test)
if(row10_test(1, i-1)==0 && row10_test(1, i)~=0)    
n = n + 1;      
stimuliOnset_test(1, n) = i;
end    
end

epochTrain = epoching(Subject1.DataSet, 0.2, 0.8, stimuliOnset);
%%
%these are filtered and epoched datas for testing ml algorithms and they could be ignored 
E = load('epochData6.mat');
epochTrain = E.epoch;
E = load('epochtest6.mat');
epochTest = E.epoch;
%%
%converting 3D matrix of epoch to 2D for using as feature matrix :
train = preparingData(epochTrain,stimuliOnset,Subject1,1); 
test = preparingData(epochTest,stimuliOnset,Subject1,0); 
X_train = train.X_train;
response = train.response;
X_test = test.X_train;
%training models with SVM & LDA
Subject1.trainSVM = fitcsvm(X_train,response);
Subject1.trainLDA = fitcdiscr(X_train,response,'Cost',[0 10;0.5 0]);
%prediction :
predictionSVM = predict(Subject1.trainSVM,X_train);
predictionLDA = predict(Subject1.trainLDA ,X_train);
%recognizing the word :
WordSVM = WordDetection(predictionSVM,stimuliOnset,Subject1, 0);
WordLDA = WordDetection(predictionLDA,stimuliOnset,Subject1, 0);

%%
%smote algorithm; used for dealing with imbalanced classification with oversampling
N = 200;
N = ceil(N/100);
X_smote = X_train;
y_smote = response;
num = find(response);
for i=1:size(num, 1)
    y = X_train(i, :);
    [index, ~] = knnsearch(X_train,y,'k',5);
    index = datasample(index, N);
    x_nearest = X_train(index, :);
    x_syn = bsxfun(@plus, bsxfun(@times, bsxfun(@minus, x_nearest,y),rand(N,1)),y);
    X_smote = cat(1, X_smote,x_syn);
    y_smote = cat(1, y_smote, ones(size(x_syn, 1),1));
end


%%
function subject = extract(data,flag)
%if flag is 1 : we have train dataset 
subject.T_sampling = data(1,2) - data(1,1);
subject.f_sampling = 1/subject.T_sampling;

ch1 = data(2,:);
ch2 = data(3,:);
ch3 = data(4,:);
ch4 = data(5,:);
ch5 = data(6,:);
ch6 = data(7,:);
ch7 = data(8,:);
ch8 = data(9,:);
%%use smapimg filter in part 1 for ploting the curve ...

%cancling the DC part using fft: 
CH1 = fft(ch1);
CH1(1) = 0;
ch1 = ifft(CH1);
%caculating the cutoff-frequency based on signal's energy : 
freq = obw(ch1,subject.f_sampling);
disp(freq);
%cancling the DC part with subtraction the mean of signal & High-pass...
%filtering  
%we use band-pass as a result of using high-pass filter for cancling the DC part
% and low-pass filter for filtering noises
ch1 = ch1 - mean(ch1);
ch2 = ch2 - mean(ch2);
ch3 = ch3 - mean(ch3);
ch4 = ch4 - mean(ch4);
ch5 = ch5 - mean(ch5);
ch6 = ch6 - mean(ch6);
ch7 = ch7 - mean(ch7);
ch8 = ch8 - mean(ch8);

ch1 = bandpass(ch1,[1 100],subject.f_sampling);
ch2 = bandpass(ch2,[1 100],subject.f_sampling);
ch3 = bandpass(ch3,[1 100],subject.f_sampling);
ch4 = bandpass(ch4,[1 100],subject.f_sampling);
ch5 = bandpass(ch5,[1 100],subject.f_sampling);
ch6 = bandpass(ch6,[1 100],subject.f_sampling);
ch7 = bandpass(ch7,[1 100],subject.f_sampling);
ch8 = bandpass(ch8,[1 100],subject.f_sampling);

%calculating n of downsampling :
n = ceil(subject.f_sampling/(100));
disp(n);
subject.row1 = downsample(data(1,:), n);
subject.ch1 = downsample(ch1, n);
subject.ch2 = downsample(ch2, n);
subject.ch3 = downsample(ch3, n);
subject.ch4 = downsample(ch4, n);
subject.ch5 = downsample(ch5, n);
subject.ch6 = downsample(ch6, n);
subject.ch7 = downsample(ch7, n);
subject.ch8 = downsample(ch8, n);
subject.row10 = downsample(data(10,:), n);

subject.DataSet(1,:) = subject.row1;
subject.DataSet(2,:) = subject.ch1;
subject.DataSet(3,:) = subject.ch2;
subject.DataSet(4,:) = subject.ch3;
subject.DataSet(5,:) = subject.ch4;
subject.DataSet(6,:) = subject.ch5;
subject.DataSet(7,:) = subject.ch6;
subject.DataSet(8,:) = subject.ch7;
subject.DataSet(9,:) = subject.ch8;
subject.DataSet(10,:) = subject.row10;
if(flag)
    subject.row11 = downsample(data(11,:), n); 
    subject.DataSet(11,:) = subject.row11;
end
end

%%
function tansor = epoching(InputSignal, BackwardSamples, ForwardSamples, StimuliOnset)
T_smpling = InputSignal(1, 2) - InputSignal(1, 1);
%calculating the numbers of blocks before and after one stimulation according to... 
%backward and forward time as input:
back = floor(BackwardSamples/T_smpling);
forward = floor(ForwardSamples/T_smpling);
y = length(StimuliOnset) ;
tansor = zeros(8,back+forward+1,length(StimuliOnset)); 

for k=1:y                
    tansor(:,:,k) = InputSignal(2:9,StimuliOnset(1,k)-back:StimuliOnset(1,k)+forward);    
end
tansor = permute(tansor,[1,3,2])
end
%%
function twoD = preparingData(epoch,stimuliOnset,Subject,flag)
%%if flag is 1 : epoch is for train data set, else it's for test data set
I = size(epoch,1);
J = size(epoch,2);
K = size(epoch,3);
twoD.response = zeros(J , 1);
for i=1:J
    k = (i-1) * K + 1;
    for j=1:I
        twoD.X_train(i,K*(j-1)+1:K*j) = epoch(j,i,:);
    end
    if(Subject.DataSet(11, stimuliOnset(i)) == 1 && flag == 1) 
        twoD.response(i, 1) = 1;
    end
end
end
%%
function tempStruct = IndexExtraction(subject,flag)
%%flag is for defining if datdet is test(=0) or train(=1)
tempStruct.index = find(subject.DataSet(10, :));
tempStruct.time(1,:) = subject.DataSet(1, tempStruct.index);
if(flag == 1)
%%target & non-target is defined in the second row of time matrix
%%if data is target: 1
%%if data is not target: 0
tempStruct.time(2,:) = subject.DataSet(11, tempStruct.index);
end
end


%%
function error = ErrorCalc(Subject,method)
n=0;
for i=1:size(Subject.DataSet, 2)
    if(Subject.prediction(i, method) ~= Subject.DataSet(11, i))
        n = n+1;
    end
end
error = n/size(Subject.DataSet, 2) * 100;
end
%%
function detectedWord = WordDetection(prediction,stimuliOnset,Subject,flag)
%%flag is 1 if the experiment is SC 
%%flag is 0 if the experiment is RC
string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
letterArray = split(string,[""]);
strcolumn1 = "ABCDEF";
letterArray1 = split(strcolumn1,[""]);
strcolumn2 = "GHIJKL";
letterArray2 = split(strcolumn2,[""]);
strcolumn3 = "MNOPQR";
letterArray3 = split(strcolumn3,[""]);
strcolumn4 = "STUVWX";
letterArray4 = split(strcolumn4,[""]);
strcolumn5 = "YZ0123";
letterArray5 = split(strcolumn5,[""]);
strcolumn6 = "456789";
letterArray6 = split(strcolumn6,[""]);
detectedWord = "";

%finding the target numbers:
index = find(prediction);
targetLabel = stimuliOnset(index);
letters = Subject.DataSet(10, targetLabel);
disp(letters);
word(1, 1) = letters(1, 1);
if(flag)
j = 2;
for i=2:size(letters, 2)
    if(letters(1, i) ~= letters(1, i-1))
        word(1, j) = letters(1, i);
        j = j + 1;
    end
end
    for i=1:5
        detectedWord = detectedWord + letterArray(word(1, i) + 1, 1);
    end 
end
if(flag == 0)
    for i=0:30:length(letters)-30
        temp = unique(letters(i+1:30+i));
        temp = sort(temp);
        n = floor(i/30) + i/30 + 1;
        word(1, n) = temp(2);
        word(1, n+1) = temp(1); 
    end
       for i=1:2:10
       switch(word(1, i))
           case 7
               detectedWord = detectedWord + letterArray1(word(1, i+1) + 1, 1);
           case 8
               detectedWord = detectedWord + letterArray2(word(1, i+1) + 1, 1);
           case 9
               detectedWord = detectedWord + letterArray3(word(1, i+1) + 1, 1);
           case 10
               detectedWord = detectedWord + letterArray4(word(1, i+1) + 1, 1);              
           case 11
               detectedWord = detectedWord + letterArray5(word(1, i+1) + 1, 1);
           case 12
               detectedWord = detectedWord + letterArray6(word(1, i+1) + 1, 1);              
               
       end
   end
    
end

disp(word);
end
