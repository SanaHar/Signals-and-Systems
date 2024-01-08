clc
%% 1 sampling
clc
% example for HalfBandFFt function
fs = 100; 
Ts = 1/fs; 
t = -1/2 : Ts : 1/2; 
x = abs(t);
figure
plot(t,x,'linewidth',2)
title('fourier transform of sampled signal')
x = ifft(x);
y = HalfBandFFt(x,fs);
title('fourier transform of signal')

% parseval theorem
sum(abs(x).^2,'all')
1/fs*sum(abs(y).^2,'all')
%% aliasing with fs=4
clc
clear all
w1 = linspace(0,3*pi/2,75);
w2 = linspace(-3*pi/2,0,75);
f = [zeros(1,25) , 2/(3*pi)*w2+1 , -2/(3*pi)*w1+1 ,zeros(1,25)];

figure
plot(linspace(-2*pi,2*pi,200),f,'r','linewidth',2)
f1 = [zeros(1,25),f]; 
hold on
plot(linspace(-0.5*pi,3.5*pi,225),f1,'r','linewidth',2)
f2 = [f,zeros(1,25)]; 
hold on
plot(linspace(-3.5*pi,0.5*pi,225),f2,'r','linewidth',2)
title('three period of sample signal with fs=4')
grid on
grid minor
%% 2 plot fft of electrodes data
clc
load 'Subject1.mat'
for i=2:9
    HalfBandFFt(SubjectData(i,:),256);
    title(strcat('fft of',num2str(i-1),'th electrode'))
end
%% 3 correlation clustering of 64channeldata
clc
clear all
load 'filter1.mat'
load '64channeldata.mat'

data0 = reshape(data,[size(data,1),size(data,2)*size(data,3)]);
f_data = zeros(size(data0));

%denoise signal
for i=1:size(data0,1)
f_data(i,:) = filtfilt(filter1,1,double(data0(i,:)));
end

% example of denoised signal n frequency domain
figure
y1 = plot(data0(1,:),'y');
hold on
y2 = plot(f_data(1,:),'r');
legend([y1,y2],{'data of channel one before denoising',...
    'data of channel one after denoising'});

HalfBandFFt(data0(1,:),2400);
title('data of channel one before denoising , frequency domain')
HalfBandFFt(f_data(1,:),2400);
title('data of channel one after denoising , frequency domain')

% downsample signal
M = 4;
l = floor(size(f_data,2)/M);
df_data = zeros(size(f_data,1),l);
for i=1:size(f_data,1)
df_data(i,:) = downSampler(double(f_data(i,:)),M);
end

figure
y1 = plot(f_data(1,:),'b');
hold on
y2 = plot(df_data(1,:),'r');
legend([y1,y2],{'data of channel one before downsampling',...
    'data of channel one after downsampling'});

HalfBandFFt(f_data(1,:),2400);
title('data of channel one before downsampling , frequency domain')

HalfBandFFt(df_data(1,:),2400/M);
title('data of channel one after downsampling , frequency domain')

% create correlation matrix
r = zeros(size(df_data,1),size(df_data,1));
for i=1:size(df_data,1)
    for j=1:size(df_data,1)
    x = df_data(i,:);
    y = df_data(j,:);
    r(i,j) = sum(x.*y,'all')/sqrt(sum(x.*x,'all')*sum(y.*y,'all'));
    end
end
% PhyloTree = seqlinkage(1-abs(r),'average');	
% view(PhyloTree)
%% plot topography of 64channeldata
clc
load locations.mat
load Standard_10-20_81ch.mat
load Standard_10-10_47ch.mat
C_mat = CorrelationCluster(1-abs(r),50,'upgma');
C = zeros(1,63);
for i=1:63
    [~,C(i)] = find(C_mat==i);
end
ch_list ={'AFZ','FP1','FP2','AF3','AF4','F7','F3','FZ','F4','F8','FC5',...
    'FC1','FC2','FC6','T7','C3','CZ','C4','T8','CP5','CP1','CP2','CP6',...
    'P7','P3','PZ','P4','P8','PO3','PO4','O1','O2','TP10','AF7','AF8',...
    'F5','F1','F2','F6','FT7','FC3','FCZ','FC4','FT8','C5','C1','C2','C6',...
    'TP7','CP3','CPZ','CP4','TP8','P5','P1','P2','P6','PO7','POZ','PO8',...
    'OZ','TP9','TP10'};
plot_topography(ch_list,C); 
%% 3 correlation clustering of SubjectData
clc
clear all
load 'filter1.mat'
load 'Subject1.mat'

data0 = SubjectData(2:9,:);
f_data = zeros(size(data0));

%denoise signal
for i=1:size(data0,1)
f_data(i,:) = filtfilt(filter1,1,double(data0(i,:)));
end

% example of denoised signal n frequency domain
figure
y1 = plot(data0(1,:),'y');
hold on
y2 = plot(f_data(1,:),'r');
legend([y1,y2],{'data of channel one before denoising',...
    'data of channel one after denoising'});

HalfBandFFt(data0(1,:),2400);
title('data of channel one before denoising , frequency domain')
HalfBandFFt(f_data(1,:),2400);
title('data of channel one after denoising , frequency domain')

% downsample signal
M = 4;
l = floor(size(f_data,2)/M);
df_data = zeros(size(f_data,1),l);
for i=1:size(f_data,1)
df_data(i,:) = downSampler(double(f_data(i,:)),M);
end

figure
y1 = plot(f_data(1,:),'b');
hold on
y2 = plot(df_data(1,:),'r');
legend([y1,y2],{'data of channel one before downsampling',...
    'data of channel one after downsampling'});

HalfBandFFt(f_data(1,:),2400);
title('data of channel one before downsampling , frequency domain')

HalfBandFFt(df_data(1,:),2400/M);
title('data of channel one after downsampling , frequency domain')

% create correlation matrix
r = zeros(size(df_data,1),size(df_data,1));
for i=1:size(df_data,1)
    for j=1:size(df_data,1)
    x = df_data(i,:);
    y = df_data(j,:);
    r(i,j) = sum(x.*y,'all')/sqrt(sum(x.*x,'all')*sum(y.*y,'all'));
    end
end
% PhyloTree = seqlinkage(1-abs(r),'average');	
% view(PhyloTree)
%% piot topography SunjectData
clc
load locations.mat
load Standard_10-20_81ch.mat
load Standard_10-10_47ch.mat
C_mat = CorrelationCluster(1-abs(r),10,'upgma');
C = zeros(1,63);
for i=1:63
    [~,C(i)] = find(C_mat==i);
end
ch_list ={'AFZ','FP1','FP2','AF3','AF4','F7','F3','FZ','F4','F8','FC5',...
    'FC1','FC2','FC6','T7','C3','CZ','C4','T8','CP5','CP1','CP2','CP6',...
    'P7','P3','PZ','P4','P8','PO3','PO4','O1','O2','TP10','AF7','AF8',...
    'F5','F1','F2','F6','FT7','FC3','FCZ','FC4','FT8','C5','C1','C2','C6',...
    'TP7','CP3','CPZ','CP4','TP8','P5','P1','P2','P6','PO7','POZ','PO8',...
    'OZ','TP9','TP10'};
ch_list ={'FZ', 'CZ', 'PZ', 'P4', 'P3', 'OZ', 'PO7','PO8'}
plot_topography(ch_list,C); 

%% 4 group delay
clc
clear all
N = 512;
h = rand(1000,1);
gd = groupdelay(h,N);
w = linspace(0,pi,length(gd));
figure;
y1 = plot(w,gd);
hold on
gdd = grpdelay(h,N);
y2 = plot(w,gdd);
legend([y1 y2],{'result of groupdelay','result of grpdelay'})

%% functions

% sampling
function y = HalfBandFFt(InputSignal,Fs)

y=fft(real(InputSignal));
n=length(y);
x=linspace(-1,1,n);
figure;
plot(Fs*x,abs(y),'linewidth',2); 
xlabel('frequency');
ylabel('|H(jW)|');
xlim([-0.5*Fs 0.5*Fs]);

end

function X_d= downSampler(X,m)
X_d = [ ];
for i=1:floor(length(X)/m)
   X_d = cat(2,X_d,X(m*(i-1)+1));   %X_d(k)=X(n(k-1)+1); 
end  
end

function clusters = CorrelationCluster(InputCorrMat,DistanceMeasure,algorithm)

N=size(InputCorrMat,1);
a=10*max(InputCorrMat,[],'all');
InputCorrMat=InputCorrMat-triu(InputCorrMat)+triu(a*ones(N));
c=linspace(1,N,N);
clusters=[c' zeros(N)]';

clusters=clusters(1:N,:);

for n =1:DistanceMeasure
[row,col]= find(InputCorrMat==min(InputCorrMat,[],'all'));
if row>col
  t = row;
  row = col;
  col = t;
end

if strcmp(algorithm,'upgma')
n1=nnz(clusters(:,col));
n2=nnz(clusters(:,row));
elseif strcmp(algorithm,'wpgma')
n1 =1;
n2 =1;
end

for i=1:row-1
   
InputCorrMat(row,i)=(n2*InputCorrMat(row,i)+n1*InputCorrMat(col,i))/(n1+n2); 

end

for i=row+1:N
    
  if(col>i)  
InputCorrMat(i,row)=(n2*InputCorrMat(i,row)+n1*InputCorrMat(col,i))/(n1+n2);
  end
  
  if(i>col)  
InputCorrMat(i,row)=(n2*InputCorrMat(i,row)+n1*InputCorrMat(i,col))/(n1+n2);
  end
    
end
InputCorrMat(col,:)=a; 
InputCorrMat(:,col)=a;    

b=zeros(1,N);

for j=1:N

  b(j)=clusters(j,col); 
      
end
clusters(:,col)=0;

l=1;
for j=1:N
    
if (clusters(j,row)==0)
   clusters(j,row) = b(l);
    l=l+1;
end


end

end

end

function gd = groupdelay(h,N)
h = reshape(h,size(h,1)*size(h,2),1);
n = 0:1:length(h)-1;
n = n';
H = fft(h,N);
dH = fft(h.*n,N);
gd = real(dH./H);
end
