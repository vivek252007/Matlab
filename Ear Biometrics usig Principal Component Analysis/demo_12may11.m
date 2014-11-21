
% combinedlda
clc
close all
clear all
num_class=77;                                    
num_inclass=3;                                  
num_all=num_class*num_inclass;                    
m=197;                                         
n=100;                                        
M=m*n;

%X=readimage(num_class,num_inclass,m,n);          
load image_data;                                  

for i=1:231
tmp=reshape(X(:,i),197,100);
% figure,imshow(uint8(tmp))
% imwrite(uint8(tmp),['data\' num2str(i) '.jpg']);
end
% figure,imshow(uint8(tmp))

%function project_combinedlda=combinedlda(X,num_inclass,num_class,M)

%num_all=num_inclass*num_class;              % num_inclass

sum_X=zeros(M,1);
mean_X=zeros(M,1);
sum_inclass=zeros(M,num_class);
mean_inclass=zeros(M,num_class);

fai_b=zeros(M,num_class);
fai_t=zeros(M,num_all);
fai_w=zeros(M,num_all);

Vtempst=zeros(num_all,num_all-1);
Vswnull=zeros(num_all-1,num_class-1);
V=zeros(num_all-num_class,num_class-1);
%project=zeros(M,2*(num_class-1));
% feature extraction

% PCA algorithm for FE
% mean calculation
for i=1:num_all
   sum_X=sum_X+X(:,i);
end
mean_X=sum_X/num_all;                               

for i=1:num_class
    for j=1:num_inclass
        sum_inclass(:,i)=sum_inclass(:,i)+X(:,(i-1)*num_inclass+j);
    end
    mean_inclass(:,i)=sum_inclass(:,i)/num_inclass;   % µmean_inclass
end

for i=1:num_class    
   fai_b(:,i)=mean_inclass(:,i)-mean_X;                           % 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:num_all
   fai_t(:,i)=X(:,i)-mean_X;   
end

temp_st=fai_t'*fai_t;

[V_tempst,D_tempst]=eig(temp_st);

for i=2:num_all
    Vtempst(:,i-1)=V_tempst(:,i);
end

Vst=fai_t*Vtempst;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i=1:num_class
    for j=1:num_inclass
       fai_w(:,(i-1)*num_inclass+j)=X(:,(i-1)*num_inclass+j)-mean_inclass(:,i); 
    end
end

temp_vfaiw=fai_w'*Vst;
temp_sw=temp_vfaiw'*temp_vfaiw;

[V_tempsw,D_tempsw]=eig(temp_sw);

for i=1:(num_class-1)
    Vswnull(:,i)=V_tempsw(:,i);                                 % Vswnull
end

for i=1:num_all-num_class    
   Vswnonnull(:,i)=V_tempsw(:,num_class-1+i);                   % Vswnonnull
end

temp_project1=Vst*Vswnull;                                      % project1
project_temp=Vst*Vswnonnull;                                    % project_temp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


temp_projectsbnull=fai_b'*temp_project1;
temp_sbnull=temp_projectsbnull'*temp_projectsbnull;
[Vsbnull,Dsbnull]=eig(temp_sbnull);

d=60;
Vnull=zeros(num_class-1,d);

for i=1:d
    Vnull(:,i)=Vsbnull(:,num_class-i);
end

project1=temp_project1*Vnull;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


temp_projectfaiw=fai_w'*project_temp;
temp_swnonnull=temp_projectfaiw'*temp_projectfaiw;                 

temp_projectfaib=fai_b'*project_temp;
temp_sbnonnull=temp_projectfaib'*temp_projectfaib;                  

[Vnonnull,Dnonnull]=eig(inv(temp_swnonnull)*temp_sbnonnull);       

for i=1:num_class-1
    V(:,i)=Vnonnull(:,i);
end

project2=project_temp*V;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


project=zeros(M,(d+num_class-1));

for i=1:d
    project(:,i)=project1(:,i);
end
for i=1:num_class-1
    project(:,(d+i))=project2(:,i);
end
project_combinedlda=project';
%project_combinedlda=project2';
%project_combinedlda=project1';
%save project_combinedlda project_combinedlda;
%save mean_X mean_X;
%save fai_t fai_t;
%return
X_combinedlda=project_combinedlda*fai_t;

%save X_combinedlda X_combinedlda;
% matching and recognition for ear data base
load imagetest_data;

min=1;

distance=zeros(1,num_all);
remember=zeros(1,num_class);



for i=1:num_class
     
    X_test_combinedlda=project_combinedlda*(X_test(:,i)-mean_X);
    
    for j=1:num_all
        temp=(X_test_combinedlda-X_combinedlda(:,j))'*(X_test_combinedlda-X_combinedlda(:,j));
        distance(1,j)=sqrt(temp);
    end
    for j=1:num_all
        if  distance(1,j)<distance(1,min)
            min=j;
        end
    end
    remember(1,i)=min;
end

result_class=zeros(1,num_class);
num_recognition=0;

for i=1:num_class
    
    remainder=mod(remember(1,i),num_inclass);
    quotient=fix(remember(1,i)/num_inclass);
    
    if(remainder==0)
        result_class(1,i)=quotient;
    else
        result_class(1,i)=quotient+1;
    end
    
    if(result_class(1,i)==i)
        num_recognition=num_recognition+1;
    end
    
end

ratio_recognition=(num_recognition/num_class)*100
