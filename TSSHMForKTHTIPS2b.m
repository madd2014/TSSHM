% TSSHM for KTH-TIPS2-b dataset
% Create date: 26/1/2018
% Author: Dongdong Ma


clc;
clear;
R = 8; 
r = 5;
train_setnum = 2;
P1 = 8;
if P1 == 8
    astype1 = 'uint8';
elseif P1 == 16
    astype1 = 'uint16';
elseif P1 == 32
    astype1 = 'uint32';
else
    astype1 = 'uint64';  
end
mapping1 = getmapping(P1,'riu2');
P2 = P1*2;
if P2 == 8
    astype2 = 'uint8';
elseif P2 == 16
    astype2 = 'uint16';
elseif P2 == 32
    astype2 = 'uint32';
else
    astype2 = 'uint64';  
end
mapping2 = getmapping(P2,'riu2');
% calculate the position of the 21 poinits. format: [delta_i,delta_j]
eight_spoins{1,1} = [0,R-r;-(R-r)/2,(R-r)/2;-(R-r),0;-(R-r)/2,-(R-r)/2;0,-(R-r);(R-r)/2,-(R-r)/2;(R-r),0;(R-r)/2,(R-r)/2];
eight_spoins{1,2} = [0,R+r;-(R+r)/2,(R+r)/2;-(R+r),0;-(R+r)/2,-(R+r)/2;0,-(R+r);(R+r)/2,-(R+r)/2;(R+r),0;(R+r)/2,(R+r)/2];
eight_spoins{1,3} = [0,R;-r,R;-(R+r)/2,(R+r)/2;-R,r;-R,0;-R,-r;-(R+r)/2,-(R+r)/2;-r,-R; ...
                     0,-R;r,-R;(R+r)/2,-(R+r)/2;R,-r;R,0;R,r;(R+r)/2,(R+r)/2;r,R];
% calculate the rest 4 points
temp = [2,4,6,8];
eight_spoins{1,1}(temp,:) = eight_spoins{1,1}(temp,:)*sqrt(2);
eight_spoins{1,2}(temp,:) = eight_spoins{1,2}(temp,:)*sqrt(2);

% read different classes 
prefix_pname = '/home/ma/work/madd/LBP/dataset/KTH-TIPS2/kth-tips2-b_col_200x200/KTH-TIPS2-b/';
classes = {'aluminium_foil','brown_bread','corduroy','cork','cotton','cracker','lettuce_leaf','linen','white_bread','wood','wool'};
correspond_num = {'15','48','42','16','46','60','23','44','52','54','22'};
sample_type = {'sample_a','sample_b','sample_c','sample_d'};
correspond_tnum = {'a','b','c','d'};

% calculate the traning vector
train_stype_num = [1 2];
test_stype_num = [3 4];
train_common_label = [];
train_TSSHM = [];
counter = 1;
for cls_num = 1:11                       % 11 classes
    for istype_num = 1:2
        for scale_num = 2:10
            for illum_num = 1:12
                stype_num = train_stype_num(istype_num);
                img_name = strcat(correspond_num{1,cls_num},correspond_tnum{1,stype_num},'-scale_',num2str(scale_num),'_im_',num2str(illum_num),'_col');
                compt_img_name = strcat(prefix_pname,classes{1,cls_num},'/',sample_type{1,stype_num},'/',img_name,'.png');
                Gray = imread(compt_img_name);

                Gray = rgb2gray(Gray);
                Gray = im2double(Gray);               
                
                image = (Gray-mean(Gray(:)))/std(Gray(:)); % normalization
                
                train_TSSHM(counter,:) = TSSHM(image,P1,P2,mapping1,mapping2,eight_spoins,R,r);
                train_common_label = [train_common_label;cls_num];
                counter = counter + 1;
            end
        end
    end
end

% calculate the testing vector
test_common_label = [];
test_TSSHM = [];
counter = 1;
for cls_num = 1:11                       % 11 classes
    for istype_num = 1:2                 % range:1~4
        for scale_num = 2:10
            for illum_num = 1:12
                stype_num = test_stype_num(istype_num);
                img_name = strcat(correspond_num{1,cls_num},correspond_tnum{1,stype_num},'-scale_',num2str(scale_num),'_im_',num2str(illum_num),'_col');
                compt_img_name = strcat(prefix_pname,classes{1,cls_num},'/',sample_type{1,stype_num},'/',img_name,'.png');
                Gray = imread(compt_img_name);                
                
                Gray = rgb2gray(Gray);
                Gray = im2double(Gray);
                image = (Gray-mean(Gray(:)))/std(Gray(:)); % normalization

                test_TSSHM(counter,:) = TSSHM(image,P1,P2,mapping1,mapping2,eight_spoins,R,r);
                test_common_label = [test_common_label;cls_num];
                counter = counter + 1;
            end
        end
    end
end

trains = train_TSSHM;
tests = test_TSSHM;
trainNum = size(trains,1);
testNum = size(tests,1);
DM = zeros(testNum,trainNum);
for i=1:testNum
    test = tests(i,:);        
    DM(i,:) = distMATChiSquare(trains,test)';
end
TSSHM_CP = ClassifyOnNN(DM,train_common_label,test_common_label)








