# -*- coding: utf-8 -*-
import cv2
import time
import os
import json
import copy
from src_bianquenet_asff_pvt import BianqueNetPlus
import pandas as pd
from function.custom_transforms_mine import *
from function.segmentation_optimization import seg_opt
from function.calcu_DHI_512 import *
from function.calcu_signal_strength import calcu_Sigs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    #anaconda的环境下存在两个libiomp5md.dll文件。所以为了防止报错加的
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #将Tensor或者Model分配到GPU位置。

time_start = time.time()        # 计算训练所用的时间

'''
1、这个类的作用

'''
def resize_img_keep_ratio(img,target_size):
    # img = cv2.imread(img_name) # 读取图片
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
    return img_new
class DualCompose_img:
    def __init__(self, transforms):
        self.transforms = transforms  # 转换为类的属性

    def __call__(self, x):    #def __call__()类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
        for t in self.transforms:
            x = t(x)                           #t(x)是什么意思
        return x
image_only_transform = DualCompose_img([
    ToTensor_img(),
    Normalize_img()
])
#将图像均衡化
def clahe_cv(image):
    b, g, r = cv2.split(image) #函数 cv2.split () 传入一个图像数组，并将图像拆分为 B/G/R 三个通道。
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   #用于生成自适应均衡化图像
    '''
    cv2.createCLAHA(clipLimit, titleGridSize)
    clipLimit 颜色对比度的阈值
    titleGridSize 进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
    '''
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r) #使用 .apply 进行均衡化操作
    output_cv = cv2.merge([b, g, r]) #函数 cv2.merge() 将 B、G、R 单通道合并为 3 通道 BGR 彩色图像。
    return output_cv
classes = 15
model = BianqueNetPlus(num_classes=classes)
model_weight_path = "./save_weights/BianqueNet.pth"
weights_dict = torch.load(model_weight_path, map_location='cpu')['model']
for k in list(weights_dict.keys()):
    if "aux" in k:
        del weights_dict[k]
# load weights
model.load_state_dict(weights_dict)
model.to(device)
data_input_path = "./input/data_input"
results_output_path = "./output"
palette_path = "./palette.json"
quantitative_analysis_results_output_name = 'quantitative_analysis_results' + '.excel'
df = pd.DataFrame(columns = ['site','ID','age','gender','VA_I','VA_II','VA_III','VA_IV','VA_V','VA_VI','DA_1','DA_2',
                             'DA_3','DA_4','DA_5','CaA_Canal1','CaA_Canal2','CaA_Canal3','CaA_Canal4',
                             'CaA_Canal5','CoA_Cord1','CoA_Cord2','CoA_Cord3','CoA_Cord4','CoA_Cord5','DH_1',
                             'DH_2','DH_3','DH_4','DH_5','VD_I','VD_II','VD_III','VD_IV','VD_V','VD_VI','VH_I',
                             'VH_II','VH_III','VH_IV','VH_V','VH_VI','DD_1','DD_2','DD_3','DD_4','DD_5','CaD-Canal1',
                             'CaD-Canal2','CaD-Canal3','CaD-Canal4','CaD-Canal5','CoD-Cord1','CoD-Cord2','CoD-Cord3',
                             'CoD-Cord4','CoD-Cord5','SI_1','SI_2','SI_3','SI_4','SI_5'])
df.to_csv("./output/raw_data.csv",index=False) #路径可以根据需要更改
df1 = pd.DataFrame(columns = ['site','ID','age','gender','DHI_1',
                              'DHI_2','DHI_3','DHI_4','DHI_5','HDR_I','HDR_II','HDR_III','HDR_IV','HDR_V','HDR_VI',
                              'HDR_1','HDR_2','HDR_3','HDR_4','HDR_5','SI_1','SI_2','SI_3','SI_4','SI_5',
                              'CoD_1','CoD_2','CoD_3','CoD_4','CoD_5','C7_slope','cSVA','C2_C7_CL'])
df1.to_csv("./output/quantitative_data.csv",index=False) #路径可以根据需要更改
# df = pd.DataFrame(columns = ['id','VA_I','VA_II','VA_III','VA_IV','VA_V','VA_VI','DA_1','DA_2',
#                              'DA_3','DA_4','DA_5','CaA_Canal1','CaA_Canal2','CaA_Canal3','CaA_Canal4',
#                              'CaA_Canal5','CoA_Cord1','CoA_Cord2','CoA_Cord3','CoA_Cord4','CoA_Cord5','DH_1',
#                              'DH_2','DH_3','DH_4','DH_5','VD_I','VD_II','VD_III','VD_IV','VD_V','VD_VI','VH_I',
#                              'VH_II','VH_III','VH_IV','VH_V','VH_VI','DD_1','DD_2','DD_3','DD_4','DD_5','CaD-Canal1',
#                              'CaD-Canal2','CaD-Canal3','CaD-Canal4','CaD-Canal5','CoD-Cord1','CoD-Cord2','CoD-Cord3',
#                              'CoD-Cord4','CoD-Cord5','SI_1','SI_2','SI_3','SI_4','SI_5','SI_avg'])
# df.to_csv("./output/raw_data.csv",index=False) #路径可以根据需要更改
# df1 = pd.DataFrame(columns = ['id','DHI_1',
#                               'DHI_2','DHI_3','DHI_4','DHI_5','HDR_I','HDR_II','HDR_III','HDR_IV','HDR_V','HDR_VI',
#                               'HDR_1','HDR_2','HDR_3','HDR_4','HDR_5','SI_1','SI_2','SI_3','SI_4','SI_5',
#                               'CoD_1','CoD_2','CoD_3','CoD_4','CoD_5','C7_slope','cSVA','C2_C7_CL'])
# df1.to_csv("./output/quantitative_data.csv",index=False) #路径可以根据需要更改
dirList = os.listdir(data_input_path)
with open(palette_path, "rb") as f:
    pallette_dict = json.load(f)
    pallette = []
    for v in pallette_dict.values():
        pallette += v
model.eval()#不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，
with torch.no_grad():  #在该模块下，所有计算得出的tensor的requires_grad即自动求导操作都自动设置为False。
    for dir in dirList:
        data_input_dir = os.path.join(data_input_path, dir)
        data_output_path = os.path.join(results_output_path, dir)
        img_list = os.listdir(data_input_dir)
        for im_name in img_list:
            im_name_no_suffix = (dir.split('.')[0]).split('-')[-1]  #输出的是.之前的内容-之后的内容
            input_age = int(im_name_no_suffix[0:2]) #输出这里前三格的数据
            input_sex = int(im_name_no_suffix[2])
            im_site_ID = im_name.split('-')[0]
            input_site = str(im_site_ID[0:3])
            input_ID = str('_')+str(im_site_ID[3:6])
            im_path = os.path.join(data_input_dir, im_name)
            print('processing ' + str(im_path) + '.' * 20)
            input = cv2.imread(im_path)
            target_size = [512, 512]
            resized_img = resize_img_keep_ratio(input, target_size)
            # print(im_path)
            out_cv = clahe_cv(resized_img)    #将图像均衡化
            input_img = image_only_transform(out_cv)  #转为tensor格式
            input_img = torch.unsqueeze(input_img, 0)  #返回一个1xnxm的tensor
            pred_img = model(input_img.to(device))
            output = torch.nn.Softmax2d()(pred_img['out'])
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            output_seg_opt = output.clone()
            #利用squeeze（）函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了
            output_seg_opt = torch.squeeze(output_seg_opt).cpu().numpy()#这里加了cpu()
            output = seg_opt(output_seg_opt)
            #得到预测图
            prediction = pred_img['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(pallette)
            if not os.path.exists("./output/mask"):
                os.makedirs("./output/mask")
            mask.save(os.path.join("./output/mask",im_name.split('.')[0] + '.png'))
            try:
                jihe_parameter = []
                dcsf_point_parameter = []
                spinalcord_point_parameter = []
                vcsf_point_parameter = []
                imgadd_point_parameter = []
                time_calcu_DHI_bf = time.time()
                DHI, DWR, HD, HV, point_big_h, point_big_w, point_fenge_h_big, point_fenge_w_big,WV,HDR_VB,WD_big = calcu_DHI(output)
                jihe_parameter.append(HD)
                jihe_parameter.append(DHI)
                jihe_parameter.append(DWR)
                time_calcu_DHI_af = time.time()
                time_calcu_DHI = time_calcu_DHI_af - time_calcu_DHI_bf
                point_big_h = np.array(point_big_h)
                point_big_h = point_big_h.flatten()
                point_big_w = np.array(point_big_w)
                point_big_w = point_big_w.flatten()
                point_input_pic = copy.deepcopy(resized_img)
                point_size = 1
                point_color = (0, 0, 255)  # BGR
                point_color2 = (255,255,255)
                thickness = 4
                lineType = 4
                thickness1 = 2
                point_color1 = (83, 156, 222)  # BGR
                center_w,center_h = curvature_center(output)
                Curvature = round(qiujiajiao(center_h[0],center_h[2],center_h[5],center_w[0],center_w[2],center_w[5]),5)
                Cobb = round(T1_slope(point_big_h,point_big_w),5)
                cSVA = round(calcu_cSVA(point_big_w),5)
                C2_C7_CL = round(calcu_C2_C7_CL(point_big_h[2],point_big_h[3],point_big_h[22],point_big_h[23],point_big_w[2],
                                    point_big_w[3],point_big_w[22],point_big_w[23]),5)
                #     这里输出的只有脊髓曲线的坐标点以及整体曲线的坐标点
                for p in range(len(point_big_h)):
                    point = (point_big_w[p], point_big_h[p])
                     # if p % 2 == 0 and p != 0 and p != 22:
                    if p % 2 == 0 :
                        put_point = yc_line( (point_big_w[p],point_big_h[p] ),
                                         ( point_big_w[p + 1],point_big_h[p + 1]),55)
                        cv2.line(point_input_pic, (point_big_w[p],point_big_h[p] ),
                                        put_point, point_color1, thickness1, lineType)
                        spinalcord_point,imgadd_point= xie_area_point(output,(point_big_w[p],point_big_h[p] ),
                                        put_point)
                        imgadd_point_parameter.append(imgadd_point)
                        spinalcord_point_parameter.append(spinalcord_point)
                # 计算椎体、椎间盘区域的面积
                VB_area_parameter, disc_area_parameter = VB_disc_area(output)
                # 这部分计算椎体、椎间盘在spinalcord、椎管区域的面积
                d_spinalcord_area_parameter,d_imgadd_area_parameter = per_area(output,spinalcord_point_parameter,imgadd_point_parameter)
                #     # 这部分计算各节椎管矢状径和高度比
                diameter_d_imgadd = diameter(d_imgadd_area_parameter,HD)
                diameter_d_spinalcord = diameter(d_spinalcord_area_parameter,HD)
                # 这部分计算信号强度
                SI_parameter = []
                inputs_Sigs = resized_img
                output_Sigs = output.copy()
                SI_big_final, disc_si_dif_final = calcu_Sigs(inputs_Sigs, output_Sigs)
                SI_parameter.append(SI_big_final)
                SI_parameter = np.ravel(SI_parameter)
                disc_si_dif_final = disc_si_dif_final.tolist()
                disc_si_dif_final = np.ravel(disc_si_dif_final)
                VHI_parameter = VHI(HV, WV)
                VB_IVD_parameter = VB_IVD_tuchuzhishu(WV, WD_big)
                Canal_Disc_ratio_parameter = canal_Disc_ratio(diameter_d_imgadd, WD_big)
                Canal_spinal_ratio_parameter = canal_spinal_ratio(diameter_d_imgadd, diameter_d_spinalcord)
                list = [input_site,input_ID,input_age,input_sex,VB_area_parameter[0],VB_area_parameter[1],VB_area_parameter[2],
                        VB_area_parameter[3],VB_area_parameter[4],VB_area_parameter[5],disc_area_parameter[0],disc_area_parameter[1],
                        disc_area_parameter[2],disc_area_parameter[3],disc_area_parameter[4],d_imgadd_area_parameter[0],
                        d_imgadd_area_parameter[1],d_imgadd_area_parameter[2],d_imgadd_area_parameter[3],d_imgadd_area_parameter[4],
                        d_spinalcord_area_parameter[0],d_spinalcord_area_parameter[1],d_spinalcord_area_parameter[2],
                        d_spinalcord_area_parameter[3],d_spinalcord_area_parameter[4],HD[0],HD[1],HD[2],HD[3],HD[4],WV[0],
                        WV[1],WV[2],WV[3],WV[4],WV[5],HV[0],HV[1],HV[2],HV[3],HV[4],HV[5],WD_big[0],WD_big[1],WD_big[2],
                        WD_big[3],WD_big[4],diameter_d_imgadd[0],diameter_d_imgadd[1],diameter_d_imgadd[2],diameter_d_imgadd[3],
                        diameter_d_imgadd[4],diameter_d_spinalcord[0],diameter_d_spinalcord[1],diameter_d_spinalcord[2],
                        diameter_d_spinalcord[3],diameter_d_spinalcord[4],SI_parameter[0],SI_parameter[1],SI_parameter[2],
                        SI_parameter[3],SI_parameter[4]]
                data = pd.DataFrame([list])
                data.to_csv('./output/raw_data.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                list1 = [input_site,input_ID,input_age,input_sex,DHI[0],DHI[1],DHI[2],DHI[3],DHI[4],HDR_VB[0],HDR_VB[1],HDR_VB[2],
                        HDR_VB[3],HDR_VB[4],HDR_VB[5],DWR[0],DWR[1],DWR[2],DWR[3],DWR[4],disc_si_dif_final[0],disc_si_dif_final[1],
                         disc_si_dif_final[2],disc_si_dif_final[3],disc_si_dif_final[4],
                         Canal_Disc_ratio_parameter[0],Canal_Disc_ratio_parameter[1],Canal_Disc_ratio_parameter[2],
                         Canal_Disc_ratio_parameter[3],Canal_Disc_ratio_parameter[4],Cobb,cSVA,C2_C7_CL]
                data1 = pd.DataFrame([list1])
                data1.to_csv('./output/quantitative_data.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                # list = [str(im_name),VB_area_parameter[0],VB_area_parameter[1],VB_area_parameter[2],
                #         VB_area_parameter[3],VB_area_parameter[4],VB_area_parameter[5],disc_area_parameter[0],disc_area_parameter[1],
                #         disc_area_parameter[2],disc_area_parameter[3],disc_area_parameter[4],d_imgadd_area_parameter[0],
                #         d_imgadd_area_parameter[1],d_imgadd_area_parameter[2],d_imgadd_area_parameter[3],d_imgadd_area_parameter[4],
                #         d_spinalcord_area_parameter[0],d_spinalcord_area_parameter[1],d_spinalcord_area_parameter[2],
                #         d_spinalcord_area_parameter[3],d_spinalcord_area_parameter[4],HD[0],HD[1],HD[2],HD[3],HD[4],WV[0],
                #         WV[1],WV[2],WV[3],WV[4],WV[5],HV[0],HV[1],HV[2],HV[3],HV[4],HV[5],WD_big[0],WD_big[1],WD_big[2],
                #         WD_big[3],WD_big[4],diameter_d_imgadd[0],diameter_d_imgadd[1],diameter_d_imgadd[2],diameter_d_imgadd[3],
                #         diameter_d_imgadd[4],diameter_d_spinalcord[0],diameter_d_spinalcord[1],diameter_d_spinalcord[2],
                #         diameter_d_spinalcord[3],diameter_d_spinalcord[4],SI_parameter[0],SI_parameter[1],SI_parameter[2],
                #         SI_parameter[3],SI_parameter[4],SI_parameter[5]]
                # data = pd.DataFrame([list])
                # data.to_csv('./output/raw_data.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                # list1 = [str(im_name),DHI[0],DHI[1],DHI[2],DHI[3],DHI[4],HDR_VB[0],HDR_VB[1],HDR_VB[2],
                #         HDR_VB[3],HDR_VB[4],HDR_VB[5],DWR[0],DWR[1],DWR[2],DWR[3],DWR[4],disc_si_dif_final[0],disc_si_dif_final[1],
                #          disc_si_dif_final[2],disc_si_dif_final[3],disc_si_dif_final[4],
                #          Canal_Disc_ratio_parameter[0],Canal_Disc_ratio_parameter[1],Canal_Disc_ratio_parameter[2],
                #          Canal_Disc_ratio_parameter[3],Canal_Disc_ratio_parameter[4],Cobb,cSVA,C2_C7_CL]
                # data1 = pd.DataFrame([list1])
                # data1.to_csv('./output/quantitative_data.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                print("-----------" + str(im_path) + "计算成功")
            except Exception as e:
                print("---------------------------------------------------------the calculation of " + str(
                    im_path) + " picture is failed!")
                continue
            continue



