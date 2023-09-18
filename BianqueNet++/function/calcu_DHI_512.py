import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.draw import line
from PIL import Image

def yc_line(center, yc_point, dist):
    c_x, c_y = center
    t_x, t_y = yc_point
    x_dst = np.sqrt((center[0] - yc_point[0]) ** 2)
    y_dst = np.sqrt((center[1] - yc_point[1]) ** 2)
    # print(x_dst, y_dst)
    if t_y == c_y:
        if t_x < c_x:
            new_point = (t_x - dist, t_y)
        else:
            new_point = (t_x + dist, t_y)
        return new_point

    if t_x == c_x:
        if t_y < c_y:
            new_point = (t_x, t_y - dist)
        else:
            new_point = (t_x, t_y + dist)
        return new_point

    angle = np.arctan(y_dst / x_dst)
    angle_dg = angle * 180 / np.pi
    x_diff = np.cos(angle) * dist
    y_diff = np.sin(angle) * dist
    # print(x_diff, y_diff)
    if t_x < c_x and t_y < c_y:
        new_point = (t_x - x_diff, t_y - y_diff)
    elif t_x < c_x and t_y > c_y:
        new_point = (t_x - x_diff, t_y + y_diff)
    elif t_x > c_x and t_y < c_y:
        new_point = (t_x + x_diff, t_y - y_diff)
    else:
        new_point = (t_x + x_diff, t_y + y_diff)
    # print(angle, angle_dg)
    # print(new_point)
    new_point = (int(new_point[0]), int(new_point[1]))
    return new_point
# VH ********************************************************************************
# L1-L5
def calcu_HV(L1_calcu_HV):
    src = L1_calcu_HV.copy()
    # print(src.shape)
    # print("L1",np.sum(L1_calcu_HV))

    gray_harris = np.float32(src)
    gray_st = gray_harris.copy()
    # 2、Shi-Tomasi 角点检测
    maxCorners = 4
    qualityLevel = 0.01
    minDistance = 12
    block_size_s = 9
    k_s = 0.04
    corners_st = cv2.goodFeaturesToTrack(gray_st, maxCorners, qualityLevel, minDistance, corners=None, mask=None,
                                         blockSize=block_size_s, useHarrisDetector=None, k=k_s)
    corners = np.int0(corners_st)
    for i in corners:
        x, y = i.ravel()  #ravel()方法将数组维度拉成一维数组
        cv2.circle(src, (x, y), 2, (255, 0, 0), -1)
    # cv2.imwrite('new_bird.jpg', src)
    # cv2.imshow('point',src)
    # cv2.waitKey(0)
    # 第二种
    corners_label_final2 = np.int0(np.squeeze(corners_st))
    corners_label_final = corners_label_final2.copy()
    corners_label_zanshi = corners_label_final2.copy()
    corners_label_zanshi = corners_label_zanshi.tolist()
    # print("L1_hv",corners_label_final)
    # print("L1_HV",corners_label_final2.shape)
    # 因为这里得出的corners_label_final2的排序是混乱的，需要排序
    # 先找最大值和最小值对应的第4个（右下角）和第1点（左上角）
    sum_final2_wh = np.sum(corners_label_final, axis=1)#这是进行行求和,找到右下角和左上角
    # print("sum_final_2wh",sum_final2_wh)
    #np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
    wh_max = np.where(sum_final2_wh == np.max(sum_final2_wh))
    wh_min = np.where(sum_final2_wh == np.min(sum_final2_wh))
    # print("wh_max",wh_max)
    # print("wh_min",wh_min)
    wh_max = wh_max[0]
    wh_min = wh_min[0]
    # 根据wh_max和wh_min索引，将corners_label_final中的第4个和第1个位置，用corners_label_final2中的值替换
    corners_label_final[3, 0] = corners_label_final2[wh_max[0], 0]  # w，这表示右下角第四个点的横坐标
    corners_label_final[3, 1] = corners_label_final2[wh_max[0], 1]  # h，这表示右下角第四个点的纵坐标
    corners_label_final[0, 0] = corners_label_final2[wh_min[0], 0]  # w，这表示左上角第一个点的横坐标
    corners_label_final[0, 1] = corners_label_final2[wh_min[0], 1]  # h，这表示左上角第一个点的纵坐标
    # 再找剩下两个不是最值的索引，就是第2个（右上角）和第3个（左下角）的点，得到的是数组类型数据，转换成列表进行删除操作
    a_a = corners_label_final2[wh_max[0], :]
    b_b = corners_label_final2[wh_min[0], :]
    # print("a_a",a_a)
    a_a = a_a.tolist()
    b_b = b_b.tolist()
    corners_label_zanshi.remove(a_a)
    corners_label_zanshi.remove(b_b)
    corners_label_zanshi = np.array(corners_label_zanshi)
    # print("corners_label_zanshi",corners_label_zanshi)
    # znashi里存的是第二个点和第三个点，这里先比较了第二个点的纵坐标和第三个点的纵坐标，纵坐标大的就是第三个点，存入原来的集合中
    if corners_label_zanshi[0, 1] > corners_label_zanshi[1, 1]:
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[0, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[1, 1]  # h
    else:
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[1, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[0, 1]  # h
    corners_label_final_end = corners_label_final.copy()
    # print("corners_label_final_end[0, 0]",corners_label_final_end[0, 0])
    # print("corners_label_final_end[1, 0]", corners_label_final_end[1, 0])
    # print("corners_label_final_end[2, 0]", corners_label_final_end[2, 0])
    # print("corners_label_final_end[3, 0]", corners_label_final_end[3, 0])
    # 比较右上角的w值否小于左上角的，是的话，调换1，2位置，否则不调换
    if corners_label_final[1, 0] < corners_label_final[0, 0]:
        corners_label_final_end[0, 0] = corners_label_final[1, 0]
        corners_label_final_end[0, 1] = corners_label_final[1, 1]
        corners_label_final_end[1, 0] = corners_label_final[0, 0]
        corners_label_final_end[1, 1] = corners_label_final[0, 1]
    # 比较右下角的w值否小于左下角的，是的话，调换3，4位置，否则不调换
    if corners_label_final[3, 0] < corners_label_final[2, 0]:
        corners_label_final_end[2, 0] = corners_label_final[3, 0]
        corners_label_final_end[2, 1] = corners_label_final[3, 1]
        corners_label_final_end[3, 0] = corners_label_final[2, 0]
        corners_label_final_end[3, 1] = corners_label_final[2, 1]
    # 提取角点位置
    # 这里l_h是角点纵坐标的集合，l_2是角点横坐标的集合
    point_L_h_bf_test = corners_label_final_end[:, 1]
    point_L_w_bf_test = corners_label_final_end[:, 0]

    if len(point_L_h_bf_test) == 4:
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()
    else:
        # 打印角点数，起提示作用
        print('The number of corners of the detected cone is', len(point_L_h_bf_test))
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()
    # print("pint_l_w",point_L_w)
    # print("pint_l_h", point_L_h)
    # 这里的point集合已经把所有的角点排序过了，已经把对应的角点位置对应的横纵坐标放在相同的索引位置了
    # img = paint(src,[point_L_h[0],point_L_w[0]],[point_L_h[1],point_L_w[1]])
    # cv2.imshow("img",img)
    # cv2.waitKey(0)




    # 计算椎体的高度
    # 函数返回它的所有参数的平方和的平方根
    # 这里先对1、2两个角点的距离计算，再对3、4两个焦点的距离进行计算，最后取了平均作为椎体的宽度
    L_wid_up = math.hypot((point_L_h[0] - point_L_h[1]), (point_L_w[0] - point_L_w[1]))
    L_wid_down = math.hypot((point_L_h[2] - point_L_h[3]), (point_L_w[2] - point_L_w[3]))
    plt.plot([point_L_w[0], point_L_h[0]], [point_L_w[1], point_L_h[1]], color="green")
    L_wid = (L_wid_up + L_wid_down) / 2.0
    HV = np.sum(L1_calcu_HV) / L_wid
    # print("area",np.sum(src))
    # print("HV",HV)
    return HV, point_L_h, point_L_w,L_wid


# S1
def calcu_HV_S1(L1_calcu_HV):
    # 保存原始L1，计算像素个数和
    src = L1_calcu_HV.copy()
    # 对椎体进行角点检测
    gray_harris = np.float32(src)
    gray_st = gray_harris.copy()

    # 2、Shi-Tomasi 角点检测
    maxCorners = 4
    qualityLevel = 0.01
    minDistance = 14
    block_size_s = 9
    k_s = 0.04
    corners_st = cv2.goodFeaturesToTrack(gray_st, maxCorners, qualityLevel, minDistance, corners=None, mask=None,
                                         blockSize=block_size_s, useHarrisDetector=None, k=k_s)
    # 第二种
    corners_label_final2 = np.int0(np.squeeze(corners_st))
    #通过np.squeeze()函数转换后，要显示的数组变成了秩为1的数组，即（5，），可以在坐标系中画出
    corners_label_final = corners_label_final2.copy()
    corners_label_zanshi = corners_label_final2.copy()
    corners_label_zanshi = corners_label_zanshi.tolist()
    # print("hv_S1",corners_label_final2)
    # print("HV_S1",corners_label_final2.shape)
    # 先找最大值和最小值对应的第4个（右下角）和第1点（左上角）
    sum_final2_wh = np.sum(corners_label_final, axis=1)
    wh_max = np.where(sum_final2_wh == np.max(sum_final2_wh))
    wh_min = np.where(sum_final2_wh == np.min(sum_final2_wh))
    wh_max = wh_max[0]
    wh_min = wh_min[0]
    # 根据wh_max和wh_min索引，将corners_label_final中的第4个和第1个位置，用corners_label_final2中的值替换
    corners_label_final[3, 0] = corners_label_final2[wh_max[0], 0]  # w
    corners_label_final[3, 1] = corners_label_final2[wh_max[0], 1]  # h
    corners_label_final[0, 0] = corners_label_final2[wh_min[0], 0]  # w
    corners_label_final[0, 1] = corners_label_final2[wh_min[0], 1]  # h
    # 再找剩下两个不是最值的索引，就是第2个（右上角）和第3个（左下角）的点，得到的是数组类型数据，转换成列表进行删除操作
    a_a = corners_label_final2[wh_max[0], :]
    b_b = corners_label_final2[wh_min[0], :]
    a_a = a_a.tolist()
    b_b = b_b.tolist()
    corners_label_zanshi.remove(a_a)
    corners_label_zanshi.remove(b_b)
    corners_label_zanshi = np.array(corners_label_zanshi)

    if (corners_label_zanshi[0,0]+corners_label_zanshi[0,1]) > (corners_label_zanshi[1,0]+corners_label_zanshi[1,1]):
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[0, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[1, 1]  # h
    else:
        # 第3个
        corners_label_final[2, 0] = corners_label_zanshi[1, 0]  # w
        corners_label_final[2, 1] = corners_label_zanshi[1, 1]  # h
        # 第2个
        corners_label_final[1, 0] = corners_label_zanshi[0, 0]  # w
        corners_label_final[1, 1] = corners_label_zanshi[0, 1]  # h

    corners_label_final_end = corners_label_final.copy()
    # 比较右上角的w值否小于左上角的，是的话，调换1，2位置，否则不调换
    if corners_label_final[1, 0] < corners_label_final[0, 0]:
        corners_label_final_end[0, 0] = corners_label_final[1, 0]
        corners_label_final_end[0, 1] = corners_label_final[1, 1]
        corners_label_final_end[1, 0] = corners_label_final[0, 0]
        corners_label_final_end[1, 1] = corners_label_final[0, 1]
    # 比较右下角的w值否小于左下角的，是的话，调换3，4位置，否则不调换
    if corners_label_final[3, 0] < corners_label_final[2, 0]:
        corners_label_final_end[2, 0] = corners_label_final[3, 0]
        corners_label_final_end[2, 1] = corners_label_final[3, 1]
        corners_label_final_end[3, 0] = corners_label_final[2, 0]
        corners_label_final_end[3, 1] = corners_label_final[2, 1]
    # 提取角点位置
    point_L_h_bf_test = corners_label_final_end[:, 1]
    point_L_w_bf_test = corners_label_final_end[:, 0]
    if len(point_L_h_bf_test) == 4:
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()
    else:
        # 打印角点数，起提示作用
        print('The number of corners of the detected cone is', len(point_L_h_bf_test))
        point_L_w = point_L_w_bf_test.copy()
        point_L_h = point_L_h_bf_test.copy()

    # 现在得到四个点，计算椎体的宽度
    L_wid_up = math.hypot((point_L_h[0] - point_L_h[1]), (point_L_w[0] - point_L_w[1]))
    L_wid_down = math.hypot((point_L_h[2] - point_L_h[3]), (point_L_w[2] - point_L_w[3]))
    L_wid = (L_wid_up + L_wid_down) / 2.0
    HV = np.sum(L1_calcu_HV) / L_wid
    return HV, point_L_h, point_L_w,L_wid


# WD ********************************************************************************
# 计算完整的腰椎间盘的宽度big_WD
def calcu_big_WD(L1_L2_disc_calcu_HD, point_D12_1, point_D12_2):
    # 对索引值进行取整，
    h0 = int(point_D12_1[0])
    w0 = int(point_D12_1[1])
    h1 = int(point_D12_2[0])
    w1 = int(point_D12_2[1])
    L1_L2_disc_calcu_HD = np.array(L1_L2_disc_calcu_HD)
    len_pic = len(L1_L2_disc_calcu_HD)
    if h0 == h1:
        print('斜率为零！')
        # 计算最右边的点,从右边中点开始往右遍历，记录像素值为0时，左边一个点
        for s in range(w1, len_pic):
            if L1_L2_disc_calcu_HD[h1, s] == 0:
                point_you = [h1, s - 1]
                break

        # 计算最左边的点，从左边点开始往左遍历，记录像素值为0时，右边一个点
        for t in range(w0, 0, -1):
            if L1_L2_disc_calcu_HD[h0, t] == 0:
                point_zuo = [h0, t + 1]
                break

    else:
        m, b = slope(point_D12_1[0], point_D12_1[1], point_D12_2[0], point_D12_2[1])  # 计算斜率还是用没有取整的数，为了精确
        # 计算最右边的点,从右边中点开始往右遍历，记录像素值为0时，左边一个点
        for s in range(w1, len_pic):
            if L1_L2_disc_calcu_HD[int(m * s + b), s] == 0:
                point_you = [int(m * (s - 1) + b), s - 1]
                break

        # 计算最左边的点，从左边点开始往左遍历，记录像素值为0时，右边一个点
        for t in range(w0, 0, -1):
            if L1_L2_disc_calcu_HD[int(m * t + b), t] == 0:
                point_zuo = [int(m * (t + 1) + b), t + 1]
                break

    point_you_zuo = np.array(point_you) - np.array(point_zuo)
    big_WD = math.hypot(point_you_zuo[0], point_you_zuo[1])

    return big_WD


# calcu_WD计算腰椎间盘宽度，腰椎间盘80%区域四个分割点
def calcu_WD(L1_L2_disc_calcu_HD, point_L1_h, point_L1_w, point_L2_h, point_L2_w):
    point_fenge = []
    # 上一个椎体的下面两个顶点，下一椎体的上面两个顶点
    point_L1_3 = np.array([point_L1_h[2], point_L1_w[2]])
    point_L1_4 = np.array([point_L1_h[3], point_L1_w[3]])
    point_L2_1 = np.array([point_L2_h[0], point_L2_w[0]])
    point_L2_2 = np.array([point_L2_h[1], point_L2_w[1]])
    # 腰椎间盘前后中点，point_Di（i+1）_k，k取1,2
    point_D12_1 = (point_L1_3 + point_L2_1) / 2
    point_D12_2 = (point_L1_4 + point_L2_2) / 2
    # 腰椎间盘宽度small_WD，small_WD，是根据椎体的四个角点来确定腰椎间盘的范围，进而计算的。在这四个角点得出的分割线之外还有膨出的腰椎间盘，
    # 故这个不是真正意义上的腰椎间盘宽度，所以称为small_WD,包括膨出的腰椎间盘的宽度称为big_WD。
    point_D12_21 = point_D12_2 - point_D12_1
    small_WD = math.hypot(point_D12_21[0], point_D12_21[1])
    big_WD = calcu_big_WD(L1_L2_disc_calcu_HD, point_D12_1, point_D12_2)
    # 计算前后中点h,w方向差值
    delta_12_h = math.fabs(point_D12_1[0] - point_D12_2[0])
    delta_12_w = math.fabs(point_D12_1[1] - point_D12_2[1])
    # 椎间盘中心点point_D12_c0
    point_D12_c0 = (point_D12_1 + point_D12_2) / 2
    # 计算腰椎间盘80%区域四个分割点,分割占比miu=0.8，分割边界的高度qiang=1，表示一个单位的delta_12_w
    miu = 0.8
    qiang = 0.75  # 0.75
    miu_half = 0.5*miu
    qiang_half = 0.5*qiang
    # 在这里要分两种情况，一种是前后中点连线的斜率是负的，前中点高于后中点（h值小）：另一种是正的，前中点低于后中点（h值大）
    if point_D12_1[0] < point_D12_2[0]:  # 对比h值
        point_D12_c0lu = np.array([point_D12_c0[0] - miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0lu))
        point_D12_c0ld = np.array([point_D12_c0[0] - miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ld))
        point_D12_c0ru = np.array([point_D12_c0[0] + miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ru))
        point_D12_c0rd = np.array([point_D12_c0[0] + miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0rd))
    else:
        point_D12_c0lu = np.array([point_D12_c0[0] + miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0lu))
        point_D12_c0ld = np.array([point_D12_c0[0] + miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] - miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ld))
        point_D12_c0ru = np.array([point_D12_c0[0] - miu_half * delta_12_h - qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w - qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0ru))
        point_D12_c0rd = np.array([point_D12_c0[0] - miu_half * delta_12_h + qiang_half * delta_12_w,
                                   point_D12_c0[1] + miu_half * delta_12_w + qiang_half * delta_12_h])
        point_fenge.append(np.int0(point_D12_c0rd))

    return small_WD, big_WD, point_fenge


# DH ********************************************************************************
# 计算斜率
def slope(h0, w0, h1, w1):
    m = (h1-h0)/(w1-w0)  # 把h当成y，w当成x,
    b = h0 - m*w0
    return m, b


# 计算两条分割线上的所有像素点位置
def get_pixel(h0, w0, h1, w1):
    piont_h = []
    piont_w = []
    if w0 == w1:
        print('斜率不存在！')
        for j in range(h0, h1):
            piont_h.append(j)
            piont_w.append(w0)
    else:
        m, b = slope(h0, w0, h1, w1)
        for j in range(h0, h1):
            point_w_temp = (j - b)/m
            piont_w.append(round(point_w_temp))
            piont_h.append(j)

    return piont_h, piont_w


# calcu_HD计算腰椎间盘高度
def calcu_HD(L1_L2_disc_calcu_HD, point_D12_l_up, point_D12_l_down, point_D12_r_up, point_D12_r_down, w_D):
    L1_L2_disc_HD = L1_L2_disc_calcu_HD.copy()
    len_pic = len(L1_L2_disc_HD)
    # 腰椎间盘80%区域分割点，四个
    point_D12_l_up = point_D12_l_up
    point_D12_l_down = point_D12_l_down
    point_D12_r_up = point_D12_r_up
    point_D12_r_down = point_D12_r_down
    # 腰椎间盘80%区域分割点连成的两条直线（前）上的所有像素点
    point_D12_l_up2down_h, point_D12_l_up2down_w = get_pixel(point_D12_l_up[0], point_D12_l_up[1], point_D12_l_down[0],
                                                             point_D12_l_down[1])
    len_D12_l_up2down_h = len(point_D12_l_up2down_h)
    # 从整个腰椎间盘取80%，去除前10%
    for j in range(len_D12_l_up2down_h):
        for i in range(point_D12_l_up2down_w[j]):
            L1_L2_disc_HD[point_D12_l_up2down_h[j], i] = 0
    point_D12_r_up2down_h, point_D12_r_up2down_w = get_pixel(point_D12_r_up[0], point_D12_r_up[1], point_D12_r_down[0],
                                                             point_D12_r_down[1])
    len_D12_r_up2down_h = len(point_D12_r_up2down_h)
    # 从整个腰椎间盘取80%，去除后10%
    for j in range(len_D12_r_up2down_h):
        for i in range(point_D12_r_up2down_w[j], len_pic):
            L1_L2_disc_HD[point_D12_r_up2down_h[j], i] = 0
    sum_A = np.sum(L1_L2_disc_HD)
    HD = sum_A/w_D
    return HD


# DHI&HDR ********************************************************************************
def calcu_DHI(output):
    # 先提取腰椎间盘上下两个椎体的各四个角点，point_Li_j，i从1到5，j从1到4
    # 腰椎间盘前后中点，point_Di（i+1）_k，k取1,2
    # 腰椎间盘80%区域前后中点，point_D+i（i+1）_l，point_Di（i+1）_r
    # 腰椎间盘80%区域分割点，四个，point_Di（i+1）_l_up,point_Di（i+1）_l_down,point_Di（i+1）_r_up,point_Di（i+1）_r_down
    # 腰椎间盘80%区域分割点连成的两条直线（前后）上的所有像素点，point_Di（i+1）_l_up2down,point_Di（i+1）_r_up2down,加h后缀表示
    # 像素点的h方向的值组成的list
    # 腰椎间盘宽度w_D
    DHI_big = []
    DWR_big = []
    HV_big = []
    HD_big = []
    WD_big = []
    WV = []
    HDR_VB = []

    point_fenge_big = []

    # DHI12
    # print('Calculating the DHI of the L1_L2_disc......')
    L1 = output[1, :, :]
    # print(L1)
    # cv2.imshow(L1)
    HV1, point_L1_h, point_L1_w,wv1 = calcu_HV(L1)
    HV_big.append(round(HV1,5))
    WV.append(round(wv1,5))
    HDR = HV1 / wv1
    HDR_VB.append(round(HDR,5))
    L1_L2_disc = output[2, :, :]
    L2 = output[3, :, :]
    HV2, point_L2_h, point_L2_w,wv2 = calcu_HV(L2)
    HV_big.append(round(HV2,5))
    WV.append(round(wv2,5))
    HDR = HV2 / wv2
    HDR_VB.append(round(HDR,5))
    small_WD12, big_WD12, point_fenge12 = calcu_WD(L1_L2_disc, point_L1_h, point_L1_w, point_L2_h, point_L2_w)
    WD_big.append(round(big_WD12,5))
    point_fenge_big.append(np.array(point_fenge12))
    DH12 = calcu_HD(L1_L2_disc, point_fenge12[0], point_fenge12[1], point_fenge12[2], point_fenge12[3], small_WD12)
    HD_big.append(round(DH12,5))
    # 计算DHI指数
    DHI12 = 2 * DH12 / (HV1 + HV2)
    DHI_big.append(round(DHI12,5))
    # print('The DHI of the L1_L2_disc is ', DHI12)
    # 计算高宽比
    DWR12 = DH12 / big_WD12
    DWR_big.append(round(DWR12,5))
    # print('The DWR of the L1_L2_disc is ', DWR12)

    # DHI23
    # print('Calculating the DHI of the L2_L3_disc......')
    L2_L3_disc = output[4, :, :]
    L3 = output[5, :, :]
    HV3, point_L3_h, point_L3_w,wv3 = calcu_HV(L3)
    HV_big.append(round(HV3,5))
    WV.append(round(wv3,5))
    HDR = HV3 / wv3
    HDR_VB.append(round(HDR,5))
    small_WD23, big_WD23, point_fenge23 = calcu_WD(L2_L3_disc, point_L2_h, point_L2_w, point_L3_h, point_L3_w)
    WD_big.append(round(big_WD23,5))
    point_fenge_big.append(np.array(point_fenge23))
    DH23 = calcu_HD(L2_L3_disc, point_fenge23[0], point_fenge23[1], point_fenge23[2], point_fenge23[3], small_WD23)
    HD_big.append(round(DH23,5))
    DHI23 = 2 * DH23 / (HV2 + HV3)
    DHI_big.append(round(DHI23,5))
    # print('The DHI of the L2_L3_disc is ', DHI23)
    # 计算高宽比
    DWR23 = DH23 / big_WD23
    DWR_big.append(round(DWR23,5))
    # print('The DWR of the L2_L3_disc is ', DWR23)

    # DHI34
    # print('Calculating the DHI of the L3_L4_disc......')
    L3_L4_disc = output[6, :, :]
    L4 = output[7, :, :]
    HV4, point_L4_h, point_L4_w,wv4 = calcu_HV(L4)
    HV_big.append(round(HV4,5))
    WV.append(round(wv4,5))
    HDR = HV4 / wv4
    HDR_VB.append(round(HDR,5))
    small_WD34, big_WD34, point_fenge34 = calcu_WD(L3_L4_disc, point_L3_h, point_L3_w, point_L4_h, point_L4_w)
    WD_big.append(round(big_WD34,5))
    point_fenge_big.append(np.array(point_fenge34))
    DH34 = calcu_HD(L3_L4_disc, point_fenge34[0], point_fenge34[1], point_fenge34[2], point_fenge34[3], small_WD34)
    HD_big.append(round(DH34,5))
    DHI34 = 2 * DH34 / (HV3 + HV4)
    DHI_big.append(round(DHI34,5))
    # print('The DHI of the L3_L4_disc is ', DHI34)
    # 计算高宽比
    DWR34 = DH34 / big_WD34
    DWR_big.append(round(DWR34,5))
    # print('The DWR of the L3_L4_disc is ', DWR34)

    # DHI45
    # print('Calculating the DHI of the L4_L5_disc......')
    L4_L5_disc = output[8, :, :]
    L5 = output[9, :, :]
    HV5, point_L5_h, point_L5_w,wv5 = calcu_HV(L5)
    HV_big.append(round(HV5,5))
    WV.append(round(wv5,5))
    HDR = HV5 / wv5
    HDR_VB.append(round(HDR,5))
    small_WD45, big_WD45, point_fenge45 = calcu_WD(L4_L5_disc, point_L4_h, point_L4_w, point_L5_h, point_L5_w)
    WD_big.append(round(big_WD45,5))
    point_fenge_big.append(np.array(point_fenge45))
    DH45 = calcu_HD(L4_L5_disc, point_fenge45[0], point_fenge45[1], point_fenge45[2], point_fenge45[3], small_WD45)
    HD_big.append(round(DH45,5))
    DHI45 = 2 * DH45 / (HV4 + HV5)
    DHI_big.append(round(DHI45,5))
    # print('The DHI of the L4_L5_disc is ', DHI45)
    # 计算高宽比
    DWR45 = DH45 / big_WD45
    DWR_big.append(round(DWR45,5))
    # print('The DWR of the L4_L5_disc is ', DWR45)


    # DHI5S1
    # print('Calculating the DHI of the L5_S1_disc......')
    L5_S1_disc = output[10, :, :]
    S1 = output[11, :, :]
    HVS1, point_S1_h, point_S1_w,wv6 = calcu_HV_S1(S1)
    HV_big.append(round(HVS1,5))
    WV.append(round(wv6,5))
    HDR = HVS1 / wv6
    HDR_VB.append(round(HDR,5))
    small_WD5S1, big_WD5S1, point_fenge5S1 = calcu_WD(L5_S1_disc, point_L5_h, point_L5_w, point_S1_h, point_S1_w)
    WD_big.append(round(big_WD5S1,5))
    point_fenge_big.append(np.array(point_fenge5S1))
    DH5S1 = calcu_HD(L5_S1_disc, point_fenge5S1[0], point_fenge5S1[1], point_fenge5S1[2], point_fenge5S1[3], small_WD5S1)
    HD_big.append(round(DH5S1,5))
    # DHI5S1 = 2 * DH5S1 / (HVS1 + HV5)
    # 稍微更改一下第五节腰椎DHI指数的计算公式，不用S1的高度了
    DHI5S1 = DH5S1 / HV5
    DHI_big.append(round(DHI5S1,5))
    # print('The DHI of the L5_S1_disc is ', DHI5S1)
    # 计算高宽比
    DWR5S1 = DH5S1 / big_WD5S1
    DWR_big.append(round(DWR5S1,5))
    # print('The DWR of the L5_S1_disc is ', DWR5S1)

    # 存储所有角点
    point_big_h = [point_L1_h, point_L2_h, point_L3_h, point_L4_h, point_L5_h, point_S1_h]
    point_big_w = [point_L1_w, point_L2_w, point_L3_w, point_L4_w, point_L5_w, point_S1_w]
    # 储存所有分割点
    point_fenge_big = np.array(point_fenge_big)
    point_fenge_h_big = [point_fenge_big[0, :, 0], point_fenge_big[1, :, 0], point_fenge_big[2, :, 0],
                         point_fenge_big[3, :, 0], point_fenge_big[4, :, 0]]
    point_fenge_w_big = [point_fenge_big[0, :, 1], point_fenge_big[1, :, 1], point_fenge_big[2, :, 1],
                         point_fenge_big[3, :, 1], point_fenge_big[4, :, 1]]

    return DHI_big, DWR_big, HD_big, HV_big, point_big_h, point_big_w, point_fenge_h_big, point_fenge_w_big,WV,HDR_VB,\
           WD_big


# 这里dcsf、vcsf的顺序反了，但是输出的结果是对的，vcsf的对应点和dcsf的对应点都是对的
def xie_area_point(output,point1,point2):
    # # 读取图片并灰度值化
    spinalcord_point = []
    imgadd_point = []

    spinal_cord = output[13, :, :]
    spinal_cord = (spinal_cord * 255).astype(np.uint8)
    # 假定只有一个轮廓conts[0]
    conts, _ = cv2.findContours(spinal_cord, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # 颜色区分，把原图再转为BGR
    spinal_cord = cv2.cvtColor(spinal_cord, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(spinal_cord, conts, -1, (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(conts[0])
    cv2.rectangle(spinal_cord, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # 画一条直线
    p1, p2 = point1, point2
    if p2[0] != p1[0]:  # 若存在斜率 y=kx+b
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - p1[0] * k
        # 求解直线和boundingbox的交点A和B
        pa, pb = (x, int(k * x + b)), ((x + w), int(k * (x + w) + b))
    else:  # 若斜率不存在，垂直的直线
        pa, pb = (p1[0], y), (p1[0], y + h)
    cv2.circle(spinal_cord, pa, 2, (0, 255, 255), 2)
    cv2.circle(spinal_cord, pb, 2, (0, 255, 255), 2)
    cv2.putText(spinal_cord, 'A', (pa[0] - 10, pa[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(spinal_cord, 'B', (pb[0] + 10, pb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.line(spinal_cord, p1, p2, (255, 255, 0), 1)

    line_point = np.array(list(zip(*line(*pa, *pb))))
    # print(len(line_point))
    for i in range(1,len(line_point)):

        point_h = line_point[i-1][0]
        point_w = line_point[i-1][1]
        point_h1 = line_point[i][0]
        point_w1 = line_point[i][1]
        if cv2.pointPolygonTest(conts[0], (int(point_h), int(point_w)), False) == 0 and cv2.pointPolygonTest(conts[0],
          (int(point_h1), int(point_w1)),False) != 0:  # 若点在轮廓上
            cv2.circle(spinal_cord, (int(point_h), int(point_w)), 2, (0, 0, 255), 2)
            spinalcord_point.append((point_h,point_w))
        elif cv2.pointPolygonTest(conts[0], (int(point_h), int(point_w)), False) + cv2.pointPolygonTest(conts[0],
            (int(point_h1), int(point_w1)), False) == 0 and cv2.pointPolygonTest(conts[0], (int(point_h), int(point_w)), False) != 0:
            point_h2 = int((point_h+point_h1)/2)
            point_w2 = int((point_w+point_w1)/2)
            cv2.circle(spinal_cord, (int(point_h2), int(point_w2)), 2, (0, 0, 255), 2)
            spinalcord_point.append((point_h2, point_w2))
    # cv2.imshow('0', spinal_cord)
    # cv2.waitKey(0)

    # print("len_vcsf_point",len(vcsf_point))
    # print("len_spinal_point", len(spinalcord_point))
    # print("len_dcsf_point", len(dcsf_point))


    # 三条曲线合并之后的整体曲线
    vcsf1 = output[12, :, :]
    spinal_cord1 = output[13,:,:]
    dcsf1 = output[14,:,:]
    imgadd = cv2.add(vcsf1,spinal_cord1,dcsf1)
    imgadd = (imgadd * 255).astype(np.uint8)
    # cv2.imshow("imgad",imgadd)
    # cv2.waitKey(0)

    # 假定只有一个轮廓conts[0]
    conts, _ = cv2.findContours(imgadd, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # 颜色区分，把原图再转为BGR
    imgadd = cv2.cvtColor(imgadd, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imgadd, conts, -1, (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(conts[0])
    cv2.rectangle(imgadd, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # 画一条直线
    p1, p2 = point1, point2
    if p2[0] != p1[0]:  # 若存在斜率 y=kx+b
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - p1[0] * k
        # 求解直线和boundingbox的交点A和B
        pa, pb = (x, int(k * x + b)), ((x + w), int(k * (x + w) + b))
    else:  # 若斜率不存在，垂直的直线
        pa, pb = (p1[0], y), (p1[0], y + h)
    cv2.circle(imgadd, pa, 2, (0, 255, 255), 2)
    cv2.circle(imgadd, pb, 2, (0, 255, 255), 2)
    cv2.putText(imgadd, 'A', (pa[0] - 10, pa[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(imgadd, 'B', (pb[0] + 10, pb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.line(imgadd, p1, p2, (255, 255, 0), 1)
    # 枚举2点之间的线段点
    # print(list(zip(*line(*pa, *pb))))
    # for pt in zip(*line(*pa, *pb)):
    #     if cv2.pointPolygonTest(conts[0], (int(pt[0]),int(pt[1])), False) == 0:  # 若点在轮廓上
    #         cv2.circle(dcsf, (int(pt[0]),int(pt[1])), 2, (0, 0, 255), 2)
    #         dcsf_point.append(pt)
    # print(dcsf_point)
    # 这里如果继续用元组那么就无法导出不在轮廓线上的点，有些点并不会落在轮廓线上，通过确定在轮廓线外和在轮廓线内的点求中点去大致得到轮廓线上的点

    line_point = np.array(list(zip(*line(*pa, *pb))))
    # print(line_point)
    for i in range(1, len(line_point)):
        # print(cv2.pointPolygonTest(conts[0], (int(pt2[0]), int(pt2[1])), False))
        point_h = line_point[i - 1][0]
        point_w = line_point[i - 1][1]
        point_h1 = line_point[i][0]
        point_w1 = line_point[i][1]
        if cv2.pointPolygonTest(conts[0], (int(point_h), int(point_w)), False) == 0 and cv2.pointPolygonTest(conts[0],
              (int(point_h1),int(point_w1)),False) != 0:  # 若点在轮廓上
            cv2.circle(imgadd, (int(point_h), int(point_w)), 2, (0, 0, 255), 2)
            imgadd_point.append((point_h, point_w))
        elif cv2.pointPolygonTest(conts[0], (int(point_h), int(point_w)), False) + cv2.pointPolygonTest(conts[0],
                    (int(point_h1),int(point_w1)),False) == 0 and cv2.pointPolygonTest(conts[0], (int(point_h), int(point_w)), False) != 0:
            point_h2 = int((point_h + point_h1) / 2)
            point_w2 = int((point_w + point_w1) / 2)
            cv2.circle(imgadd, (int(point_h2), int(point_w2)), 2, (0, 0, 255), 2)
            imgadd_point.append((point_h2, point_w2))
    # print("imgadd_point",imgadd_point)
    # cv2.imshow('0', imgadd)
    # cv2.waitKey(0)
    return spinalcord_point,imgadd_point

# 先计算s1
def xie_area(dcsf,dcsf_point_left_up,dcsf_point_right_up,dcsf_point_left_down,dcsf_point_right_down):
    # dcsf = output[12, :, :]
    # dcsf = (dcsf * 255).astype(np.uint8)
    # conts, _ = cv2.findContours(dcsf, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # dcsf_point_left_up = dcsf_point_parameter[0][0]
    # dcsf_point_right_up = dcsf_point_parameter[0][1]
    # dcsf_point_zanshi1_y = max(dcsf_point_left_up[1],dcsf_point_right_up[1])
    # dcsf_point_left_down = dcsf_point_parameter[1][0]
    # dcsf_point_right_down = dcsf_point_parameter[1][1]
    # dcsf_point_zanshi2_y = min(dcsf_point_left_down[1],dcsf_point_right_down[1])
    dcsf_point_left_up = dcsf_point_left_up
    dcsf_point_right_up = dcsf_point_right_up
    dcsf_point_left_down = dcsf_point_left_down
    dcsf_point_right_down = dcsf_point_right_down
    dcsf =dcsf
    dcsf = (dcsf * 255).astype(np.uint8)
    conts, _ = cv2.findContours(dcsf, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    dcsf_point_zanshi1_y = int(max(dcsf_point_left_up[1],dcsf_point_right_up[1]))
    dcsf_point_zanshi2_y = int(min(dcsf_point_left_down[1],dcsf_point_right_down[1]))
    s1 = 0
    s2 = 0
    s3 = 0
    # print("dcsf_point_zanshi1_y",dcsf_point_zanshi1_y)
    for y in range(dcsf_point_zanshi1_y,dcsf_point_zanshi2_y):
        for x in range(len(dcsf)):
            # print("2")
            if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 0:
                s1 = s1 +1
            if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 1:
                s1 = s1 +1
    # print("s1",s1)

    if dcsf_point_left_up[1] > dcsf_point_right_up[1]:
        m,b = slope(dcsf_point_left_up[1],dcsf_point_left_up[0],dcsf_point_right_up[1],dcsf_point_right_up[0])
        # print("dcsf_point_left_up[1]",dcsf_point_left_up[1])
        # print("dcsf_point_right_up[1]",dcsf_point_right_up[1])
        for y in range(dcsf_point_right_up[1],dcsf_point_left_up[1]):
            for x in range(len(dcsf)):
                if x != dcsf_point_left_up[0]:
                    # print("2")
                    if (y-dcsf_point_left_up[1])/(x-dcsf_point_left_up[0]) < 0 and (y-dcsf_point_left_up[1])/(x-dcsf_point_left_up[0]) >= m:
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 0:
                            s2 = s2 + 1
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 1:
                            s2 = s2 + 1
    elif dcsf_point_left_up[1] < dcsf_point_right_up[1]:
        m, b = slope(dcsf_point_right_up[1], dcsf_point_right_up[0],dcsf_point_left_up[1], dcsf_point_left_up[0])
        # print("m", m)
        for y in range(dcsf_point_left_up[1],dcsf_point_right_up[1]):
            for x in range(len(dcsf)):
                if x != dcsf_point_right_up[0]:
                    if (y - dcsf_point_right_up[1]) / (x - dcsf_point_right_up[0]) > 0 and (y - dcsf_point_right_up[1]) / (
                            x - dcsf_point_right_up[0]) <= m:
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 0:
                            s2 = s2 + 1
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 1:
                            s2 = s2 + 1
    # print("s2",s2)
    # 算下边两个点围成的面积
    # print("dcsf_point_left_down[1]",dcsf_point_left_down[1])
    # print("dcsf_point_right_down[1]",dcsf_point_right_down[1])
    if dcsf_point_left_down[1] > dcsf_point_right_down[1]:
        m, b = slope(dcsf_point_right_down[1], dcsf_point_right_down[0],dcsf_point_left_down[1], dcsf_point_left_down[0])
        # print("m",m)
        for y in range( dcsf_point_right_down[1],dcsf_point_left_down[1]):
            for x in range(len(dcsf)):
                if x != dcsf_point_right_down[0]:
                    if (y - dcsf_point_right_down[1]) / (x - dcsf_point_right_down[0]) < 0 and (y - dcsf_point_right_down[1]) / (
                            x - dcsf_point_right_down[0]) >= m:
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 0:
                            s3 = s3 + 1
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 1:
                            s3 = s3 + 1
    elif dcsf_point_left_down[1] < dcsf_point_right_down[1]:
        m,b = slope(dcsf_point_left_down[1],dcsf_point_left_down[0],dcsf_point_right_down[1],dcsf_point_right_down[0])
        for y in range(dcsf_point_left_down[1],dcsf_point_right_down[1]):
            for x in range(len(dcsf)):
                if x != dcsf_point_left_down[0]:
                    if (y-dcsf_point_left_down[1])/(x-dcsf_point_left_down[0]) > 0 and (y-dcsf_point_left_down[1])/(x-dcsf_point_left_down[0]) <= m:
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 0:
                            s3 = s3 + 1
                        if cv2.pointPolygonTest(conts[0], (int(x), int(y)), False) == 1:
                            s3 = s3 + 1
    # print("s3",s3)
    s = s1 + s2 + s3
    return s

def per_area(output,spinalcord_point_parameter,imgadd_point_parameter):

    spinal_cord = output[13,:,:]
    d_spinalcord_area_parameter = []
    for i in range(1,len(spinalcord_point_parameter)-2,2):
        spinalcord_point_left_up = spinalcord_point_parameter[i][0]
        spinalcord_point_right_up = spinalcord_point_parameter[i][1]
        spinalcord_point_left_down = spinalcord_point_parameter[i+1][0]
        spinalcord_point_right_down = spinalcord_point_parameter[i+1][1]
        s = xie_area(spinal_cord,spinalcord_point_left_up,spinalcord_point_right_up,spinalcord_point_left_down,spinalcord_point_right_down)
        d_spinalcord_area_parameter.append(s)


    vcsf = output[12,:,:]
    dcsf = output[14,:,:]
    imgadd = cv2.add(vcsf,spinal_cord,dcsf)
    d_imgadd_area_parameter = []
    for i in range(1,len(imgadd_point_parameter)-2,2):
        imgadd_point_left_up = imgadd_point_parameter[i][0]
        imgadd_point_right_up = imgadd_point_parameter[i][1]
        imgadd_point_left_down = imgadd_point_parameter[i+1][0]
        imgadd_point_right_down = imgadd_point_parameter[i+1][1]
        s = xie_area(imgadd,imgadd_point_left_up,imgadd_point_right_up,imgadd_point_left_down,imgadd_point_right_down)
        d_imgadd_area_parameter.append(s)
    return d_spinalcord_area_parameter,d_imgadd_area_parameter


def diameter(area_parameter,HD):
    diameter_jihe = []
    for i in range(len(HD)):
        d = area_parameter[i] / HD[i]
        diameter_jihe.append(round(d,5))
    return diameter_jihe

def VHI(VH,VD):
    VHI_parameter = []
    for i in range(len(VH)):
        VHI = VH[i] / VD[i]
        VHI_parameter.append(round(VHI,5))
    return VHI_parameter




def VB_disc_area(output):
    VB_area_parameter = []
    disc_area_parameter = []

    for i in range(1,len(output)-2,2):
        VB_area = np.sum(output[i])
        VB_area_parameter.append(VB_area)
    for i in range(2, len(output) - 4, 2):
        disc_area = np.sum(output[i])
        disc_area_parameter.append(disc_area)

    return VB_area_parameter,disc_area_parameter



def VB_IVD_tuchuzhishu(WV,WD):
    VB_IVD_parameter = []
    for i in range(len(WD)):
        ratio = WD[i] / ((WV[i]+WV[i+1]) / 2)
        VB_IVD_parameter.append(round(ratio,5))
    return VB_IVD_parameter

#V2 这里开始修改了参数，取消了两条csf曲线的参数，增加了两条csf和spinalcord融合的整体的参数

def canal_Disc_ratio(diameter_d_imgadd,WD_big):
    canal_Disc_ratio_parameter = []
    for i in range(len(WD_big)):
        ratio = diameter_d_imgadd[i] / WD_big[i]
        canal_Disc_ratio_parameter.append(round(ratio,5))
    return canal_Disc_ratio_parameter

def canal_spinal_ratio(diameter_d_imgadd,diameter_d_spinalcord):
    canal_spinal_ratio_parameter = []
    for i in range(len(diameter_d_spinalcord)):
        ratio = diameter_d_imgadd[i] / diameter_d_spinalcord[i]
        canal_spinal_ratio_parameter.append(round(ratio,5))
    return canal_spinal_ratio_parameter


def curvature_center(output):
    center_w = []
    center_h = []
    for i in range(1, len(output)-2, 2):
        img = output[i]
        img = (img * 255).astype(np.uint8)
        h1, w1 = img.shape
        contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:  # 轮廓总数
            continue
        M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        # image = np.zeros([h1, w1], dtype=img.dtype)
        # cv2.drawContours(image, contours, 0, 255, -1)  # 绘制轮廓，填充
        # cv2.circle(image, (center_x, center_y), 1, 128, -1)  # 绘制中心点
        center_w.append(center_x)
        center_h.append(center_y)
    return center_w,center_h

def qiujiajiao(Coords1y,Coords2y,Coords3y,Coords1x,Coords2x,Coords3x):
    # 求出斜率
    k1 = (Coords2y - Coords1y) / (float(Coords2x - Coords1x))
    k2 = (Coords3y - Coords2y) / (float(Coords3x - Coords2x))

    # 方向向量
    x = np.array([1, k1])
    y = np.array([1, k2])
    # 模长
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    # 根据向量之间求其夹角并四舍五入
    Cobb = float((np.arccos(x.dot(y) / (float(Lx * Ly))) * 180 / np.pi) + 0.5)
    if Cobb > 90:
        Cobb_ruijiao = float(abs(180 - Cobb))
    else:
        Cobb_ruijiao = Cobb
    return Cobb_ruijiao

def T1_slope(point_big_h,point_big_w):
    T1_left_point_w = point_big_w[22]
    T1_left_point_h = point_big_h[22]
    T1_right_point_w = point_big_w[23]
    T1_right_point_h = point_big_h[23]
    k = (T1_right_point_h - T1_left_point_h) / (T1_right_point_w - T1_left_point_w)
    # print('T1_left_point_h',T1_left_point_h)
    #
    # print('h',T1_right_point_h - T1_left_point_h)
    # print('w',T1_right_point_w - T1_left_point_w)
    # print('k',k)
    Cobb = abs(float(np.arctan(float(k)) * 180 / np.pi + 0.5))
    if Cobb >60:
        Cobb_ruijiao = float(abs(90 - Cobb))
    else:
        Cobb_ruijiao = Cobb
    return Cobb_ruijiao


def calcu_cSVA(point_big_w):
    C2_left_down_point_w = point_big_w[2]
    C7_right_up_point_w = point_big_w[21]
    csva_1 =  round(abs(C7_right_up_point_w-C2_left_down_point_w),5)
    return csva_1

def calcu_C2_C7_CL(Coords1y,Coords2y,Coords3y,Coords4y,Coords1x,Coords2x,Coords3x,Coords4x):
    # 求出斜率
    k1 = (Coords2y - Coords1y) / (float(Coords2x - Coords1x))
    k2 = (Coords4y - Coords3y) / (float(Coords4x - Coords3x))

    # 方向向量
    x = np.array([1, k1])
    y = np.array([1, k2])
    # 模长
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    # 根据向量之间求其夹角并四舍五入
    Cobb = float((np.arccos(x.dot(y) / (float(Lx * Ly))) * 180 / np.pi) + 0.5)
    if Cobb > 90:
        Cobb_ruijiao = float(abs(180 - Cobb))
    else:
        Cobb_ruijiao = Cobb
    return Cobb_ruijiao