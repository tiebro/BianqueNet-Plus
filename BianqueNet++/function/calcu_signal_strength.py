import numpy as np
import cv2
import math
from collections import Counter
from scipy import signal


def gaussBlur(image, sigma, H, W, _boundary='fill', _fillvalue=0):
    gaussKenrnel_x = cv2.getGaussianKernel(sigma, W, cv2.CV_64F)
    gaussKenrnel_x = np.transpose(gaussKenrnel_x)
    gaussBlur_x = signal.convolve2d(image, gaussKenrnel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    gaussKenrnel_y = cv2.getGaussianKernel(sigma, H, cv2.CV_64F)
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKenrnel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy


def get_pixel_value(inputs_Sigs, output_Sigs):
    inputs_Sigs = cv2.cvtColor(inputs_Sigs, cv2.COLOR_BGR2GRAY)
    blurImage = np.round(inputs_Sigs)
    inputs_Sigs = blurImage.astype(np.uint8)
    pixel_value_all = []


    for i in range(math.ceil(len(output_Sigs))):

        pixel_weizhi = np.where(output_Sigs[i , :, :] == 1)
        pixel_weizhi_w = pixel_weizhi[1]
        pixel_weizhi_h = pixel_weizhi[0]
        pixel_value_slice = []

        for j in range(len(pixel_weizhi_w)):
            pixel_value_point = inputs_Sigs[pixel_weizhi_h[j], pixel_weizhi_w[j]]

            pixel_value_slice.append(pixel_value_point)

        pixel_value_all.append(pixel_value_slice)
    return pixel_value_all


def calcu_Sigs(inputs, output_Sigs):
    pixel_value_all = get_pixel_value(inputs, output_Sigs)
    firstpeak = []
    secondpeak = []
    most_pixel = []
    for i in range(len(pixel_value_all)):
        most_pixel_slice = Counter(pixel_value_all[i]).most_common(1)[0][0]

        most_pixel.append(most_pixel_slice)
        '''
        cv2.calcHist()函数的作用：通过直方图可以很好的对整幅图像的灰度分布有一个整体的了解，直方图的x轴是灰度值（0~255），
        y轴是图片中具有同一个灰度值的点的数目。而calcHist（）函数则可以帮助我们统计一幅图像的直方图
        cv2.calcHist(images,channels,mask,histSize,ranges)
        images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应 用中括号 [] 括来例如[img]
        channels: 同样用中括号括来它会告函数我们统幅图 像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0][1][2] 
        它们分别对应着 BGR。
        mask: 掩模图像。统整幅图像的直方图就把它为 None。但是如果你想统图像某一分的直方图的你就制作一个掩模图像并使用它。
        histSize:BIN 的数目。也应用中括号括来
        BINS ：上面的直方图显示了每个像素值的像素数，即从0到255。即您需要256个值才能显示上述直方图。但是请考虑一下，如果您不需要单独查找所有像素值的
        像素数，而是在像素值间隔内查找像素数，该怎么办？例如，您需要找到介于 0 到 15 之间的像素数，然后是 16 到 31、...、240 到 255。您只需要 16 个值来表示直方图。
        因此，只需将整个直方图拆分为 16 个子部分，每个子部分的值就是其中所有像素计数的总和。这每个子部分都称为"BIN"。在第一种情况下，
        条柱数为256（每个像素一个），而在第二种情况下，它只有16。BINS 在 OpenCV 文档中由术语histSize表示。
        '''

        hist = cv2.calcHist([np.mat(pixel_value_all[i])], [0], None, [256], [0.0, 255.0])
        maxLoc = np.where(hist == np.max(hist))
        firstPeak_slice = maxLoc[0][0]
        firstpeak.append(firstPeak_slice)

        measureDists = np.zeros([256], np.float32)
        for k in range(256):
            measureDists[k] = math.fabs(k - firstPeak_slice * 0.95) * hist[k]  # 原始的系数是1.15
        maxLoc2 = np.where(measureDists == np.max(measureDists))
        secondPeak_slice = maxLoc2[0][0]
        secondpeak.append(secondPeak_slice)

    # 分别计算五个腰椎间盘高低信号强度（两个波峰）的差值，还有脑脊液和髂骨前脂肪区域的信号强度
    most_pixel_array = np.array(most_pixel, dtype=int)
    secondpeak_array = np.array(secondpeak, dtype=int)
    diff_fir_sec_peaks = np.abs((most_pixel_array - secondpeak_array))
    diff_fir_sec_peaks[0] = most_pixel_array[0]
    diff_fir_sec_peaks[12] = most_pixel_array[12]
    diff_fir_sec_peaks[14] = most_pixel_array[14]
    # print("most_pixel_array[3]",most_pixel_array[3])
    # print("secondpeak_array[3]", secondpeak_array[3])

    disc_si_dif_final = [diff_fir_sec_peaks[2], diff_fir_sec_peaks[4], diff_fir_sec_peaks[6],
                   diff_fir_sec_peaks[8], diff_fir_sec_peaks[10],diff_fir_sec_peaks[13]]


    #这一步是求最大的信号强度
    SI_disc_big = []
    diff_peaks = most_pixel_array - secondpeak_array
    for j in range(len(diff_peaks)):
        if diff_peaks[j] > 0:
            SI_disc = most_pixel_array[j]
        else:
            SI_disc = secondpeak_array[j]
        SI_disc_big.append(SI_disc)

    diff_fir_seg_peaks_average = (int(diff_fir_sec_peaks[12])+int(diff_fir_sec_peaks[14]))/2
    diff_fir_vb_average = (int(diff_fir_sec_peaks[2])+int(diff_fir_sec_peaks[4])+int(diff_fir_sec_peaks[6])+
                           int(diff_fir_sec_peaks[8])+int(diff_fir_sec_peaks[10])+int(diff_fir_sec_peaks[1])+
                           int(diff_fir_sec_peaks[3])+int(diff_fir_sec_peaks[5])+int(diff_fir_sec_peaks[11])+
                           int(diff_fir_sec_peaks[7])+int(diff_fir_sec_peaks[9]))/11

    SI_big_final = [SI_disc_big[2], SI_disc_big[4], SI_disc_big[6], SI_disc_big[8], SI_disc_big[10], SI_disc_big[13],
                         diff_fir_seg_peaks_average]

    disc_si_dif_final = np.array(disc_si_dif_final)*255/int(diff_fir_seg_peaks_average)
    # disc_si_dif_final = np.array(disc_si_dif_final)  / int(diff_fir_vb_average)
    # print("disc_si_dif_final",disc_si_dif_final)
    return SI_big_final, disc_si_dif_final