import math

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from numpy.linalg import norm


class FusionEval_R():

    def __init__(self, img_label, img_result):
        assert img_label.shape == img_result.shape, 'Fusion Evaluation: image shapes do not match!'

        # original image uint8
        self.img_label = img_label
        self.img_result = img_result

        # new image float64
        self.label_np = img_label.astype(np.float64)
        self.result_np = img_result.astype(np.float64)
        self.multi = (len(img_label.shape) != 2)
        pass

    # 1. spectral angle mapper
    def cal_sam(self):
        # cal sam for single image
        def _sam(img_label, img_result):
            img_label_1D = img_label.flatten()  # [H, W, C] ==> [ H * W * C ]
            img_result_1D = img_result.flatten()  # [H, W, C] ==> [ H * W * C ]
            up = np.sum(img_label_1D * img_result_1D)

            label_nor = np.linalg.norm(img_label_1D, ord=2)
            result_nor = np.linalg.norm(img_result_1D, ord=2)
            down = label_nor * result_nor
            return np.arccos(np.clip(up / down, -1, 1))

        return _sam(self.label_np, self.result_np)

    # cal cc for single image
    def _cc(self, img_label, img_result):
        label_mean = np.mean(img_label)
        result_mean = np.mean(img_result)
        up = np.sum((img_label - label_mean) * (img_result - result_mean))
        down = np.sqrt(np.sum((img_label - label_mean) ** 2) * np.sum((img_result - result_mean) ** 2))
        return up / down

    # 2. Correlation Coefficient
    def cal_cc(self):
        if self.multi:
            res = 0
            for i in range(self.label_np.shape[2]):
                res += self._cc(self.label_np[:, :, i], self.result_np[:, :, i])
            return res / self.label_np.shape[2]
        else:
            return self._cc(self.label_np, self.result_np)
        pass

    # 3. Spatial Correlation Coefficient
    def cal_scc(self):
        kernel = np.array((
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]), dtype="float32")
        if self.multi:
            res = 0
            for i in range(self.img_label.shape[2]):
                label = cv2.filter2D(self.img_label[:, :, i], -1, kernel).astype(np.float64)
                result = cv2.filter2D(self.img_result[:, :, i], -1, kernel).astype(np.float64)
                res += self._cc(label, result)
            return res / self.img_label.shape[2]
        else:
            label = cv2.filter2D(self.img_label, -1, kernel).astype(np.float64)
            result = cv2.filter2D(self.img_result, -1, kernel).astype(np.float64)
            return self._cc(label, result)
        pass

    # 4. universal image quality index
    def cal_uiqi(self):
        def _uiqi(label, result):
            label_var = ((label - label.mean()) ** 2).sum() / (label.shape[0] * label.shape[1] - 1)
            result_var = ((result - result.mean()) ** 2).sum() / (result.shape[0] * result.shape[1] - 1)

            cov = ((result - result.mean()) * (label - label.mean())).sum() / (result.shape[0] * result.shape[1] - 1)

            up = 4 * cov * label.mean() * result.mean()
            down = (label_var + result_var) * (label.mean() ** 2 + result.mean() ** 2)

            return up / down

        if self.multi:
            res = 0
            for i in range(self.label_np.shape[2]):
                res += _uiqi(self.label_np[:, :, i], self.result_np[:, :, i])
            return res / self.label_np.shape[2]
        else:
            return _uiqi(self.label_np, self.result_np)
        pass

    # cal cc for single image
    def _rmse(self, label, result):
        return ((label - result) ** 2).mean()

    # 5. Root Mean Square Error
    def cal_rmse(self):

        if self.multi:
            res = 0
            for i in range(self.label_np.shape[2]):
                res += self._rmse(self.label_np[:, :, i], self.result_np[:, :, i])
            return res / self.label_np.shape[2]
        else:
            return self._rmse(self.label_np, self.result_np)

    # 6.erreur relative globale adimensionnelle de synth`ese
    def cal_ergas(self):
        def _ergas(label, result):
            up = self._rmse(label, result)
            down = label.mean()
            return (up / down) ** 2

        if self.multi:
            res = 0
            for i in range(self.label_np.shape[2]):
                res += _ergas(self.label_np[:, :, i], self.result_np[:, :, i])
            return 100 * (self.label_np.shape[0] / self.label_np.shape[1]) * np.sqrt(res / self.label_np.shape[2])
        else:
            return 100 * (self.label_np.shape[0] / self.label_np.shape[1]) * np.sqrt(
                _ergas(self.label_np, self.result_np))
        pass

    # 7. peak signal-to-noise ratio
    def cal_psnr(self):
        return peak_signal_noise_ratio(self.label_np, self.result_np, data_range=255)

    # 8. structural similarity index
    def cal_ssim(self):
        return structural_similarity(self.label_np, self.result_np, multichannel=self.multi, data_range=255)


def cal_all_fusioneval_r(img_label, img_result) -> list:
    fusioneval_r = FusionEval_R(img_label, img_result)
    return [fusioneval_r.cal_sam(), fusioneval_r.cal_cc(), fusioneval_r.cal_scc(), fusioneval_r.cal_uiqi(),
            fusioneval_r.cal_rmse(), fusioneval_r.cal_ergas()]


def cal_all_fusioneval_nr(img_fused, img_pan, img_ms):
    fusioneval_nr = FusionEval_NR(img_fused, img_pan, img_ms)
    return fusioneval_nr.QNR()


def cal_all_nriq(img_gray) -> list:
    nriq = NRIQ(img_gray)
    return [nriq.Tenengrad(), nriq.Brenner(), nriq.Laplacian(), nriq.SMD(), nriq.SMD2(), nriq.Variance(), nriq.Energy(),
            nriq.Vollath(), nriq.Entropy(), nriq.NRSS()]


class FusionEval_NR():
    def __init__(self, img_fused, img_pan, img_ms):
        assert len(img_fused.shape) > 2, 'Fusion Evaluation: fused image shapes should be 3 Dimension!'
        assert len(img_ms.shape) > 2, 'Fusion Evaluation: ms image shapes should be 3 Dimension!'
        assert img_fused.shape[2] == img_ms.shape[2], 'Fusion Evaluation: fused and ms image shapes do not match!'

        # original image uint8
        self.img_fused = img_fused
        self.img_pan = img_pan
        self.img_ms = img_ms

        # new image float64
        self.fused_np = img_fused.astype(np.float64)
        self.pan_np = img_pan.astype(np.float64)
        self.ms_np = img_ms.astype(np.float64)
        self.pan_lr_np = cv2.resize(self.img_pan, (self.img_ms.shape[0], self.img_ms.shape[1]), cv2.INTER_CUBIC)
        self.channel = self.fused_np.shape[2]
        pass

    def _ssim(self, img_a, img_b, multi=False):
        return structural_similarity(img_a, img_b, multichannel=multi, data_range=255)

    # 1. spectral distortion
    def D_lamda(self):
        res = 0
        for i in range(self.channel):
            for j in range(self.channel):
                res += np.abs(self._ssim(self.fused_np[:, :, i],
                                         self.fused_np[:, :, j] - self._ssim(self.ms_np[:, :, i], self.ms_np[:, :, j])))
        res *= 1 / (self.channel * (self.channel - 1))
        return res

    # 2. spatial distortion
    def D_s(self):
        res = 0
        for i in range(self.channel):
            res += np.abs(
                self._ssim(self.fused_np[:, :, i], self.pan_np) - self._ssim(self.ms_np[:, :, i], self.pan_lr_np))
        res *= 1 / (self.channel * (self.channel - 1))
        return res

    # 3. Quality with no reference
    def QNR(self):
        return (1 - self.D_lamda()) * (1 - self.D_s())


class NRIQ():
    def __init__(self, img, avg=True):
        self.img = img

        self.img_np = img.astype(np.float64)
        self.avg = avg

    def _imageToMatrix(self, img):
        """
        根据名称读取图片对象转化矩阵
        :param strName:
        :return: 返回矩阵
        """
        imgMat = np.matrix(img)
        return imgMat

    def _sobel(self, img):
        '''
        sobel算子
        :param img:
        :return:
        '''
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        return cv2.addWeighted(x, 0.5, y, 0.5, 0)

    def Tenengrad(self):
        '''
        平均 Tenengrad 梯度函数
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        x = cv2.Sobel(self.img_np, cv2.CV_64F, 1, 0)
        y = cv2.Sobel(self.img_np, cv2.CV_64F, 0, 1)
        source = np.sum(np.sqrt(x ** 2 + y ** 2))
        if self.avg:
            source /= (self.img_np.shape[0] * self.img_np.shape[1])
        return source

    def Brenner(self):
        '''
        平均Brenner 梯度函数
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        shape = np.shape(self.img_np)
        out = 0
        for x in range(0, shape[0] - 2):
            for y in range(0, shape[1]):
                out += (int(self.img_np[x + 2, y]) - int(self.img_np[x, y])) ** 2
        if self.avg:
            out /= ((shape[0] - 2) * shape[1])
        return out

    def Laplacian(self):
        '''
        Laplacian 梯度函数
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        return cv2.Laplacian(self.img_np, cv2.CV_64F).var()

    def SMD(self):
        '''
        SMD（灰度方差）
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        shape = np.shape(self.img_np)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(1, shape[1]):
                out += math.fabs(int(self.img_np[x, y]) - int(self.img_np[x, y - 1]))
                out += math.fabs(int(self.img_np[x, y] - int(self.img_np[x + 1, y])))
        if self.avg:
            out /= ((shape[0] - 1) * (shape[1] - 1))
        return out

    def SMD2(self):
        '''
        SMD2（灰度方差乘积）
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        shape = np.shape(self.img_np)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += math.fabs(int(self.img_np[x, y]) - int(self.img_np[x + 1, y])) * math.fabs(
                    int(self.img_np[x, y] - int(self.img_np[x, y + 1])))
        if self.avg:
            out /= ((shape[0] - 1) * (shape[1] - 1))
        return out

    def Variance(self):
        '''
        方差函数
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        return self.img_np.var()

    def Energy(self):
        '''
        能量梯度函数
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        shape = np.shape(self.img_np)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += ((int(self.img_np[x + 1, y]) - int(self.img_np[x, y])) ** 2) + (
                        (int(self.img_np[x, y + 1] - int(self.img_np[x, y]))) ** 2)
        if self.avg:
            out /= ((shape[0] - 1) * (shape[1] - 1))
        return out

    def Vollath(self):
        '''
        Vollath函数
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        shape = np.shape(self.img_np)
        u = np.mean(self.img_np)
        out = -shape[0] * shape[1] * (u ** 2)
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1]):
                out += int(self.img_np[x, y]) * int(self.img_np[x + 1, y])
        if self.avg:
            out /= ((shape[0] - 1) * (shape[1]))
        return out

    def Entropy(self):
        '''
        熵函数
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        out = 0
        count = np.shape(self.img_np)[0] * np.shape(self.img_np)[1]
        p = np.bincount(np.array(self.img_np.astype(np.uint8)).flatten())
        for i in range(0, len(p)):
            if p[i] != 0:
                out -= p[i] * math.log(p[i] / count) / count
        return out

    def NRSS(self):
        Ir = cv2.GaussianBlur(self.img_np, (7, 7), 0)
        G = self._sobel(self.img_np)
        Gr = self._sobel(Ir)
        (h, w) = G.shape
        G_blk_list = []
        Gr_blk_list = []
        sp = 6
        for i in range(sp):
            for j in range(sp):
                G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
                Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h), int((j / sp) * w):int(((j + 1) / sp) * w)]
                G_blk_list.append(G_blk)
                Gr_blk_list.append(Gr_blk)
        sum = 0
        for i in range(sp * sp):
            mssim = structural_similarity(G_blk_list[i], Gr_blk_list[i], data_range=255)
            sum = mssim + sum
        nrss = 1 - sum / (sp * sp * 1.0)
        return nrss
