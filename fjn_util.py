import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


# -------------------------- img index --------------------------

def check_img_data_range(img):
    if img.dtype == np.uint8:
        return 255
    else:
        return 1.0


def psnr(img1, img2):
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    return peak_signal_noise_ratio(img1, img2, data_range=check_img_data_range(img1))


def ssim(img1, img2):
    '''
    :param img1: image 1
    :param img2:  image 2
    :return: ssim between image1 and image2
    '''
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    return structural_similarity(img1, img2, multichannel=(len(img1.shape) == 3), data_range=check_img_data_range(img1))


def mse(img1, img2):
    '''
    :param img1: image 1
    :param img2:  image 2
    :return: mse between image1 and image2
    '''
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    if check_img_data_range(img1) == 1.0:
        img1, img2 = (img1 * 255.).astype(np.uint8), (img2 * 255.).astype(np.uint8)
    return mean_squared_error(img1, img2)


def mae(img1, img2):
    '''
    :param img1: image 1
    :param img2:  image 2
    :return: mae between image1 and image2
    '''
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    if check_img_data_range(img1) == 1.0:
        img1, img2 = (img1 * 255.).astype(np.uint8), (img2 * 255.).astype(np.uint8)
    return np.mean(abs(img1 - img2))


def pearsonr_corr(img1, img2):
    '''
    :param img1: image 1
    :param img2:  image 2
    :return: pearsonr between image1 and image2
    '''

    # 1. check img's dtype and shape
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'

    # 2. def pearsonr for 2D shape list
    def cal_pear(img1, img2):
        result = 0
        for i in range(img1.shape[0]):
            pear = pearsonr(img1[i], img2[i])[0]
            if np.isnan(pear):
                result += 0
            else:
                result += pear
        return result

    # 3. select single channel or multi-channel
    if len(img1.shape) == 2:
        return cal_pear(img1, img2) / img1.shape[0]
    else:
        result = 0
        img1s = np.array_split(img1, 3, axis=2)
        img2s = np.array_split(img2, 3, axis=2)
        for i in range(len(img1s)):
            result += cal_pear(np.squeeze(img1s[i], 2), np.squeeze(img2s[i], 2))
        return result / img1s[0].shape[0] / 3


def cal_all_index(img1, img2) -> list:
    '''
    :param img1:
    :param img2:
    :return: [psnr, ssim, mse, mae, pearsonr]
    '''
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    return [psnr(img1, img2), ssim(img1, img2), mse(img1, img2), mae(img1, img2), pearsonr_corr(img1, img2)]


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AverageCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class AccuracyCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.accu = 0
        self.sum = 0
        self.correct = 0

    def update(self, val):
        if val:
            self.correct += 1
        self.sum += 1
        self.accu = self.correct / self.sum


class Model_Statistics():

    def __init__(self, statistics_path):

        self.train_accuracy_list = []
        self.test_accuracy_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_psnr_list = []
        self.train_ssim_list = []
        self.val_ssim_list = []

        self.txt_path = statistics_path + 'txt/'
        self.graph_path = statistics_path + 'graph/'

        self.all_count = {
            'train_accuracy': self.train_accuracy_list,
            'test_accuracy': self.test_accuracy_list,
            'train_loss': self.train_loss_list,
            'val_loss': self.val_loss_list,
            'val_psnr': self.val_psnr_list,
            'train_ssim': self.train_ssim_list,
            'val_ssim': self.val_ssim_list,
        }

        self.train_accuracy_count = AccuracyCounter()
        self.test_accuracy_count = AccuracyCounter()
        self.train_loss_count = AverageCounter()
        self.test_loss_count = AverageCounter()
        self.train_psnr_count = AverageCounter()
        self.train_ssim_count = AverageCounter()

        make_folder(self.txt_path)
        make_folder(self.graph_path)

        self.read_from_txt()

    # append new value to count_list
    def update(self, name, val):
        assert self.all_count.__contains__(name), 'MyAssert: not found ' + name + ' in all_count'
        self.all_count[name].append(val)

    def draw_accuracy(self, saved=False):
        plt.plot(list(range(len(self.train_accuracy_list))), self.all_count['train_accuracy'], linestyle="-", marker="",
                 linewidth=2, label='train_accuracy', c='blue')
        plt.plot(list(range(len(self.test_accuracy_list))), self.all_count['test_accuracy'], linestyle="-", marker="",
                 linewidth=2, label='test_accuracy', c='red')
        plt.title('accuracy')
        if saved:
            make_folder(self.graph_path)
            plt.savefig(self.graph_path + 'accuracy.png')
        plt.show()

    def draw_loss(self, saved=False):
        plt.plot(list(range(1, len(self.train_loss_list) + 1)), self.all_count['train_loss'], linestyle="-", marker="",
                 linewidth=2, label='train_loss', c='blue')
        plt.title('loss')
        if saved:
            make_folder(self.graph_path)
            plt.savefig(self.graph_path + 'loss.png')
        # plt.show()

    # write all records to the txt
    def write_all_to_txt(self):
        for name in self.all_count:
            file = open(os.path.join(self.txt_path, name + '.txt'), 'w')
            for item in self.all_count[name]:
                file.write(str(item) + '\n')

    # read records from txt_path
    def read_from_txt(self):
        for name in self.all_count:
            if (os.path.exists(os.path.join(self.txt_path, name + '.txt'))):
                for line in open(os.path.join(self.txt_path, name + '.txt'), "r"):
                    self.all_count[name].append(float(line[:-1]))


def model_summary(model, input_size, batch_size=-1, device="cuda", show_detail=False):
    '''
    :param model:
    :param input_size:
    :param batch_size:
    :param device:
    :param show_detail:
    :return: [
        total params of model
        input size (related with batchsize)                     MB
        output size (related with batchsize and input size)     MB
        params size of model                                    MB
     ]
    '''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if isinstance(input_size, tuple):
        input_size = [input_size]
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    summary = OrderedDict()
    hooks = []
    model.apply(register_hook)
    model(*x)
    for h in hooks:
        h.remove()
    if show_detail:
        print("----------------------------------------------------------------")
        print("{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #"))
        print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]), )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if show_detail:
            print(line_new)
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    params_list = [total_params.numpy().tolist(), total_input_size, total_output_size, total_params_size]
    if show_detail:
        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
    return params_list


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
