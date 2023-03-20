import numpy as np
import torch.nn as nn

from src.model.MFF import *
from src.model.resnet12 import resnet12
from pyts.image import RecurrencePlot, MarkovTransitionField, GramianAngularField

def convertRP(data, img_size):
    rp = RecurrencePlot(threshold='point', percentage=20)
    # data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    rpData = rp.fit_transform(data)
    resized_result = []
    for i in range(0,rpData.shape[0]):
        if(len(resized_result) == 0):
            resized_result = cv2.resize(rpData[i],dsize=(img_size,img_size),interpolation=cv2.INTER_AREA)
            resized_result = resized_result[np.newaxis, ...]
        else:
            a = cv2.resize(rpData[i], dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
            a = a[np.newaxis, ...]
            resized_result = np.vstack((resized_result, a))
    return resized_result

def convertMTF(data):
    mtf = MarkovTransitionField(n_bins=8)
    # data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    mtfData = mtf.fit_transform(data)
    return mtfData

def convertGASF(data):
    gasf = GramianAngularField(method='summation')
    # data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    gasfData = gasf.fit_transform(data)
    return gasfData

def convertGADF(data):
    gadf = GramianAngularField(method='difference')
    # data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    gadfData = gadf.fit_transform(data)
    return gadfData


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model=resnet12()

        self.rp_ker=RP_kernel()
        self.mtf_ker=MTF_kernel()
        self.gadf_ker=GADF_kernel()
        self.gasf_ker = GASF_kernel()

    def forward(self, x):
        device = torch.device('cuda:0')

        x_RP = convertRP(x, 28)
        x_RP = x[:, np.newaxis, :, :]
        x_MTF = convertMTF(x, 28)
        x_MTF = x[:, np.newaxis, :, :]
        x_GASF = convertGASF(x, 28)
        x_GASF = x[:, np.newaxis, :, :]
        x_GADF = convertGADF(x, 28)
        x_GADF = x[:, np.newaxis, :, :]

        x_RP = torch.tensor(x_RP, dtype=torch.float32)
        x_MTF = torch.tensor(x_MTF, dtype=torch.float32)
        x_GASF = torch.tensor(x_GASF, dtype=torch.float32)
        x_GADF = torch.tensor(x_GADF, dtype=torch.float32)

        x_RP = x_RP.to(device)
        x_MTF = x_MTF.to(device)
        x_GASF = x_GASF.to(device)
        x_GADF = x_GADF.to(device)

        output_rp = self.model(x_RP)
        output_mtf = self.model(x_MTF)
        output_gasf = self.model(x_GASF)
        output_gadf = self.model(x_GADF)

        rp_corr = torch.mul(output_rp, torch.add(torch.add(output_mtf, output_gasf), output_gadf))
        mtf_corr = torch.mul(output_mtf, torch.add(torch.add(output_rp, output_gasf), output_gadf))
        gasf_corr = torch.mul(output_gasf, torch.add(torch.add(output_mtf, output_rp), output_gadf))
        gadf_corr = torch.mul(output_gadf, torch.add(torch.add(output_mtf, output_gasf), output_rp))

        refine_rp = torch.mul(output_rp,self.rp_ker(rp_corr))
        refine_mtf = torch.mul(output_mtf, self.mtf_ker(mtf_corr))
        refine_gasf = torch.mul(output_gasf, self.gasf_ker(gasf_corr))
        refine_gadf = torch.mul(output_gadf, self.gadf_ker(gadf_corr))

        feature = torch.mean((refine_rp,refine_mtf,refine_gasf,refine_gadf),dim=0)
        return feature



