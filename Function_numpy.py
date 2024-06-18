import numpy as np
from Load import *
import numpy.matlib
import scipy.io
class GetInst_A(object):
    def __init__(self, useful_sp_lab, img3d, gt, trpos, Z):
        self.useful_sp_lab = useful_sp_lab
        self.img3d = img3d
        [self.r, self.c, self.l] = np.shape(img3d)
        self.num_classes = int(np.max(gt))
        self.img2d = np.reshape(img3d,[self.r*self.c, self.l])
        self.sp_num = np.array(np.max(self.useful_sp_lab), dtype='int')
        gt = np.array(gt, dtype='int')
        self.gt1d = np.reshape(gt, [self.r*self.c])
        self.gt_tr = np.array(np.zeros([self.r*self.c]), dtype='int')
        self.gt_te = self.gt1d
        trpos = np.array(trpos, dtype='int')
        self.trpos = (trpos[:,0]-1)*self.c+trpos[:,1]-1  # 得到半监督的部分数据索引
        ###
        self.sp_mean = np.zeros([self.sp_num, self.l])
        self.sp_center_px = np.zeros([self.sp_num, self.l]) 
        self.sp_label = np.zeros([self.sp_num]) 
        self.trmask = np.zeros([self.sp_num])
        self.temask = np.ones([self.sp_num])
        self.sp_A = Z
        self.l1 = []
        self.support = self.l1
        self.CalSpMean()    
        # self.CalSpA()
    def CalSpMean(self):            #计算区域均值
        self.gt_tr[self.trpos] = self.gt1d[self.trpos] # 把抽样数据的类别保存
        mark_mat = np.zeros([self.r*self.c])
        mark_mat[self.trpos] = -1
        for sp_idx in range(1, self.sp_num+1):
            region_pos_2d = np.argwhere(self.useful_sp_lab == sp_idx) #2d的位置
            region_pos_1d = region_pos_2d[:, 0]*self.c + region_pos_2d[:, 1]#一维的位置
            px_num = np.shape(region_pos_2d)[0] # 返回超像素分割中的分割区域的像素个数
            if np.sum(mark_mat[region_pos_1d])<0: # 如果这个区域中有抽样数据
                self.trmask[sp_idx-1] = 1
                self.temask[sp_idx-1] = 0
            region_fea = self.img2d[region_pos_1d, :] # 得到二位图片中这个区域特征
            if self.trmask[sp_idx-1] == 1: # 如果这个区域中有抽样数据 则给这个区域设置标签
                region_labels = self.gt_tr[region_pos_1d]
            else:
                region_labels = self.gt_te[region_pos_1d]
            self.sp_label[sp_idx-1] = np.argmax(np.delete(np.bincount(region_labels), 0))+1 # 为分割区域赋值标签
            region_pos_idx = np.argwhere(region_labels == self.sp_label[sp_idx-1])
            pos1 = region_pos_1d[region_pos_idx]
            self.sp_rps = np.mean(self.img2d[pos1, :], axis = 0) #取均值
            vj = np.sum(np.power(np.matlib.repmat(self.sp_rps, px_num, 1)-region_fea, 2), axis=1)
            vj= np.exp(-0.2*vj)
            self.sp_mean[sp_idx-1, :] = np.sum(np.reshape(vj, [np.size(vj), 1])*region_fea, axis=0)/np.sum(vj)
        sp_label_mat = np.zeros([self.sp_num, self.num_classes])
        for row_idx in range(np.shape(self.sp_label)[0]):
            col_idx = int(self.sp_label[row_idx])-1
            sp_label_mat[row_idx, col_idx] = 1
        self.sp_label_vec = self.sp_label
        self.sp_label = sp_label_mat # 区域类别
        scio.savemat('sp_label_mat.mat', {'sp_label_mat': sp_label_mat})
        scio.savemat('sp_mean.mat', {'sp_mean': self.sp_mean})
        scio.savemat('gt_tr.mat', {'gt_tr': self.gt_tr})
    # def CalSpA(self):
    #     data_name = 'IP'
    #     Data = load_HSI_data(data_name)
    #     self.sp_A = Data['Z']
    def CalSupport(self, A):  #归一化矩阵A
        num1 = np.shape(A)[0]
        A_ = A + np.eye(num1)
        D_ = np.sum(A_, 1)
        D_05 = np.diag(D_**(-0.5))
        support = np.matmul(np.matmul(D_05, A_), D_05)
        return support
