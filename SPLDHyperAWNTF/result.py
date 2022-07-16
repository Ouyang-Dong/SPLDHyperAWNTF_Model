import numpy as np
from data import MDAv2_GetData
from method import model
import tensorly as tl
import pandas as pd
import math

class Experiments(object):
    def __init__(self, mir_dis_data, model_name='SPLDHyperNTF', **kwargs):
        super().__init__()
        self.mir_dis_data = mir_dis_data
        self.model = model.Hyper_Model(model_name)
        self.parameters = kwargs

    def result(self):

        association_matrix = self.mir_dis_data.type_tensor.sum(2)
        index_matrix = np.array(np.where(association_matrix > 0))


        train_matrix = self.mir_dis_data.type_tensor.sum(2)
        train_matrix[np.where(train_matrix > 0)] = 1

        miRNA_func_similarity_matrix = np.mat(self.mir_dis_data.mi_sim)
        dis_sen_similarity_matrix = np.mat(self.mir_dis_data.dis_sim)

        nd = train_matrix.shape[1]
        nm = train_matrix.shape[0]

        rd = np.zeros([nd, 1])
        rm = np.zeros([nm, 1])

        for i in range(nd):
            rd[i] = math.pow(np.linalg.norm(train_matrix[:, i]), 2)
        gamad = nd / rd.sum()

        for j in range(nm):
            rm[j] = math.pow(np.linalg.norm(train_matrix[j, :]), 2)
        gamam = nm / rm.sum()

        DGSM = np.zeros([nd, nd])
        for m in range(nd):
            for n in range(nd):
                DGSM[m, n] = np.exp(
                    -gamad * math.pow(np.linalg.norm(train_matrix[:, m] - train_matrix[:, n]), 2))

        MGSM = np.zeros([nm, nm])
        for r in range(nm):
            for t in range(nm):
                MGSM[r, t] = np.exp(
                    -gamam * math.pow(np.linalg.norm(train_matrix[r, :] - train_matrix[t, :]), 2))

        ID = np.zeros([nd, nd])

        for h1 in range(nd):
            for h2 in range(nd):
                if dis_sen_similarity_matrix[h1, h2] == 0:
                    ID[h1, h2] = DGSM[h1, h2]
                else:
                    ID[h1, h2] = dis_sen_similarity_matrix[h1, h2]

        IM = np.zeros([nm, nm])

        for q1 in range(nm):
            for q2 in range(nm):
                if miRNA_func_similarity_matrix[q1, q2] == 0:
                    IM[q1, q2] = MGSM[q1, q2]
                else:
                    IM[q1, q2] = miRNA_func_similarity_matrix[q1, q2]

        # new_IM = pd.DataFrame(IM)
        # new_IM.to_csv('./IM.csv')


        temp_association_matrix = self.mir_dis_data.type_tensor.sum(2)
        temp_association_matrix[np.where(temp_association_matrix > 0)] = 1

        concat_miRNA = np.mat(np.hstack([temp_association_matrix, IM]))
        concat_disease = np.mat(np.hstack([temp_association_matrix.T, ID]))
        mi_num, dis_num, type_num = self.mir_dis_data.type_tensor.shape

        '自步权重'
        W = tl.tensor(np.ones(mi_num * dis_num * type_num).reshape(mi_num, dis_num, type_num))

        predict_tensor = self.model()(self.mir_dis_data.type_tensor, concat_miRNA, concat_disease, W,
                                      r=self.parameters['r'], alpha=self.parameters['alpha'],
                                      beta=self.parameters['beta'],
                                      lam_t=self.parameters['lam_t'], lam_c=self.parameters['lam_c'],
                                      tol=1e-5, max_iter=2000)


        miRNA_name = pd.read_csv('D:\\python_PhD\\python_research\\Fourth_paper\\HMDD_data\\MDAv2.0\\mi_name_2.0.csv',index_col=0)
        dis_name = pd.read_csv('D:\\python_PhD\\python_research\\Fourth_paper\\HMDD_data\\MDAv2.0\\dis_name_2.0.csv',index_col=0)

        mi_dis_list = []
        for m in range(index_matrix.shape[1]):
            mi_dis_list.append([miRNA_name.iloc[index_matrix[0][m],:].values[0],dis_name.iloc[index_matrix[1][m],:].values[0]])

        mi_dis_list_dataframe = pd.DataFrame(mi_dis_list)
        mi_dis_list_dataframe.to_csv('D:/python_PhD/mi_dis_list_2.0.csv')

        sample_num_eval = np.array(index_matrix).shape[1]

        type_dic = {0: 'target', 1: 'circulation', 2: 'epigenetics', 3: 'genetics'}
        type_list = []
        max_score = []
        for t in range(sample_num_eval):
            predict_matrix = predict_tensor[index_matrix[0][t], index_matrix[1][t]]
            predict_score = np.mat(predict_matrix.flatten())
            # print(predict_score, file = pre_list)


            real_matrix = self.mir_dis_data.type_tensor[index_matrix[0][t], index_matrix[1][t]]
            real_score = np.mat(real_matrix.flatten())
            # print(real_score, file = real_list)
            sort_index = np.array(np.argsort(predict_score))[0]
            for n in range(len(list(type_dic.keys()))):
                if sort_index[-1:] == list(type_dic.keys())[n]:
                    type_list.append(list(type_dic.values())[n])

            max_score.append(predict_score[:, sort_index[-1:]].tolist()[0])
        max_score_dataframe = pd.DataFrame(max_score)
        max_score_dataframe.to_csv('D:/python_PhD/max_score.csv')
        # print(max_score_dataframe)
        type_list_dataframe = pd.DataFrame(type_list)
        type_list_dataframe.to_csv('D:/python_PhD/type_list.csv')
        # print(type_list_dataframe)





if __name__ == '__main__':
    root = 'D:/python_PhD/python_research/Fourth_paper/HMDD_data'
    mir_dis_data = MDAv2_GetData.MDAv2_GetData(root)
    experiment = Experiments(mir_dis_data, model_name='SPLDHyperNTF', r = 6, alpha = 0.0002, beta = 1, lam_t = 0.001, lam_c = 1, tol=1e-5,
                             max_iter=500)
    print(experiment.result())




