import csv
import os.path as osp
import numpy as np


class MDAv2_2_GetData(object):
    def __init__(self, root, miRNA_num=322, dis_num=151):
        super().__init__()
        self.root = osp.join(root, 'MDAv2.0_2')
        self.miRNA_num = miRNA_num
        self.dis_num = dis_num
        self.dis_sim, self.mi_sim, self.type_tensor = self.__get_data__()

    def __get_data__(self):

        type_name = ['target', 'circulation', 'epigenetics', 'genetics']
        type_association_matrix = np.zeros((self.miRNA_num, self.dis_num, 4))
        for i in range(4):
            with open(osp.join(self.root, '{}.csv'.format(type_name[i])), 'r') as type_:
                type_mat = csv.reader(type_)
                row = -1

                for line in type_mat:
                    if row >= 0:
                        col = -1
                        for association in line:
                            if col >= 0:
                                type_association_matrix[row, col, i] = eval(association)
                            col = col + 1
                    row = row + 1

        disease_similarity_mat = np.zeros((self.dis_num, self.dis_num))
        with open(osp.join(self.root, 'DSSM2.0_2.csv'), 'r') as dis_sim:
            sim_mat = csv.reader(dis_sim)
            row = -1

            for line in sim_mat:
                if row >= 0:
                    col = -1
                    for sim in line:
                        if col >= 0:
                            disease_similarity_mat[row, col] = eval(sim)
                        col = col + 1
                row = row + 1
        disease_similarity_mat = np.mat(disease_similarity_mat)

        mi_fun_sim_mat = np.zeros((self.miRNA_num, self.miRNA_num))
        with open(osp.join(self.root, 'mi_fun_sim_2.0_2.csv'), 'r') as mi_sim:
            mi_sim_mat = csv.reader(mi_sim)
            row_m = -1

            for line_m in mi_sim_mat:
                if row_m >= 0:
                    col_m = -1
                    for sim_m in line_m:
                        if col_m >= 0:
                            mi_fun_sim_mat[row_m, col_m] = eval(sim_m)
                        col_m = col_m + 1
                row_m = row_m + 1
        mi_fun_sim_mat = np.mat(mi_fun_sim_mat)
        return disease_similarity_mat, mi_fun_sim_mat, type_association_matrix

