import numpy as np
import tensorly as tl
import ConstructHW
import math

class Hyper_Model(object):

    def __init__(self, name='SPLDHyperAWNTF'):
        super().__init__()
        self.name = name

    def SPLDHyperAWNTF(self, X, m_embeding, d_embeding, W, r=4, alpha = 2, beta = 2, lam_t = 0.001, lam_c = 0.3, max_iter=2000, tol=1e-5):


        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]


        np.random.seed(0)
        M = np.mat(np.random.rand(m, r))
        D = np.mat(np.random.rand(d, r))
        T = np.mat(np.random.rand(t, r))


        '''Hypergraph Learning'''
        Dv_m,S_m = ConstructHW.constructHW(m_embeding)
        Dv_d,S_d = ConstructHW.constructHW(d_embeding)


        X_1 = np.mat(tl.unfold(X, 0))
        X_2 = np.mat(tl.unfold(X, 1))
        X_3 = np.mat(tl.unfold(X, 2))

        '''adaptive weight tensor'''
        np.random.seed(1)
        C_1 = np.mat(np.random.rand(X_1.shape[0],X_1.shape[1]))
        C_2 = np.mat(np.random.rand(X_2.shape[0],X_2.shape[1]))
        C_3 = np.mat(np.random.rand(X_3.shape[0],X_3.shape[1]))


        I_1 = np.mat(np.ones(X_1.shape))
        I_2 = np.mat(np.ones(X_2.shape))
        I_3 = np.mat(np.ones(X_3.shape))


        W_1 = np.mat(tl.unfold(W, 0))
        W_2 = np.mat(tl.unfold(W, 1))
        W_3 = np.mat(tl.unfold(W, 2))

        k = 1.2
        k_end = 0.008
        gama = 1.2

        while k > k_end:
            for i in range(max_iter):
                A = np.mat(tl.tenalg.khatri_rao([D, T]))
                output_X_old = tl.fold(np.array(M * A.T), 0, X.shape)

                '''Updating the factor matrix M'''
                new_C_1 = np.multiply((I_1 - X_1),C_1)
                Fin_C_1 = np.multiply(W_1,new_C_1)
                new_X_1 = np.multiply(W_1, X_1)

                temp_M_m = new_X_1 * A + Fin_C_1 * A + alpha * S_m * M
                temp_M_d = np.multiply(W_1, (M * A.T)) * A + alpha * Dv_m * M + lam_t * M
                temp_M = temp_M_m / (temp_M_d + 1e-8)
                M = np.multiply(M,temp_M)

                '''Updating the factor matrix D'''
                new_C_2 = np.multiply((I_2 - X_2),C_2)
                Fin_C_2 = np.multiply(W_2,new_C_2)
                new_X_2 = np.multiply(W_2, X_2)

                B = np.mat(tl.tenalg.khatri_rao([M, T]))
                temp_D_m = new_X_2 * B + Fin_C_2 * B + beta * S_d * D
                temp_D_d = np.multiply(W_2, (D * B.T)) * B + beta * Dv_d * D + lam_t * D
                temp_D = temp_D_m / (temp_D_d + 1e-8)
                D = np.multiply(D,temp_D)

                '''Updating the factor matrix T'''
                new_C_3 = np.multiply((I_3 - X_3), C_3)
                Fin_C_3 = np.multiply(W_3, new_C_3)
                new_X_3 = np.multiply(W_3, X_3)

                E = np.mat(tl.tenalg.khatri_rao([M, D]))
                temp_T_m = new_X_3 * E + Fin_C_3 * E
                temp_T_d = np.multiply(W_3, (T * E.T)) * E + lam_t * T
                temp_T = temp_T_m / (temp_T_d + 1e-8)
                T = np.multiply(T,temp_T)


                '''Updating the adaptive weight tensor'''
                #Updating C_1
                X_1_err = X_1 - (M * A.T)
                X_1_bar = I_1 - X_1
                temp_C_1_m = - np.multiply(np.multiply(W_1, X_1_err), X_1_bar)
                temp_C_1_d = np.multiply(W_1, np.multiply(X_1_bar,X_1_bar)) + lam_c
                C_1 = temp_C_1_m / temp_C_1_d

                #Updating C_2
                X_2_err = X_2 - (D * B.T)
                X_2_bar = I_2 - X_2
                temp_C_2_m = - np.multiply(np.multiply(W_2, X_2_err), X_2_bar)
                temp_C_2_d = np.multiply(W_2, np.multiply(X_2_bar,X_2_bar)) + lam_c
                C_2 = temp_C_2_m / temp_C_2_d

                #Updating C_3
                X_3_err = X_3 - (T * E.T)
                X_3_bar = I_3 - X_3
                temp_C_3_m = - np.multiply(np.multiply(W_3, X_3_err), X_3_bar)
                temp_C_3_d = np.multiply(W_3, np.multiply(X_3_bar,X_3_bar)) + lam_c
                C_3 = temp_C_3_m / temp_C_3_d


                output_X = tl.fold(np.array(T * E.T), 2, X.shape)
                err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)

                if err < tol:
                    break

            '''reconstructed tensor'''
            output_X = tl.fold(np.array(T * E.T), 2, X.shape)
            New_X_1 = np.mat(tl.unfold(output_X, 0))
            New_X_2 = np.mat(tl.unfold(output_X, 1))
            New_X_3 = np.mat(tl.unfold(output_X, 2))
            for I in range(New_X_1.shape[0]):
                for S in range(New_X_1.shape[1]):
                    loss1 = math.pow((X_1[I, S] - New_X_1[I, S] + (I_1[I,S] - X_1[I, S]) * C_1[I,S]),2)
                    if (loss1 <= 1 / math.pow((k + 1 / gama), 2)):
                        W_1[I, S] = 1
                    elif (loss1 >= 1 / math.pow(k, 2)):
                        W_1[I, S] = 0
                    else:
                        W_1[I, S] = gama * (1 / np.sqrt(loss1) - k)

            for O in range(New_X_2.shape[0]):
                for N in range(New_X_2.shape[1]):
                    loss2 = math.pow((X_2[O, N] - New_X_2[O, N] + (I_2[O, N] - X_2[O, N]) * C_2[O, N]), 2)
                    if (loss2 <= 1 / math.pow((k + 1 / gama), 2)):
                        W_2[O, N] = 1
                    elif (loss2 >= 1 / math.pow(k, 2)):
                        W_2[O, N] = 0
                    else:
                        W_2[O, N] = gama * (1 / np.sqrt(loss2) - k)

            for K in range(New_X_3.shape[0]):
                for L in range(New_X_3.shape[1]):
                    loss3 = math.pow((X_3[K, L] - New_X_3[K, L] + (I_3[K, L] - X_3[K, L]) * C_3[K, L]), 2)
                    if (loss3 <= 1 / math.pow((k + 1 / gama), 2)):
                        W_3[K, L] = 1
                    elif (loss3 >= 1 / math.pow(k, 2)):
                        W_3[K, L] = 0
                    else:
                        W_3[K, L] = gama * (1 / np.sqrt(loss3) - k)
            k = k / 1.2

        predict_X = np.array(tl.fold(np.array(M * np.mat(tl.tenalg.khatri_rao([D, T])).T), 0, X.shape))

        return predict_X

    def __call__(self):

        return getattr(self, self.name, None)
