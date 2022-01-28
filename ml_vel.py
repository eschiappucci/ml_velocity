#Imports
from utils import *

if __name__ == '__main__':
    #Reading the catalogue
    filename = 'spt3g_yuuki_150GHz_m500c_0.6-5e14_MF_beta1_thetac05_noise5.dat'

    decmin = -65
    decmax = -40
    zmin = 0.1
    zmax = 0.8
    richmin= 0.6e14*h
    richmax= None #4e14*h
    photoz = None #0.01
    nobj = None #25_000
    seed = None #100
    sigmaM = None #0.3

    RA, DEC, com_dists, mass, TSZ, Z, vlos = readdata(filename, DECmin=decmin, DECmax=decmax, Zmin=zmin, Zmax=zmax, richmin=richmin, richmax=richmax, photoz=photoz, sigmaM=sigmaM, nobj=nobj, seed=seed)

    n_neighbours = 5
    class_mat, reg_vec, class_vec = build_tf_matrix_vec(RA, DEC, Z, com_dists, mass, vlos, close_pairs = n_neighbours)
    reg_mat, _, _ = build_tf_matrix(RA, DEC, Z, com_dists, mass, vlos, close_pairs = n_neighbours)

    class_mat_train, class_mat_test, reg_mat_train, reg_mat_test, reg_vec_train, reg_vec_test, class_vec_train, class_vec_test = array_split(class_mat, reg_mat, reg_vec, class_vec, test_size=0.25)
    model_class = tf_classification(class_mat_train)
    history = model_class.fit(class_mat_train, class_vec_train, epochs=50, verbose=0)
    model_class.evaluate(class_mat_test, class_vec_test, verbose=1)

    plot_metric(history, metric='accuracy')

    #reg_mat_train = np.append(reg_mat_train, np.array([class_vec_train]).T, axis=-1)
    #reg_mat_test = np.append(reg_mat_test, np.array([class_vec_test]).T, axis=-1)
    model_reg = tf_regression(class_mat_train)
    history = model_reg.fit(class_mat_train, reg_vec_train, epochs=50, verbose=0)
    model_reg.evaluate(class_mat_test, reg_vec_test, verbose=1)

    plot_metric(history, metric='loss')

    print(np.mean(reg_vec), max(reg_vec), min(reg_vec), len(np.where(reg_vec==0)[0]))
    print(np.mean(class_vec), len(class_vec))

