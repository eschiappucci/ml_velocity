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

    """
    plt.figure()
    plt.hist(class_mat[:,0,0], bins=50)
    plt.title('RA')
    plt.show()

    plt.figure()
    plt.hist(class_mat[:,0,1], bins=50)
    plt.title('DEC')
    plt.show()

    plt.figure()
    plt.hist(class_mat[:,0,2], bins=50)
    plt.title('z')
    plt.show()

    plt.figure()
    plt.hist(class_mat[:,0,3], bins=50)
    plt.title('mass')
    plt.show()

    plt.figure()
    plt.hist(reg_vec, bins=50)
    plt.title('v_los')
    plt.show()


    """
    new_models = False

    class_loc = 'Models/vl_class.f5'
    reg_loc = 'Models/vl_reg.f5'
    class_mat_train, class_mat_test, reg_mat_train, reg_mat_test, reg_vec_train, reg_vec_test, class_vec_train, class_vec_test = array_split(class_mat, reg_mat, reg_vec, class_vec, test_size=0.25)

    if new_models:
        model_class = tf_classification(class_mat_train, rate=1e-3)
        history = model_class.fit(class_mat_train, class_vec_train, epochs=200, verbose=0, shuffle=True)
        model_class.evaluate(class_mat_test, class_vec_test, verbose=1)
        model_class.save(class_loc)
        plot_metric(history, metric='accuracy')
        model_reg = tf_regression(class_mat_train, rate=1e-3)
        history = model_reg.fit(class_mat_train, reg_vec_train, epochs=200, verbose=0, shuffle=True)
        model_reg.evaluate(class_mat_test, reg_vec_test, verbose=1)
        model_class.save(reg_loc)
        plot_metric(history, metric='loss')
    else:
        model_class = recompile(class_loc, rate=1e-7, loss_func='binary_crossentropy', metric='accuracy')
        model_reg = recompile(reg_loc, rate=1e-7)

        history = model_class.fit(class_mat_train, class_vec_train, epochs=100, verbose=0, shuffle=True)
        model_class.evaluate(class_mat_test, class_vec_test, verbose=1)
        model_class.save(class_loc)
        plot_metric(history, metric='accuracy')
        model_class.save(class_loc)
        history = model_reg.fit(class_mat_train, reg_vec_train, epochs=100, verbose=0, shuffle=True)
        model_reg.evaluate(class_mat_test, reg_vec_test, verbose=1)
        model_class.save(reg_loc)
        plot_metric(history, metric='mean_absolute_percentage_error')
        model_class.save(reg_loc)

    print(np.mean(reg_vec), max(reg_vec), min(reg_vec), len(np.where(reg_vec==0)[0]))
    print(np.mean(class_vec), len(class_vec))


