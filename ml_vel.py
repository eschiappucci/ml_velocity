#Imports
from time import time
from utils import *

if __name__ == '__main__':
    #Reading the catalogue
    #filename = 'spt3g_yuuki_150GHz_m500c_0.6-5e14_MF_beta1_thetac05_noise5.dat'
    filename = 'catalog_fullsky_zsm1.fits.gz'
    decmin = -65
    decmax = -40
    Mmin=1e14
    Mmax=1e30
    zmin = 0.1
    zmax = 0.8
    richmin= 0.6e14*h
    richmax= None #4e14*h
    photoz = None #0.01
    nobj = None #25_000
    seed = None #100
    sigmaM = None #0.3

    #RA, DEC, com_dists, mass, TSZ, Z, vlos = readdata(filename, DECmin=decmin, DECmax=decmax, Zmin=zmin, Zmax=zmax, richmin=richmin, richmax=richmax, photoz=photoz, sigmaM=sigmaM, nobj=nobj, seed=seed)
    start = time()
    RA, DEC, Z, vlos, mass = fn_load_halo(filename, Mmin=Mmin, Mmax=Mmax, zmin=zmin, zmax=zmax)

    n_neighbours = 6
    reg_mat, reg_vec = build_tf_matrix_vec(RA, DEC, Z, mass, vlos, close_pairs = n_neighbours)

    print(time() - start)
    """
    plt.figure()
    plt.hist(reg_mat[:,0,0], bins=50)
    plt.title('RA')
    plt.show()

    plt.figure()
    plt.hist(reg_mat[:,0,1], bins=50)
    plt.title('DEC')
    plt.show()

    plt.figure()
    plt.hist(reg_mat[:,0,2], bins=50)
    plt.title('z')
    plt.show()

    plt.figure()
    plt.hist(reg_mat[:,0,3], bins=50)
    plt.title('mass')
    plt.show()

    plt.figure()
    plt.hist(reg_vec, bins=50)
    plt.title('v_los')
    plt.show()
    """
    print(reg_mat.shape)
    new_models = True
    retrain_num = 2
    reg_loc = 'Models/vl_reg.f5'
    reg_mat_train, reg_mat_test, reg_vec_train, reg_vec_test = array_split(reg_mat, reg_vec, test_size=0.3, random_state=986178946)

    if new_models:
       start = time()
       for i in range(retrain_num):
            rate = 1e-3
            if i == 0:
                model_reg = tf_regression(reg_mat_train, rate=rate)
                history = model_reg.fit(reg_mat_train, reg_vec_train, epochs=100, verbose=0, shuffle=True)
                model_reg.evaluate(reg_mat_test, reg_vec_test, verbose=1)
                model_reg.save(reg_loc)
            else:
                rate /= 10
                model_reg = recompile(reg_loc, rate=rate)
                history = model_reg.fit(reg_mat_train, reg_vec_train, epochs=100, verbose=0, shuffle=True)
                model_reg.evaluate(reg_mat_test, reg_vec_test, verbose=1)
                model_reg.save(reg_loc)

            period = time() - start
            print(f'It took {period/60} minutes.')
       plot_metric(history, metric='mean_absolute_percentage_error')

    else:
        model_reg = recompile(reg_loc, rate=1e-7)
        history = model_reg.fit(reg_mat_train, reg_vec_train, epochs=100, verbose=0, shuffle=True)
        model_reg.evaluate(reg_mat_test, reg_vec_test, verbose=1)
        model_reg.save(reg_loc)
        plot_metric(history, metric='mean_absolute_percentage_error')

    print(np.mean(reg_vec), max(reg_vec), min(reg_vec), len(np.where(reg_vec==0)[0]))

