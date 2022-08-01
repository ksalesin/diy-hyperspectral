import io
import argparse
import itertools
import numpy as np
import scipy.optimize
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

import matlab.engine
eng = matlab.engine.start_matlab()
out = io.StringIO()
err = io.StringIO()

import tape_data
import sensor_data
import munsell_data
import color_utils
from experiment import *

# Seed rng
np.random.seed(1)

def basis(spectra, n):
    """ Generate orthogonal basis functions from spectra using PCA. """
    pca = PCA(n_components = n)
    pca.fit(spectra)
    # print(pca.explained_variance_ratio_)

    return pca.components_

def create_dataset(wavelengths, waveplates, n_samples = 1000, thetas = None, fixed_waveplates = False):
    """ Create a dataset representing all possible combinations of given parameters. 
    
        `waveplates` should be a list of Waveplate objects, not a numpy array.
        Assumes at most 3 layers of waveplates could be used in a stack.

        if `fixed_waveplates` is True, use only the exact permutation of waveplates given in
        the `waveplates` list (used to minimize shuffling of waveplates during measurement, which is
        time-consuming).
    """
    n_wv = len(wavelengths)
    n_wp = len(waveplates)
    wp_permutations = []

    if fixed_waveplates:
        wp_permutations.append(waveplates)
    else:
        for i in range(n_wp):
            wp_permutations += list(itertools.permutations(waveplates, r = i+1))

    n_perm = len(wp_permutations)

    # Create dataset of random variations of all parameters
    if thetas is not None:
        dataset = np.empty((len(thetas), n_samples), dtype = object)

        # Draw random samples for which combination of tapes to use
        u_perm = np.random.randint(0, n_perm, n_samples)

        # Draw random samples for alpha
        u_alpha = np.random.rand(n_samples, n_wp) * 180.

        for t in range(len(thetas)):
            idx = 0
            while idx < n_samples:
                if fixed_waveplates:
                    perm = wp_permutations[0]
                else:
                    perm = wp_permutations[u_perm[idx]]

                config = Config(u_alpha[idx, :len(perm)], thetas[t], perm)    

                dataset[t, idx] = config
                idx += 1
    
    else:
        dataset = np.empty(n_samples, dtype = object)

        total = 0
        idx = 0
        while idx < n_samples:
            # Draw random sample for which combination of tapes to use
            if fixed_waveplates:
                perm = waveplates
            else:
                u_perm = np.random.randint(0, n_perm, n_samples)
                perm = wp_permutations[u_perm[idx]]

            # Draw random sample for theta
            u_theta = np.random.rand(1) * 180.

            # Draw random samples for alpha
            u_alpha = np.random.rand(n_wp) * 180.

            config = Config(u_alpha[:len(perm)], u_theta[0], perm)

            # Check if transmission ratio is high enough (average above 20% per wavelength)
            if (np.sum(config.transmission(wavelengths)) / n_wv >= 0.20):
                dataset[idx] = config
                idx += 1

            # total += 1
        # print("accepted samples: " + str(idx) + "/" + str(total) + " ({:.2f}%)".format(idx / total * 100))

    return dataset

def choose_optimal_filters(dataset, wavelengths, n_coreset, n_meas = 0, pca_basis = None):
    """ Choose a set of optimal filters by a diversification metric:
        maximize the minimum distance between elements of the set in reduced-dimensional space.

        if n_meas == 0, assume dataset is a 1D array where every data point has a random theta.
        if n_meas != 0, assume dataset is a 2D array where every row has a fixed theta.
    """
    n_theta = dataset.shape[0]
    transmissions = np.array([config.transmission(wavelengths) for config in dataset.flatten()])

    # Compute basis functions from dataset
    if pca_basis is None:
        n_basis = 4
        pca = PCA(n_components = n_basis)
        pca.fit(transmissions)
        pca_basis = pca.components_
    else:
        n_basis = pca_basis.shape[0]

    # Project filter transmissions down to basis function coefficients
    filter_coefs, _, _, _ = np.linalg.lstsq(pca_basis.transpose(), transmissions.transpose(), rcond=None)

    # Run greedy algorithm to minimize the maximum intercluster distance:
    #    "Clustering to minimize the maximum intercluster distance." Teofilo F. Gonzalez. Theoretical Computer Science. 1985.
    #
    # Variation as described in:
    #    "Composable Core-sets for Diversity and Coverage Maximization." 
    #     Piotr Indyk, Sepideh Mahabadi, Mohammad Mahdian, and Vahab S. Mirrokni. 
    #     PODS '14: Proceedings of the 33rd ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems. 2014.
    if n_meas > 0:
        filter_coefs = filter_coefs.transpose().reshape(n_theta, -1, n_basis).transpose(0, 2, 1)  # n_theta x n_basis x n_samples
        pair_idx = list(itertools.combinations(list(range(n_theta)), 2))

        sample_idx = []

        # Choose initial sample with furthest intracluster distance
        init_idx = 0
        init_max_min_d = 0.

        for i in range(filter_coefs.shape[2]):
            min_d = float('inf')

            for pair in pair_idx:
                d = np.linalg.norm(filter_coefs[pair[0], :, i] - filter_coefs[pair[1], :, i])

                if d < min_d:
                    min_d = d

            if min_d > init_max_min_d:
                init_idx = i
                init_max_min_d = min_d

        sample_idx.append(init_idx)

        for _ in tqdm(range(n_coreset - 1)):
            min_p = 0
            max_min_d = 0.

            for i in range(filter_coefs.shape[2]):
                min_d = float('inf')

                for pair in pair_idx:
                    p1 = filter_coefs[pair[0], :, i]
                    p2 = filter_coefs[pair[1], :, i]
                    
                    for j in sample_idx:
                        q1 = filter_coefs[pair[0], :, j]
                        q2 = filter_coefs[pair[1], :, j]

                        d1 = linalg.norm(p1 - q1)
                        d2 = linalg.norm(p1 - q2)
                        d3 = linalg.norm(p2 - q1)
                        d4 = linalg.norm(p2 - q2)

                        min_d = np.amin([min_d, d1, d2, d3, d4])

                if min_d > max_min_d:
                    min_p = i
                    max_min_d = min_d

            sample_idx.append(min_p)

        # Extract coreset
        return dataset[:, sample_idx].flatten(), pca_basis

    else:
        # Indices into dataset / filter_coefs
        sample_idx = [0]

        for _ in range(n_coreset - 1):
            min_p = 0
            max_min_d = 0.

            for i in range(filter_coefs.shape[1]):
                p = filter_coefs[:,i]
                min_d = float('inf')

                for j in sample_idx:
                    min_d = min(min_d, linalg.norm(filter_coefs[:,j] - p))

                if min_d > max_min_d:
                    min_p = i
                    max_min_d = min_d

            sample_idx.append(min_p)

        # Extract coreset
        return dataset[sample_idx], pca_basis

def fit_basis_nonzero(spectrum, basis, alpha = 1e-2):
    """ Find the best-fit coefficients for given basis functions such that the 
        final spectrum is >= 0 for all wavelengths and obey smoothness constraint. """
    n_wav = spectrum.shape[0]

    b = np.vstack((spectrum.reshape(-1, 1), np.zeros((n_wav - 2, 1))))
    b_mb = matlab.double(b.tolist())

    b_neg_mb = matlab.double((-1 * basis).tolist())
    z_mb = matlab.double(np.zeros((n_wav, 1)).tolist())

    W = np.zeros((n_wav - 2, n_wav))
    for i in range(n_wav - 2):
        W[i, i  ] =  1
        W[i, i+1] = -2
        W[i, i+2] =  1
    
    # Smoothness constraint matrices
    Wb = alpha * W @ basis

    A = np.vstack((basis, Wb))
    A_mb = matlab.double(A.tolist())

    coefs_mb = eng.lsqlin(A_mb, b_mb, b_neg_mb, z_mb, nargout = 2)

    return np.array(coefs_mb[0]._data)

def fit_basis_zero_one(spectrum, basis, alpha = 1e-2):
    """ Find the best-fit coefficients for given basis functions such that the 
        final spectrum is >= 0 for all wavelengths and obey smoothness constraint. """
    n_wav = spectrum.shape[0]

    b = np.vstack((spectrum.reshape(-1, 1), np.zeros((n_wav - 2, 1))))
    b_mb = matlab.double(b.tolist())

    B_constr = np.vstack((-1 * basis, basis))
    z_constr = np.vstack((np.zeros((n_wav, 1)), np.ones((n_wav, 1))))

    B_mb = matlab.double(B_constr.tolist())
    z_mb = matlab.double(z_constr.tolist())

    W = np.zeros((n_wav - 2, n_wav))
    for i in range(n_wav - 2):
        W[i, i  ] =  1
        W[i, i+1] = -2
        W[i, i+2] =  1
    
    # Smoothness constraint matrices
    Wb = alpha * W @ basis

    A = np.vstack((basis, Wb))
    A_mb = matlab.double(A.tolist())

    coefs_mb = eng.lsqlin(A_mb, b_mb, B_mb, z_mb, nargout = 2)

    return np.array(coefs_mb[0]._data)

def recover_sensor_illum(exp, spectra, c0, e0 = None, alpha = 1e-2, beta = 1e-2, n_iter = 100, lr_sensor = 0., lr_illum = 0.):
    """ Find spectral correction product for each channel (sensor response * illuminant * correction)
        from RGB measurements of color checker (known reflectance spectra)
    """
    # Assume here that pixels have no filter applied to them and are in a 1D list
    pixels = exp.pixel_major()
    pixels = pixels.reshape(-1, 3)

    pw = np.vstack((pixels.reshape(-1, 1), np.zeros((exp.n_wav - 2, 1))))
    p_mb = matlab.double(pw.tolist())
    Pw = np.vstack((pixels, np.zeros((exp.n_wav - 2, 3))))

    # Basis function matrices
    S = np.transpose(exp.sensor_basis, axes = (0, 2, 1))   # channel x n_wav x n_basis_s
    E = exp.illum_basis.transpose()                        # n_wav x n_basis_e

    S_mb = [matlab.double((-1 * S_).tolist()) for S_ in S]
    E_mb = matlab.double((-1 * E).tolist())
    z_mb = matlab.double(np.zeros((exp.n_wav, 1)).tolist())
    n_mb = matlab.double([])

    # Second order difference matrix
    W = np.zeros((exp.n_wav - 2, exp.n_wav))
    for i in range(exp.n_wav - 2):
        W[i, i  ] =  1
        W[i, i+1] = -2
        W[i, i+2] =  1
    
    # Smoothness constraint matrices
    Ws = [alpha * W @ S_ for S_ in S]
    Wa = beta  * W @ E

    # Unknowns (camera response and illuminant)
    c = np.copy(c0)
    e = np.zeros(exp.n_basis_e) if e0 is None else e0

    resnorm = np.zeros(n_iter)

    for iter in tqdm(range(n_iter)):
        # np.save('data/sensor_illum_tests/sensor-' + str(iter), c)
        # np.save('data/sensor_illum_tests/illum-' + str(iter), e)
        c_spectra = [S_ @ c[:, k] for k, S_ in enumerate(S)]

        A = np.zeros((exp.n_pix * 3, exp.n_basis_e))
        for i in range(exp.n_pix):
            for k in range(3):
                for j in range(exp.n_basis_e):
                    A[i * 3 + k, j] = np.sum(spectra[i, :] * c_spectra[k][:] * E[:, j])

        # Solve for illuminant
        A = np.vstack((A, Wa))

        A_mb = matlab.double(A.tolist())
        e_mb = eng.lsqlin(A_mb, p_mb, E_mb, z_mb, nargout = 2)
        resnorm[iter] = e_mb[1]
        # e_tmp = np.array(e_mb[0]._data)
        # e = e + lr_illum * (e_tmp - e)
        e = np.array(e_mb[0]._data)

        e_spectrum = (E @ e).flatten()

        B = np.zeros((3, exp.n_pix, exp.n_basis_s))
        for k in range(3):
            for i in range(exp.n_pix):
                for j in range(exp.n_basis_s):
                    B[k, i, j] = np.sum(spectra[i, :] * e_spectrum * S[k, :, j])

        # Solve for camera response 
        for k in range(3):
            B_k = np.vstack((B[k, :, :], Ws[k]))
            B_mb = matlab.double(B_k.tolist())
            k_mb = matlab.double(Pw[:, k].tolist())
            result = eng.lsqlin(B_mb, k_mb, S_mb[k], z_mb)
            c[:, k] = np.array(result._data)
            # c_tmp = np.array(result._data)
            # c[:,k] = c[:,k] + lr_sensor * (c_tmp - c[:,k])

        # Renormalize sensor data to max value
        s_r = S[0,:,:] @ c[:,0]
        s_g = S[1,:,:] @ c[:,1]
        s_b = S[2,:,:] @ c[:,2]

        s_max = np.amax(np.hstack((s_r, s_g, s_b)))

        s_r /= s_max
        s_g /= s_max
        s_b /= s_max

        c[:,0] = fit_basis_nonzero(s_r, S[0,:,:], alpha = alpha)
        c[:,1] = fit_basis_nonzero(s_g, S[1,:,:], alpha = alpha)
        c[:,2] = fit_basis_nonzero(s_b, S[2,:,:], alpha = alpha)

    # np.save('data/sensor_illum_tests/resnorm', resnorm)

    return c, e

def recover_sensor(exp, spectra, illum, c0 = None, n_iter = 100, alpha = 1e-2):
    """ Find spectral correction product for each channel (sensor response * illuminant * correction)
        from RGB measurements of color checker (known reflectance spectra)
    """
    # Assume here that pixels have no filter applied to them and are in a 1D list
    pixels = exp.pixel_major()
    pixels = pixels.reshape(-1, 3)

    pw_ = pixels.reshape(-1, 1)
    p_mb_ = matlab.double(pw_.tolist())

    pw = np.vstack((pixels.reshape(-1, 1), np.zeros((exp.n_wav - 2, 1))))
    p_mb = matlab.double(pw.tolist())
    Pw = np.vstack((pixels, np.zeros((exp.n_wav - 2, 3))))

    # Basis function matrices
    S = np.transpose(exp.sensor_basis, axes = (0, 2, 1))   # channel x n_wav x n_basis_s

    S_mb = [matlab.double((-1 * S_).tolist()) for S_ in S]
    z_mb = matlab.double(np.zeros((exp.n_wav, 1)).tolist())

    # Second order difference matrix
    W = np.zeros((exp.n_wav - 2, exp.n_wav))
    for i in range(exp.n_wav - 2):
        W[i, i  ] =  1
        W[i, i+1] = -2
        W[i, i+2] =  1
    
    # Smoothness constraint matrices
    Ws = [alpha * W @ S_ for S_ in S]

    # Unknowns (camera response and illuminant)
    c = np.copy(c0) if c0 is not None else np.zeros((S.shape[2], 3))
    illuminant_scale = 1.

    # B = np.zeros((3, exp.n_pix, exp.n_basis_s))
    # for k in range(3):
    #     for i in range(exp.n_pix):
    #         for j in range(exp.n_basis_s):
    #             B[k, i, j] = np.sum(spectra[i, :] * illum * illuminant_scale * S[k, :, j])

    # # Solve for camera response 
    # for k in range(3):
    #     B_k = np.vstack((B[k, :, :], Ws[k]))
    #     B_mb = matlab.double(B_k.tolist())
    #     k_mb = matlab.double(Pw[:, k].tolist())
    #     result = eng.lsqlin(B_mb, k_mb, S_mb[k], z_mb, nargout = 2)
    #     c[:, k] = np.array(result[0]._data)

    resnorm_illum = np.zeros(n_iter)
    resnorm_sensor = np.zeros((3,n_iter))

    for iter in tqdm(range(n_iter)):
        np.save('data/sensor_illum_tests/sensor-' + str(iter), c)
        np.save('data/sensor_illum_tests/illum-' + str(iter), illum * illuminant_scale)

        B = np.zeros((3, exp.n_pix, exp.n_basis_s))
        for k in range(3):
            for i in range(exp.n_pix):
                for j in range(exp.n_basis_s):
                    B[k, i, j] = np.sum(spectra[i, :] * illum * illuminant_scale * S[k, :, j])

        # Solve for camera response 
        for k in range(3):
            B_k = np.vstack((B[k, :, :], Ws[k]))
            B_mb = matlab.double(B_k.tolist())
            k_mb = matlab.double(Pw[:, k].tolist())
            result = eng.lsqlin(B_mb, k_mb, S_mb[k], z_mb, nargout = 2)
            resnorm_sensor[k,iter] = result[1]
            c[:, k] = np.array(result[0]._data)

        # Renormalize sensor data to max value
        s_r = S[0,:,:] @ c[:,0]
        s_g = S[1,:,:] @ c[:,1]
        s_b = S[2,:,:] @ c[:,2]

        s_max = np.amax(np.hstack((s_r, s_g, s_b)))

        s_r /= s_max
        s_g /= s_max
        s_b /= s_max

        c[:,0] = fit_basis_nonzero(s_r, S[0,:,:], alpha = alpha)
        c[:,1] = fit_basis_nonzero(s_g, S[1,:,:], alpha = alpha)
        c[:,2] = fit_basis_nonzero(s_b, S[2,:,:], alpha = alpha)

        c_spectra = [S_ @ c[:, k] for k, S_ in enumerate(S)]

        A = np.zeros((exp.n_pix * 3, 1))
        for i in range(exp.n_pix):
            for k in range(3):
                spectra_illum = spectra[i, :] * illum
                A[i * 3 + k, 0] = np.sum(spectra_illum * c_spectra[k][:])

        # Solve for illuminant scale factor
        A_mb = matlab.double(A.tolist())
        e_mb = eng.lsqlin(A_mb, p_mb_, nargout = 2)
        resnorm_illum[iter] = e_mb[1]
        illuminant_scale = e_mb[0]

    np.save('data/sensor_illum_tests/resnorm-illum', resnorm_illum)
    np.save('data/sensor_illum_tests/resnorm-sensor', resnorm_sensor)

    return c, illuminant_scale

def recover_illum(exp, gt_spectra, sensor_response, alpha = 1e-2):
    """ Recover illuminant + correction factor
        from RGB measurements of color checker (known reflectance spectra)
        and known sensor responses.
    """
    # Assume here that pixels have no filter applied to them and are in a 1D list
    pixels = exp.pixel_major()
    pixels = pixels.reshape(-1, 3)

    pw = np.vstack((pixels.reshape(-1, 1), np.zeros((exp.n_wav - 2, 1))))
    p_mb = matlab.double(pw.tolist())
    Pw = np.vstack((pixels, np.zeros((exp.n_wav - 2, 3))))

    # Basis function matrices
    E = exp.illum_basis.transpose()                        # n_wav x n_basis_e

    E_mb = matlab.double((-1 * E).tolist())
    z_mb = matlab.double(np.zeros((exp.n_wav, 1)).tolist())

    # Second order difference matrix
    W = np.zeros((exp.n_wav - 2, exp.n_wav))
    for i in range(exp.n_wav - 2):
        W[i, i  ] =  1
        W[i, i+1] = -2
        W[i, i+2] =  1
    
    # Smoothness constraint matrices
    Wa = alpha * W @ E

    A = np.zeros((exp.n_pix * 3, exp.n_basis_e))
    for i in range(exp.n_pix):
        for k in range(3):
            for j in range(exp.n_basis_e):
                A[i * 3 + k, j] = np.sum(gt_spectra[i, :] * sensor_response[k, :] * E[:, j])

    # Solve for illuminant
    A = np.vstack((A, Wa))

    A_mb = matlab.double(A.tolist())
    e_mb = eng.lsqlin(A_mb, p_mb, E_mb, z_mb)
    e = np.array(e_mb._data)
    # e = np.sign(e[0]) * e

    return e

def recover_illum_known(exp, gt_spectra, sensor_response, gt_illum):
    """ Find scale factor for illuminant of known spectrum given known
        reflectance spectra and sensor responses. """

    # Assume here that pixels have no filter applied to them and are in a 1D list
    pixels = exp.pixel_major()
    pixels = pixels.reshape(-1, 3)

    pw = pixels.reshape(-1, 1)
    p_mb = matlab.double(pw.tolist())

    A = np.zeros((exp.n_pix * 3, 1))
    for i in range(exp.n_pix):
        for k in range(3):
            spectra_illum = gt_spectra[i, :] * gt_illum
            A[i * 3 + k, 0] = np.sum(spectra_illum * sensor_response[k, :])

    # Solve for illuminant scale factor
    A_mb = matlab.double(A.tolist())
    e_mb = eng.lsqlin(A_mb, p_mb)

    return e_mb

def reconstruct_blind(exp, a = None, n_iter = 30, alpha = 1e-8, beta = 1e-2, R0 = None):
    """ Reconstruct unknown spectra given the RGB response of camera 
        through various filters.
        
        Following the method in "Do It Yourself Hyperspectral Imaging 
        with Everyday Digital Cameras." https://doi.org/10.1109/CVPR.2016.270
        See supplemental for detailed description of solving the bilinear system.
        Equation numbers are in reference to the main paper or supplemental.
    """
    pixels = exp.pixel_major() # n_pix x (n_ftr * 3)
    dim = pixels.shape[:-1]

    if len(dim) == 2:
        pixels = pixels.reshape(-1, exp.n_ftr * 3)

    # pixels /= np.amax(pixels)

    # Construct g vector (below Eq. 3 of supplemental)
    g = pixels.transpose().reshape(-1, 1)  # (n_pix * n_ftr * 3) x 1
    g_tilde = np.vstack((g, np.zeros((exp.n_wav - 2, 1))))
    g_mb = matlab.double(g_tilde.tolist())
    G_tilde = np.vstack((pixels.transpose(), np.zeros((exp.n_wav - 2, exp.n_pix))))

    # Basis function matrices
    B = exp.reflectance_basis.transpose()  # n_wav x n_basis_r
    E = exp.illum_basis.transpose()        # n_wav x n_basis_e

    B_constr = np.vstack((-1 * B, B))
    E_constr = np.vstack((-1 * E, E))
    z_constr = np.vstack((np.zeros((exp.n_wav, 1)), np.ones((exp.n_wav, 1))))

    B_mb = matlab.double(B_constr.tolist())
    E_mb = matlab.double(E_constr.tolist())
    z_mb = matlab.double(z_constr.tolist())
    n_mb = matlab.double([])

    # Second order difference matrix
    W = np.zeros((exp.n_wav - 2, exp.n_wav))
    for i in range(exp.n_wav - 2):
        W[i, i  ] = -1.
        W[i, i+1] =  2.
        W[i, i+2] = -1.

    # Smoothness constraint matrices
    Wr = alpha * W @ B
    Wa = beta  * W @ E

    # Construct A matrices (below Eq. 5 of main paper)
    A = np.zeros((exp.n_ftr * 3, exp.n_basis_r, exp.n_basis_e))
    for i in range(exp.n_basis_r):
        for j in range(exp.n_basis_e):
            for m in range(exp.n_ftr):
                for k in range(3):
                    A[m * 3 + k, i, j] = np.sum(B[:,i] * E[:,j] * exp.filters[m,:] * exp.sensor_response[k,:])

    # Unknowns
    e = np.copy(a) if a is not None else np.zeros((exp.n_basis_e, 1))
    R = np.zeros((exp.n_basis_r, exp.n_pix))
    R[0, :] = 1.   # init to first spectral basis

    if a is None:
        # ---------------------------------------------------------------------------------------------
        # ------------------------- SOLVE FOR ILLUMINANT AND REFLECTANCE ------------------------------
        # ---------------------------------------------------------------------------------------------
        for _ in range(n_iter):
            F = np.zeros((exp.n_pix * exp.n_ftr * 3, exp.n_basis_e))
            for m in range(exp.n_ftr):
                for k in range(3):
                    start = (m * 3 + k) * exp.n_pix
                    F[start : start + exp.n_pix, :] = R.transpose() @ A[m * 3 + k, :, :]

            # Add smoothness constraint
            F_tilde = np.vstack((F, Wa))

            # Solve for coefficients of illuminant basis (Eq. 4 of supplemental)
            F_mb = matlab.double(F_tilde.tolist())
            e_mb = eng.lsqlin(F_mb, g_mb, E_mb, z_mb, stdout=out,stderr=err)
            e = np.array(e_mb._data)
            e = np.sign(e[0]) * e

            # Construct T matrix (Eq. 5 of supplemental)
            T = np.zeros((exp.n_basis_r, exp.n_ftr * 3))
            for m in range(exp.n_ftr):
                for k in range(3):
                    T[:, m * 3 + k] = (A[m * 3 + k, :, :] @ e).squeeze()

            # Add smoothness constraint (Eq. 6 of supplemental)
            T_tilde = np.vstack((T.transpose(), Wr))

            # Solve for coefficients of reflectance basis for each pixel (Eq. 6 of supplemental)
            T_mb = matlab.double(T_tilde.tolist())
            for i in range(exp.n_pix):
                b_mb = matlab.double(G_tilde[:,i].tolist())
                result = eng.lsqlin(T_mb, b_mb, B_mb, z_mb, stdout=out,stderr=err)
                R[:,i] = np.array(result._data)

    else:
        # ---------------------------------------------------------------------------------------------
        # ------------------------------ SOLVE FOR REFLECTANCE ONLY -----------------------------------
        # ---------------------------------------------------------------------------------------------

        # Construct T matrix (Eq. 5 of supplemental)
        T = np.zeros((exp.n_basis_r, exp.n_ftr * 3))
        for m in range(exp.n_ftr):
            for k in range(3):
                T[:, m * 3 + k] = (A[m * 3 + k, :, :] @ e).squeeze()

        # Add smoothness constraint (Eq. 6 of supplemental)
        T_tilde = np.vstack((T.transpose(), Wr))
        G_tilde = np.vstack((pixels.transpose(), np.zeros((exp.n_wav - 2, exp.n_pix))))

        # Solve for coefficients of reflectance basis for each pixel (Eq. 6 of supplemental)
        T_mb = matlab.double(T_tilde.tolist())
        for i in range(exp.n_pix):
            b_mb = matlab.double(G_tilde[:,i].tolist())
            result = eng.lsqlin(T_mb, b_mb, B_mb, z_mb, stdout=out,stderr=err)
            R[:,i] = np.array(result._data)

    # Reconstruct spectra at each pixel from basis functions
    reconstructed = B @ R

    return reconstructed.transpose(), e

def reconstruct_blind_illum_known(exp, gt_illum, alpha = 1e-8):
    """ Reconstruct unknown spectra given the RGB response of camera 
        through various filters.
        
        Following the method in "Do It Yourself Hyperspectral Imaging 
        with Everyday Digital Cameras." https://doi.org/10.1109/CVPR.2016.270
        See supplemental for detailed description of solving the bilinear system.
    """
    pixels = exp.pixel_major()
    dim = pixels.shape[:-1]

    if len(dim) == 2:
        pixels = pixels.reshape(-1, exp.n_ftr * 3)

    # pixels /= np.amax(pixels)

    # Construct g vector (below Eq. 3 of supplemental)
    g = pixels.transpose().reshape(-1, 1)  # (n_pix * n_ftr * 3) x 1
    g_tilde = np.vstack((g, np.zeros((exp.n_wav - 2, 1))))
    g_mb = matlab.double(g_tilde.tolist())
    G_tilde = np.vstack((pixels.transpose(), np.zeros((exp.n_wav - 2, exp.n_pix))))

    # Basis function matrices
    B = exp.reflectance_basis.transpose()  # n_wav x n_basis_r

    B_constr = np.vstack((-1 * B, B))
    z_constr = np.vstack((np.zeros((exp.n_wav, 1)), np.ones((exp.n_wav, 1))))

    B_mb = matlab.double(B_constr.tolist())
    z_mb = matlab.double(z_constr.tolist())

    # Second order difference matrix
    W = np.zeros((exp.n_wav - 2, exp.n_wav))
    for i in range(exp.n_wav - 2):
        W[i, i  ] = -1.
        W[i, i+1] =  2.
        W[i, i+2] = -1.

    # Smoothness constraint matrices
    Wr = alpha * W @ B

    # Construct A matrices (below Eq. 5 of main paper)
    A = np.zeros((exp.n_ftr * 3, exp.n_basis_r))
    for i in range(exp.n_basis_r):
        for m in range(exp.n_ftr):
            for k in range(3):
                A[m * 3 + k, i] = np.sum(B[:,i] * gt_illum * exp.filters[m,:] * exp.sensor_response[k,:])

    # Unknowns
    R = np.zeros((exp.n_basis_r, exp.n_pix))
    R[0, :] = 1.   # init to first spectral basis

    # ---------------------------------------------------------------------------------------------
    # ------------------------------ SOLVE FOR REFLECTANCE ONLY -----------------------------------
    # ---------------------------------------------------------------------------------------------

    # Construct T matrix (Eq. 5 of supplemental)
    T = np.zeros((exp.n_basis_r, exp.n_ftr * 3))
    for m in range(exp.n_ftr):
        for k in range(3):
            T[:, m * 3 + k] = (A[m * 3 + k, :]).squeeze()

    # Add smoothness constraint (Eq. 6 of supplemental)
    T_tilde = np.vstack((T.transpose(), Wr))
    G_tilde = np.vstack((pixels.transpose(), np.zeros((exp.n_wav - 2, exp.n_pix))))

    # Solve for coefficients of reflectance basis for each pixel (Eq. 6 of supplemental)
    T_mb = matlab.double(T_tilde.tolist())
    for i in range(exp.n_pix):
        b_mb = matlab.double(G_tilde[:,i].tolist())
        result = eng.lsqlin(T_mb, b_mb, B_mb, z_mb, stdout=out,stderr=err)
        R[:,i] = np.array(result._data)

    # Reconstruct spectra at each pixel from basis functions
    reconstructed = B @ R

    return reconstructed.transpose()

