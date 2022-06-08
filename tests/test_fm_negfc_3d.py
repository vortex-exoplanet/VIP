"""
Tests for fm/negfc*.py (3D ADI cube)

"""

import copy
from .helpers import aarc, np, parametrize, fixture
from vip_hci.fm import (confidence, firstguess, mcmc_negfc_sampling,
                        nested_negfc_sampling, nested_sampling_results,
                        speckle_noise_uncertainty, cube_planet_free,
                        show_walk_plot, show_corner_plot)
from vip_hci.psfsub import median_sub, pca, pca_annular, pca_annulus


# ====== utility function for injection
@fixture(scope="module")
def injected_cube_position(example_dataset_adi):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    print("injecting fake planet...")
    dsi = copy.copy(example_dataset_adi)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    gt = (30, 0, 300)
    dsi.inject_companions(gt[2], rad_dists=gt[0], theta=gt[1])

    return dsi, dsi.injections_yx[0], gt


# ====== Actual negfc tests for different parameters
@parametrize("pca_algo, negfc_algo, ncomp, mu_sigma, fm, force_rpa, conv_test",
             [
                 (pca_annular, firstguess, 3, False, 'stddev', False, None),
                 (pca, firstguess, 3, True, None, False, None),
                 (median_sub, firstguess, None, False, 'sum', False, None),
                 (pca_annulus, mcmc_negfc_sampling, 3, False, 'stddev', False, 'gb'),
                 (pca_annulus, mcmc_negfc_sampling, 3, True, None, True, 'ac'),
                 (pca_annulus, nested_negfc_sampling, 3, False, 'sum', False, None)
                 ])
def test_algos(injected_cube_position, pca_algo, negfc_algo, ncomp, mu_sigma,
               fm, force_rpa, conv_test):
    ds, yx, gt = injected_cube_position

    # run firstguess with simplex only if followed by mcmc or nested sampling
    if pca_algo == median_sub:
        algo_options = {'imlib': 'opencv', 'verbose': False}
    else:
        algo_options = {'imlib': 'opencv'}
    res0 = firstguess(ds.cube, ds.angles, ds.psf, ncomp=ncomp,
                      planets_xy_coord=np.array([[yx[1], yx[0]]]), fwhm=ds.fwhm,
                      simplex=negfc_algo == firstguess, algo=pca_algo, fmerit=fm,
                      mu_sigma=mu_sigma, force_rPA=force_rpa,
                      aperture_radius=2, annulus_width=4*ds.fwhm,
                      algo_options=algo_options)
    res = (res0[0][0], res0[1][0], res0[2][0])
    init = np.array(res)

    if negfc_algo == firstguess:
        # use injection of 180 companions in empty cube to estimate error bars
        cube_emp = cube_planet_free(res, ds.cube, ds.angles, ds.psf,
                                    imlib='opencv')
        algo_options = {'imlib': 'opencv'}
        if pca_algo != median_sub:
            algo_options['ncomp'] = ncomp
        if pca_algo == pca_annular:
            algo_options['radius_int'] = res0[0][0]-2*ds.fwhm
            algo_options['asize'] = 4*ds.fwhm
            algo_options['delta_rot'] = 1
        if pca_algo == pca:
            # just test it once because very slow
            sp_unc = speckle_noise_uncertainty(cube_emp, res, 
                                               np.arange(0, 360, 3), ds.angles,
                                               algo=pca_algo, psfn=ds.psf,
                                               fwhm=ds.fwhm, aperture_radius=2,
                                               fmerit=fm, mu_sigma=mu_sigma,
                                               verbose=True, full_output=False,
                                               algo_options=algo_options,
                                               nproc=1)
        else:
            sp_unc = (2, 2, 0.1*gt[2])
        # compare results
        for i in range(3):
            aarc(res[i], gt[i], rtol=1e-1, atol=3*sp_unc[i])
    elif negfc_algo == mcmc_negfc_sampling:
        # define fake unit transmission (to test that branch of the algo)
        trans = np.zeros([2,10])
        trans[0] = np.linspace(0, ds.cube.shape[-1], 10, endpoint=True)
        trans[1,:] = 1
        # run MCMC
        if force_rpa:
            niteration_limit = 400
        else:
            niteration_limit = 210
        res = negfc_algo(ds.cube, ds.angles, ds.psf, initial_state=init,
                         algo=pca_algo, ncomp=ncomp, annulus_width=4*ds.fwhm,
                         aperture_radius=2, fwhm=ds.fwhm, mu_sigma=mu_sigma,
                         sigma='spe+pho', fmerit=fm, imlib='opencv', 
                         nwalkers=100, niteration_min=200, 
                         niteration_limit=niteration_limit, conv_test=conv_test, 
                         nproc=1, save=True, transmission=trans, 
                         force_rPA=force_rpa, verbosity=2)
        burnin = 0.3
        if force_rpa:
            labels = ['f']
            isamples = res[:, int(res.shape[1]//(1/burnin)):, :].reshape((-1, 1))
        else:
            labels = ['r', 'theta', 'f']
            isamples = res[:, int(res.shape[1]//(1/burnin)):, :].reshape((-1, 3))
        show_walk_plot(res, save=True, labels=labels)
        show_corner_plot(res, burnin=burnin, save=True, labels=labels)
        # infer most likely values + confidence intervals
        val_max, ci = confidence(isamples, cfd=68.27, gaussian_fit=False,
                                 verbose=True, save=True, labels=labels)
        # infer mu and sigma from gaussian fit
        mu, sigma = confidence(isamples, cfd=68.27, bins=100, gaussian_fit=True,
                               verbose=True, save=True, labels=labels)
        # make sure it is between 0 and 360 for theta for both mu and gt
        if not force_rpa:
            if val_max['theta']-gt[1] > 180:
                val_max['theta'] -= 360
            elif val_max['theta']-gt[1] < -180:
                val_max['theta'] += 360
            if mu[1]-gt[1] > 180:
                mu[1] -= 360
            elif mu[1]-gt[1] < -180:
                mu[1] += 360
        # compare results for each param
        for i, lab in enumerate(labels):
            if force_rpa:
                j = i+2
            else:
                j = i
            ci_max = np.amax(np.abs(ci[lab]))
            aarc(val_max[lab], gt[j], atol=3*ci_max)  # diff within 3 sigma
            aarc(val_max[lab], gt[j], atol=3*sigma[i])  # diff within 3 sigma
    else:
        # run nested sampling
        res = negfc_algo(init, ds.cube, ds.angles, ds.psf, ds.fwhm,
                         mu_sigma=mu_sigma, sigma='spe', fmerit=fm,
                         annulus_width=4*ds.fwhm, aperture_radius=2,
                         ncomp=ncomp, algo=pca_algo, w=(5, 5, 200),
                         method='single', npoints=100, dlogz=0.1,
                         decline_factor=None, rstate=None, verbose=True,
                         algo_options={'imlib': 'opencv'})
        # infer mu, sigma from nested sampling result
        mu_sig = nested_sampling_results(res, burnin=0.3, bins=None, save=False)
        # compare results for each param
        for i in range(3):
            # diff within 3 sigma
            aarc(mu_sig[i, 0], gt[i], atol=3*mu_sig[i, 1])
