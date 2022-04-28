"""
Tests for fm/negfc*.py (3D ADI cube)

"""

import copy
from .helpers import aarc, np, parametrize, fixture
from vip_hci.fm import (confidence, firstguess, mcmc_negfc_sampling, 
                        nested_negfc_sampling, nested_sampling_results,
                        speckle_noise_uncertainty, cube_planet_free)
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
@parametrize("pca_algo, negfc_algo, ncomp, mu_sigma, fm",
    [
        (pca_annular, firstguess, 2, False, 'sum'),
        (pca, firstguess, 2, False, 'stddev'),
        (median_sub, firstguess, None, True, None),
        (pca_annulus, mcmc_negfc_sampling, 1, False, 'sum'),
        (pca_annulus, nested_negfc_sampling, 1, True, None)
    ])
def test_algos(injected_cube_position, pca_algo, negfc_algo, ncomp, mu_sigma, 
               fm):
    ds, yx, gt = injected_cube_position
    
    # run firstguess with simplex only if followed by mcmc or nested sampling
    res0 = firstguess(ds.cube, ds.angles, ds.psf, ncomp=ncomp, 
                      planets_xy_coord=np.array([[yx[1],yx[0]]]), fwhm=ds.fwhm,
                      simplex=negfc_algo==firstguess, algo=pca_algo, fmerit=fm, 
                      mu_sigma=mu_sigma, imlib='opencv', aperture_radius=2, 
                      annulus_width=4*ds.fwhm)
    res = (res0[0][0], res0[1][0], res0[2][0])
    init = np.array(res)
    
    if negfc_algo == firstguess:
        # use injection of 180 companions in empty cube to estimate error bars
        cube_emp = cube_planet_free(res, ds.cube, ds.angles, ds.psf, 
                                    imlib='opencv')
        sp_unc = speckle_noise_uncertainty(cube_emp, res, np.arange(0,360,2), 
                                           ds.angles, algo=pca_algo, 
                                           psfn=ds.psf, fwhm=ds.fwhm, 
                                           aperture_radius=2, fmerit=fm, 
                                           mu_sigma=mu_sigma, verbose=False, 
                                           full_output=False, 
                                           algo_options={'imlib':'opencv'})
        # compare results
        for i in range(3):
            aarc(res[i], gt[i], rtol=1e-1, atol=sp_unc[i])
    elif negfc_algo == mcmc_negfc_sampling:
        # run MCMC
        
        res = negfc_algo(ds.cube, ds.angles, ds.psf, initial_state=init, 
                         algo=pca_algo, ncomp=ncomp, annulus_width=4*ds.fwhm, 
                         aperture_radius=2, fwhm=ds.fwhm, mu_sigma=mu_sigma, 
                         sigma='spe', fmerit=fm, imlib='opencv', nwalkers=100, 
                         niteration_limit=500, conv_test='ac')
        burnin = 0.3
        isamples = res[:, int(res.shape[1]//(1/burnin)):, :].reshape((-1,3))
        # infer most likely values + confidence intervals
        labels =['r', 'theta', 'f']
        val_max, ci = confidence(isamples, cfd=68.27, gaussian_fit=True, 
                                 verbose=False, save=False, labels=labels)
        # compare results for each param
        for i, lab in enumerate(labels):
            ci_max = np.amax(np.abs(ci[lab]))
            aarc(val_max[lab], gt[i], atol=ci_max) #diff within 1 sigma
        # infer mu and sigma from gaussian fit
        mu, sigma = confidence(isamples, cfd=68.27, bins=100, gaussian_fit=True, 
                               verbose=False, save=False, labels=labels)
        # compare results for each param
        for i in range(3):
            aarc(mu[i], gt[i], atol=sigma[i]) #diff within 1 sigma
    else:
        # run nested sampling
        res = negfc_algo(init, ds.cube, ds.angles, ds.psf, ds.fwhm, 
                         mu_sigma=mu_sigma, sigma='spe', fmerit=fm, 
                         annulus_width=4*ds.fwhm, aperture_radius=2, 
                         ncomp=ncomp, algo=pca_algo, w=(5, 5, 200), 
                         method='single', npoints=100, dlogz=0.1, 
                         decline_factor=None, rstate=None, verbose=False,
                         algo_options={'imlib':'opencv'})
        # infer mu, sigma from nested sampling result
        mu_sig = nested_sampling_results(res, burnin=0.3, bins=None, save=False)
        # compare results for each param
        for i in range(3):
            aarc(mu_sig[i,0], gt[i], atol=mu_sig[i,1]) #diff within 1 sigma

