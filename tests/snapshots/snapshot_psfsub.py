"""Generate snapshots for every psfsub function."""

__author__ = "Thomas Bedrine"

__all__ = "PSFADI_PATH"

import copy

import numpy as np
from vip_hci.psfsub import median_sub, frame_diff, llsg, xloci, nmf, nmf_annular
from vip_hci.metrics import detection, snrmap
from vip_hci.fits import open_fits
from vip_hci.objects import Dataset
from vip_hci.config import VLT_NACO
from tests.helpers import download_resource

PSFADI_PATH = "./tests/snapshots/psfsub_adi/"
DATASET_ELEMENTS = ["cube", "angles", "psf"]


def make_dataset_adi():
    """
    Download example FITS cube from github + prepare Dataset object.

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    We use the helper function ``download_resource`` which handles the request
    and puts it to sleep for a defined duration if too many requests are done.
    They inherently call the Astropy's ``download_file`` function which uses caching,
    so the file is downloaded at most once per test run.

    """
    print("downloading data...")

    url_prefix = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_resource(f"{url_prefix}/naco_betapic_cube_cen.fits")
    f2 = download_resource(f"{url_prefix}/naco_betapic_psf.fits")
    f3 = download_resource(f"{url_prefix}/naco_betapic_pa.fits")

    # load fits
    cube = open_fits(f1)
    angles = open_fits(f3).flatten()  # shape (61,1) -> (61,)
    psf = open_fits(f2)

    # create dataset object
    dataset = Dataset(cube=cube, angles=angles, psf=psf, px_scale=VLT_NACO["plsc"])

    dataset.normalize_psf(size=20, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


# Note : this function comes from the former test for adi psfsub, I did not write it,
# and I didn't found the author (feel free to add the author if you know them)
def injected_cube_position():
    """
    Inject a fake companion into an example cube.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    print("injecting fake planet...")
    dsi = copy.copy(make_dataset_adi())
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.inject_companions(300, rad_dists=30)

    return dsi, dsi.injections_yx[0]


# TODO: manually saving snapshots can be long, the use of pytest-snapshots is highly
# recommended, but only if you can overload the assert_match function, which currently
# only support exact equal assertions, thus being incompatible with ``np.allclose``
# More info on : https://pypi.org/project/pytest-snapshot/
def save_snapshots_psfsub_adi():
    """
    Save expected results (snapshots) for all PSF subtraction functions.

    The obtained frame and the list of companion candidates are saved in separate numpy
    files to be loaded in tests that require to compare if said function still generate
    an appropriate result.

    """
    betapic, position = injected_cube_position()

    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm
    psf = betapic.psf
    snr_thresh = 2

    imlib_rot = "vip-fft"
    interpolation = None

    # Median subtraction (full-frame)

    medsub_adi = median_sub(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        mode="fullfr",
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    # Median subtraction (annular)

    medsub_ann_adi = median_sub(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        mode="annular",
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    # Frame differencing

    framediff_adi = frame_diff(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        metric="l1",
        dist_threshold=90,
        delta_rot=0.5,
        radius_int=4,
        asize=fwhm,
        nproc=None,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    # Frame differencing (median of 4 similar)

    framediff4_adi = frame_diff(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        metric="l1",
        dist_threshold=90,
        delta_rot=0.5,
        radius_int=4,
        n_similar=4,
        asize=fwhm,
        nproc=None,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    # Local Low-rank plus Sparse plus Gaussian-noise decomposition

    llsg_adi = llsg(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        rank=5,
        thresh=1,
        max_iter=20,
        random_seed=10,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    # Locally Optimized Combination of Images

    loci_adi = xloci(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        asize=fwhm,
        n_segments="auto",
        nproc=None,
        metric="correlation",
        radius_int=20,
        dist_threshold=90,
        delta_rot=0.5,
        optim_scale_fact=3,
        solver="lstsq",
        tol=0.01,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    # Non-negative Matrix Factorization (full-frame)

    nmf_adi = nmf(
        cube=cube,
        angle_list=angles,
        ncomp=14,
        max_iter=10000,
        init_svd="nndsvdar",
        mask_center_px=None,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    # Non-negative Matrix Factorization (annular)

    nmf_ann_adi = nmf_annular(
        cube=cube,
        angle_list=angles,
        ncomp=9,
        max_iter=10000,
        init_svd="nndsvdar",
        radius_int=0,
        nproc=None,
        fwhm=fwhm,
        asize=fwhm,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    )

    for name, value in locals().items():
        if isinstance(value, np.ndarray) and name not in DATASET_ELEMENTS:
            snr_map = snrmap(value, fwhm, verbose=False)
            detect = detection(
                snr_map,
                fwhm=fwhm,
                mode="lpeaks",
                bkg_sigma=5,
                matched_filter=False,
                mask=True,
                snr_thresh=snr_thresh,
                plot=False,
                debug=True,
                full_output=False,
                verbose=False,
            )
            det_array = np.stack(detect, axis=-1)
            # NMF frames are somewhat random and cannot be compared, no need to save
            if "nmf" not in name:
                np.save(f"./tests/snapshots/psfsub_adi/{name}.npy", value)
            np.save(f"./tests/snapshots/psfsub_adi/{name}_detect.npy", det_array)
    return


save_snapshots_psfsub_adi()
