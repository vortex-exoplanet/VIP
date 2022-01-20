"""
Tests for the post-processing pipeline, using the functional API.

"""

import copy
import vip_hci as vip
from .helpers import np, parametrize, fixture


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

    dsi.inject_companions(300, rad_dists=30)

    return dsi, dsi.injections_yx[0]


# ====== algos
def algo_medsub(ds):
    return vip.psfsub.median_sub(ds.cube, ds.angles, fwhm=ds.fwhm,
                                 mode="fullfr")


def algo_medsub_annular(ds):
    return vip.psfsub.median_sub(ds.cube, ds.angles, fwhm=ds.fwhm,
                                 mode="annular")


def algo_xloci(ds):
    return vip.psfsub.xloci(ds.cube, ds.angles, fwhm=ds.fwhm,
                             radius_int=20)  # <- speed up


def algo_frdiff(ds):
    return vip.psfsub.frame_diff(ds.cube, ds.angles)

def algo_frdiff4(ds):
    return vip.psfsub.frame_diff(ds.cube, ds.angles, n_similar=4)

def algo_llsg(ds):
    return vip.psfsub.llsg(ds.cube, ds.angles, ds.fwhm, rank=2)


def algo_nmf(ds):
    return vip.psfsub.nmf(ds.cube, ds.angles)

def algo_nmf_annular(ds):
    return vip.psfsub.nmf_annular(ds.cube, ds.angles)

def algo_pca(ds):
    return vip.psfsub.pca(ds.cube, ds.angles)


def algo_pca_grid(ds):
    """ PCA grid, obtaining the optimal residual for given location
    """
    return vip.psfsub.pca(ds.cube, ds.angles, ncomp=(1, 2),
                       source_xy=ds.injections_yx[0][::-1])


def algo_pca_incremental(ds):
    return vip.psfsub.pca(ds.cube, ds.angles, batch=int(ds.cube.shape[0]/2))


def algo_pca_annular(ds):
    return vip.psfsub.pca_annular(ds.cube, ds.angles, fwhm=ds.fwhm)


def algo_andromeda(ds):
    res = vip.invprob.andromeda(ds.cube[:,:-1,:-1], oversampling_fact=1.8,
                                  angles=ds.angles, psf=ds.psf)
    contrast, snr, snr_n, stdcontrast, stdcontrast_n, likelihood, r = res
    return snr_n


def algo_andromeda_fast(ds):
    res = vip.invprob.andromeda(ds.cube[:,:-1,:-1], oversampling_fact=0.5,
                                  fast=10, angles=ds.angles, psf=ds.psf)
    contrast, snr, snr_n, stdcontrast, stdcontrast_n, likelihood, r = res
    return snr_n


# ====== SNR map
def snrmap_fast(frame, ds):
    return vip.metrics.snrmap(frame, fwhm=ds.fwhm, approximated=True)


def snrmap(frame, ds):
    return vip.metrics.snrmap(frame, fwhm=ds.fwhm)


# ====== Detection with ``vip_hci.metrics.detection``, by default with a
# location error or 3px
def check_detection(frame, yx_exp, fwhm, snr_thresh, deltapix=3):
    """
    Verify if injected companion is recovered.

    Parameters
    ----------
    frame : 2d ndarray
    yx_exp : tuple(y, x)
        Expected position of the fake companion (= injected position).
    fwhm : int or float
        FWHM.
    snr_thresh : int or float, optional
        S/N threshold.
    deltapix : int or float, optional
        Error margin in pixels, between the expected position and the recovered.

    """
    def verify_expcoord(vectory, vectorx, exp_yx):
        for coor in zip(vectory, vectorx):
            print(coor, exp_yx)
            if np.allclose(coor[0], exp_yx[0], atol=deltapix) and \
                    np.allclose(coor[1], exp_yx[1], atol=deltapix):
                return True
        return False

    table = vip.metrics.detection(frame, fwhm=fwhm, mode='lpeaks', bkg_sigma=5,
                                  matched_filter=False, mask=True,
                                  snr_thresh=snr_thresh, plot=False,
                                  debug=True, full_output=True, verbose=True)
    msg = "Injected companion not recovered"
    assert verify_expcoord(table.y, table.x, yx_exp), msg


@parametrize("algo, make_detmap",
    [
        (algo_medsub, snrmap_fast),
        (algo_medsub, snrmap),
        (algo_medsub_annular, snrmap_fast),
        (algo_xloci, snrmap_fast),
        (algo_nmf, snrmap_fast),
        (algo_nmf_annular, snrmap_fast),
        (algo_llsg, snrmap_fast),
        (algo_frdiff, snrmap_fast),
        (algo_frdiff4, snrmap_fast),
        (algo_pca, snrmap_fast),
        (algo_pca_grid, snrmap_fast),
        (algo_pca_incremental, snrmap_fast),
        (algo_pca_annular, snrmap_fast),
        (algo_andromeda, None),
        (algo_andromeda_fast, None),
    ],
    ids=lambda x: (x.__name__.replace("algo_", "") if callable(x) else x))
def test_algos(injected_cube_position, algo, make_detmap):
    ds, position = injected_cube_position
    frame = algo(ds)

    if make_detmap is not None:
        detmap = make_detmap(frame, ds)
    else:
        detmap = frame

    check_detection(detmap, position, ds.fwhm, snr_thresh=2)
