"""
Tests for the post-processing pipeline, using the functional API.

"""

import copy
import vip_hci as vip
from .helpers import np, parametrize, fixture


def print_debug(s, *args, **kwargs):
    print(("\033[34m" + s + "\033[0m").format(*args, **kwargs))


@fixture(scope="module")
def injected_cube_position(example_dataset_ifs):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_ifs : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    print_debug("injecting fake planet...")
    dsi = copy.copy(example_dataset_ifs)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.inject_companions(50, rad_dists=30)

    return dsi, dsi.injections_yx[0]

@fixture(scope="module")
def estimated_scal_factor(example_dataset_ifs):
    """
    Estimate the scaling factor required to align stellar halo in IFS cube.

    Parameters
    ----------
    example_dataset_ifs : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    print_debug("estimating scaling factor...")
    dsi = copy.copy(example_dataset_ifs)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi_flux = np.ones_like(dsi.wavelengths)
    scal_fac_ori = dsi.wavelengths[-1]/dsi.wavelengths
    scal_fac, _ = vip.preproc.find_scal_vector(dsi.psf, dsi.wavelengths, 
                                               dsi_flux, nfp=2, fm="stddev")

    return scal_fac_ori, scal_fac

# ====== algos
def algo_medsub(ds, sc):
    return vip.psfsub.median_sub(ds.cube, ds.angles, fwhm=ds.fwhm,
                                 scale_list=sc)


def algo_medsub_annular(ds, sc):
    return vip.psfsub.median_sub(ds.cube, ds.angles, fwhm=ds.fwhm,
                                 scale_list=sc, mode='annular',
                                 radius_int=10)


def algo_xloci(ds, sc):
    return vip.psfsub.xloci(ds.cube, ds.angles, fwhm=ds.fwhm, scale_list=sc, 
                            asize=12)

def algo_xloci_double(ds, sc):
    return vip.psfsub.xloci(ds.cube, ds.angles, fwhm=ds.fwhm, scale_list=sc, 
                            adimsdi='double', asize=12)

def algo_pca_single(ds, sc):
    return vip.psfsub.pca(ds.cube, ds.angles, scale_list=sc,
                          adimsdi='single', ncomp=10)

def algo_pca_double(ds, sc):
    return vip.psfsub.pca(ds.cube, ds.angles, scale_list=sc,
                          adimsdi='double', ncomp=(1, 2))

def algo_pca_annular(ds, sc):
    return vip.psfsub.pca_annular(ds.cube, ds.angles, scale_list=sc,
                                  radius_int=10, ncomp=(1, 1), delta_sep=0.1)


# ====== SNR map
def snrmap_fast(frame, ds):
    return vip.metrics.snrmap(frame, fwhm=np.mean(ds.fwhm), approximated=True)


def snrmap(frame, ds):
    return vip.metrics.snrmap(frame, fwhm=np.mean(ds.fwhm))


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
                                  debug=False, full_output=True, verbose=True)
    msg = "Injected companion not recovered"
    assert verify_expcoord(table.y, table.x, yx_exp), msg


@parametrize("algo, make_detmap",
    [
        (algo_medsub, None),
        (algo_medsub_annular, None),
        (algo_xloci, snrmap_fast),
        (algo_xloci_double, snrmap_fast),
        (algo_pca_single, snrmap_fast),
        (algo_pca_double, snrmap_fast),
        (algo_pca_annular, None),
    ],
    ids=lambda x: (x.__name__.replace("algo_", "") if callable(x) else x))
def test_algos(injected_cube_position, estimated_scal_factor, algo, make_detmap):
    ds, position = injected_cube_position
    sc, scal_fac = estimated_scal_factor
    frame = algo(ds, sc)

    if make_detmap is not None:
        detmap = make_detmap(frame, ds)
    else:
        detmap = frame

    check_detection(detmap, position, np.mean(ds.fwhm), snr_thresh=2)
