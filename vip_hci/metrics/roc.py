"""
ROC curves generation.
"""
__all__ = ['EvalRoc',
           'compute_binary_map']

import copy
import numpy as np
import matplotlib.pyplot as plt
from hciplot import plot_frames
from scipy import stats
from photutils import detect_sources
from munch import Munch
from ..config import time_ini, timing, Progressbar
from ..fm import cube_inject_companions
from ..psfsub.svd import SVDecomposer
from ..var import frame_center, get_annulus_segments, get_circle

# TODO: remove the munch dependency
class EvalRoc(object):
    """
    Class for the generation of receiver operating characteristic (ROC) curves.
    """

    COLOR_1 = "#d62728"  # CADI
    COLOR_2 = "#ff7f0e"  # PCA
    COLOR_3 = "#2ca02c"  # LLSG
    COLOR_4 = "#9467bd"  # SODIRF
    COLOR_5 = "#1f77b4"  # SODINN
    SYMBOL_1 = "^"  # CADI
    SYMBOL_2 = "X"  # PCA
    SYMBOL_3 = "P"  # LLSG
    SYMBOL_4 = "s"  # SODIRF
    SYMBOL_5 = "p"  # SODINN
    # For model PSF subtraction algos that rely on a S/N map
    THRESHOLDS_05_5 = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    # For algos that output a likelihood or probability map
    THRESHOLDS_01_099 = np.linspace(0.1, 0.99, 10).tolist()

    def __init__(self, dataset, plsc=0.0272, n_injections=100, inrad=8,
                 outrad=12, dist_flux=("uniform", 2, 500), mask=None):
        """
        [...]
        dist_flux : tuple ('method', *args)
            'method' can be a string, e.g:
                ("skewnormal", skew, mean, var)
                ("uniform", low, high)
                ("normal", loc, scale)
            or a function.
        [...]
        """
        self.dataset = dataset
        self.plsc = plsc
        self.n_injections = n_injections
        self.inrad = inrad
        self.outrad = outrad
        self.dist_flux = dist_flux
        self.mask = mask
        self.methods = []

    def add_algo(self, name, algo, color, symbol, thresholds):
        """
        Parameters
        ----------
        algo : HciPostProcAlgo
        thresholds : list of lists

        """
        self.methods.append(Munch(algo=algo, name=name, color=color,
                                  symbol=symbol, thresholds=thresholds))

    def inject_and_postprocess(self, patch_size, cevr=0.9,
                               expvar_mode='annular', nproc=1):
        """

        Notes
        -----
        # TODO `methods` are not returned inside `results` and are *not* saved!
        # TODO order of parameters for `skewnormal` `dist_flux` changed! (was [3], [1], [2])
        # TODO `save` not implemented

        """
        from .. import hci_postproc

        starttime = time_ini()

        frsize = self.dataset.cube.shape[1]

        # ===== number of PCs for PCA / rank for LLSG
        if cevr is not None:
            svdecomp = SVDecomposer(self.dataset.cube, mode=expvar_mode,
                                    inrad=self.inrad, outrad=self.outrad,
                                    svd_mode='lapack', verbose=False)
            _ = svdecomp.get_cevr(ncomp_list=None)
            ratio_cumsum = svdecomp.cevr
            self.optpcs = np.searchsorted(ratio_cumsum, cevr) + 1
            print("{}% of CEVR with {} PCs".format(cevr, self.optpcs))

            # for m in methods:
            #     if hasattr(m, "ncomp") and m.ncomp is None:  # PCA
            #         m.ncomp = self.optpcs
            #
            #     if hasattr(m, "rank") and m.rank is None:  # LLSG
            #         m.rank = self.optpcs

            #
            #   ------> this should be moved inside the HCIPostProcAlgo classes!
            #
        # Getting indices in annulus
        width = self.outrad - self.inrad
        yy, xx = get_annulus_segments(self.dataset.cube[0], self.inrad,
                                      width)[0]
        num_patches = yy.shape[0]

        # Defining Fluxes according to chosen distribution
        dist_fkt = dict(skewnormal=stats.skewnorm.rvs,
                        normal=np.random.normal,
                        uniform=np.random.uniform).get(self.dist_flux[0],
                                                       self.dist_flux[0])

        self.fluxes = dist_fkt(*self.dist_flux[1:], size=self.n_injections)
        self.fluxes.sort()
        inds_inj = np.random.randint(0, num_patches, size=self.n_injections)

        self.dists = []
        self.thetas = []
        for m in range(self.n_injections):
            injx = xx[inds_inj[m]]
            injy = yy[inds_inj[m]]
            injx -= frame_center(self.dataset.cube[0])[1]
            injy -= frame_center(self.dataset.cube[0])[0]
            dist = np.sqrt(injx**2 + injy**2)
            theta = np.mod(np.arctan2(injy, injx) / np.pi * 180, 360)
            self.dists.append(dist)
            self.thetas.append(theta)

        for m in self.methods:
            m.frames = []
            m.probmaps = []

        self.list_xy = []

        # Injections
        for n in Progressbar(range(self.n_injections), desc="injecting"):
            cufc, cox, coy = _create_synt_cube(self.dataset.cube,
                                               self.dataset.psf,
                                               self.dataset.angles, self.plsc,
                                               theta=self.thetas[n],
                                               flux=self.fluxes[n],
                                               dist=self.dists[n],
                                               verbose=False)
            cox = int(np.round(cox))
            coy = int(np.round(coy))
            self.list_xy.append((cox, coy))

            for m in self.methods:
                # TODO: this is not elegant at all.
                # shallow copy. Should not copy e.g. the cube in memory,
                # just reference it.
                algo = copy.copy(m.algo)
                _dataset = copy.copy(self.dataset)
                _dataset.cube = cufc

                if isinstance(algo, hci_postproc.HCIPca):
                    algo.ncomp = self.optpcs
                # elif isinstance(algo, hci_postproc.HCILLSG):
                #     algo.rank = self.optpcs

                algo.run(dataset=_dataset, verbose=False)
                algo.make_snrmap(approximated=True, nproc=nproc, verbose=False)

                m.frames.append(algo.frame_final)
                m.probmaps.append(algo.snr_map)

        timing(starttime)

    def compute_tpr_fps(self, **kwargs):
        """
        Calculate number of dets/fps for every injection/method/threshold.

        Take the probability maps and the desired thresholds for every method,
        and calculates the binary map, number of detections and FPS using
        ``compute_binary_map``. Sets each methods ``detections``, ``fps`` and
        ``bmaps`` attributes.

        Parameters
        ----------
        **kwargs : keyword arguments
            Passed to ``compute_binary_map``

        """
        starttime = time_ini()

        for m in self.methods:
            m.detections = []
            m.fps = []
            m.bmaps = []

        print('Evaluating injections:')
        for i in Progressbar(range(self.n_injections)):
            x, y = self.list_xy[i]

            for m in self.methods:
                dets, fps, bmaps = compute_binary_map(
                    m.probmaps[i], m.thresholds, fwhm=self.dataset.fwhm,
                    injections=(x, y), **kwargs
                )
                m.detections.append(dets)
                m.fps.append(fps)
                m.bmaps.append(bmaps)

        timing(starttime)

    def plot_detmaps(self, i=None, thr=9, dpi=100,
                     axis=True, grid=False, vmin=-10, vmax='max',
                     plot_type="horiz"):
        """
        Plot the detection maps for one injection.

        Parameters
        ----------
        i : int or None, optional
            Index of the injection, between 0 and self.n_injections. If None,
            takes the 30st injection, or if there are less injections, the
            middle one.
        thr : int, optional
            Index of the threshold.
        dpi, axis, grid, vmin, vmax
            Passed to ``pp_subplots``
        plot_type : {"horiz" or "vert"}, optional
            Plot type.

            ``horiz``
                One row per algorithm (frame, probmap, binmap)
            ``vert``
                1 row for final frames, 1 row for probmaps and 1 row for binmaps

        """
        # input parameters
        if i is None:
            if len(self.list_xy) > 30:
                i = 30
            else:
                i = len(self.list_xy) // 2

        if vmax == 'max':
            # TODO: document this feature.
            vmax = np.concatenate([m.frames[i] for m in self.methods if
                                   hasattr(m, "frames") and
                                   len(m.frames) >= i]).max()/2

        # print information
        print('X,Y: {}'.format(self.list_xy[i]))
        print('dist: {:.3f}, flux: {:.3f}'.format(self.dists[i],
                                                  self.fluxes[i]))
        print()

        if plot_type in [1, "horiz"]:
            for m in self.methods:
                print('detection state: {} | false postives: {}'.format(
                    m.detections[i][thr], m.fps[i][thr]))
                labels = ('{} frame'.format(m.name), '{} S/Nmap'.format(m.name),
                          'Thresholded at {:.1f}'.format(m.thresholds[thr]))
                plot_frames((m.frames[i] if len(m.frames) >= i else
                            np.zeros((2, 2)), m.probmaps[i], m.bmaps[i][thr]),
                            label=labels, dpi=dpi, horsp=0.2, axis=axis,
                            grid=grid, cmap=['viridis', 'viridis', 'gray'])

        elif plot_type in [2, "vert"]:
            labels = tuple('{} frame'.format(m.name) for m in self.methods if
                           hasattr(m, "frames") and len(m.frames) >= i)
            plot_frames(tuple(m.frames[i] for m in self.methods if
                        hasattr(m, "frames") and len(m.frames) >= i),
                        dpi=dpi, label=labels, vmax=vmax, vmin=vmin, axis=axis,
                        grid=grid)

            plot_frames(tuple(m.probmaps[i] for m in self.methods), dpi=dpi,
                        label=tuple(['{} S/Nmap'.format(m.name) for m in
                                     self.methods]), axis=axis, grid=grid)

            for m in self.methods:
                msg = '{} detection: {}, FPs: {}'
                print(msg.format(m.name, m.detections[i][thr], m.fps[i][thr]))

            labels = tuple('Thresholded at {:.1f}'.format(m.thresholds[thr])
                           for m in self.methods)
            plot_frames(tuple(m.bmaps[i][thr] for m in self.methods),
                        dpi=dpi, label=labels, axis=axis, grid=grid,
                        colorbar=False, cmap='bone')
        else:
            raise ValueError("`plot_type` unknown")

    def plot_roc_curves(self, dpi=100, figsize=(5, 5), xmin=None, xmax=None,
                        ymin=-0.05, ymax=1.02, xlog=True, label_skip_one=False,
                        legend_loc='lower right', legend_size=6,
                        show_data_labels=True, hide_overlap_label=True,
                        label_gap=(0, -0.028), save_plot=False, label_params={},
                        line_params={}, marker_params={}, verbose=True):
        """
        Parameters
        ----------


        Returns
        -------
        None, but modifies `methods`: adds .tpr and .mean_fps attributes

        Notes
        -----
        # TODO: load `roc_injections` and `roc_tprfps` from file (`load_res`)
        # TODO: print flux distro information (is it actually stored in inj?
        What to do with functions, do they pickle?)
        # TODO: hardcoded `methodconf`?

        """
        labelskw = dict(alpha=1, fontsize=5.5, weight="bold", rotation=0,
                        annotation_clip=True)
        linekw = dict(alpha=0.2)
        markerkw = dict(alpha=0.5, ms=3)
        labelskw.update(label_params)
        linekw.update(line_params)
        markerkw.update(marker_params)
        n_thresholds = len(self.methods[0].thresholds)

        if verbose:
            print('{} injections'.format(self.n_injections))
            # print('Flux distro : {} [{}:{}]'.format(roc_injections.flux_distribution,
            #                                         roc_injections.fluxp1, roc_injections.fluxp2))
            print('Annulus from {} to {} pixels'.format(self.inrad,
                                                        self.outrad))

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        if not isinstance(label_skip_one, (list, tuple)):
            label_skip_one = [label_skip_one]*len(self.methods)
        labels = []

        # methodconf = {"CADI": dict(color="#d62728", symbol="^"),
        #              "PCA": dict(color="#ff7f0e", symbol="X"),
        #              "LLSG": dict(color="#2ca02c", symbol="P"),
        #              "SODIRF": dict(color="#9467bd", symbol="s"),
        #              "SODINN": dict(color="#1f77b4", symbol="p"),
        #              "SODINN-pw": dict(color="#1f77b4", symbol="p")
        #             }  # maps m.name to plot style

        for i, m in enumerate(self.methods):

            if not hasattr(m, "detections") or not hasattr(m, "fps"):
                raise AttributeError("method #{} has no detections/fps. Run"
                                     "`compute_tpr_fps` first.".format(i))

            m.tpr = np.zeros((n_thresholds))
            m.mean_fps = np.zeros((n_thresholds))

            for j in range(n_thresholds):
                m.tpr[j] = np.asarray(m.detections)[:, j].tolist().count(1) / \
                           self.n_injections
                m.mean_fps[j] = np.asarray(m.fps)[:, j].mean()

            plt.plot(m.mean_fps, m.tpr, '--', color=m.color, **linekw)
            plt.plot(m.mean_fps, m.tpr, m.symbol, label=m.name, color=m.color,
                     **markerkw)

            if show_data_labels:
                if label_skip_one[i]:
                    lab_x = m.mean_fps[1::2]
                    lab_y = m.tpr[1::2]
                    thr = m.thresholds[1::2]
                else:
                    lab_x = m.mean_fps
                    lab_y = m.tpr
                    thr = m.thresholds

                for i, xy in enumerate(zip(lab_x + label_gap[0],
                                           lab_y + label_gap[1])):
                    labels.append(ax.annotate('{:.2f}'.format(thr[i]),
                                  xy=xy, xycoords='data', color=m.color,
                                              **labelskw))
                    # TODO: reverse order of `self.methods` for better annot. z-index?

        plt.legend(loc=legend_loc, prop={'size': legend_size})
        if xlog:
            ax.set_xscale("symlog")
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.xlim(xmin=xmin, xmax=xmax)
        plt.ylabel('TPR')
        plt.xlabel('Full-frame mean FPs')
        plt.grid(alpha=0.4)

        if show_data_labels:
            mask = np.zeros(fig.canvas.get_width_height(), bool)

            fig.canvas.draw()

            for label in labels:
                bbox = label.get_window_extent()
                negpad = -2
                x0 = int(bbox.x0) + negpad
                x1 = int(np.ceil(bbox.x1)) + negpad
                y0 = int(bbox.y0) + negpad
                y1 = int(np.ceil(bbox.y1)) + negpad

                s = np.s_[x0:x1, y0:y1]
                if np.any(mask[s]):
                    if hide_overlap_label:
                        label.set_visible(False)
                else:
                    mask[s] = True

        if save_plot:
            if isinstance(save_plot, str):
                plt.savefig(save_plot, dpi=dpi, bbox_inches='tight')
            else:
                plt.savefig('roc_curve.pdf', dpi=dpi, bbox_inches='tight')


def compute_binary_map(frame, thresholds, injections, fwhm, npix=1,
                       overlap_threshold=0.7, max_blob_fact=2, plot=False,
                       debug=False):
    """
    Take a list of ``thresholds``, create binary maps and counts detections/fps.
    A blob which is "too big" is split into apertures, and every aperture adds
    one 'false positive'.

    Parameters
    ----------
    frame : numpy ndarray
        Detection map.
    thresholds : list or numpy ndarray
        List of thresholds (detection criteria).
    injections : tuple, list of tuples
        Coordinates (x,y) of the injected companions. Also accepts 1d/2d
        ndarrays.
    fwhm : float
        FWHM, used for obtaining the size of the circular aperture centered at
        the injection position (and measuring the overlapping with found blobs).
        The circular aperture has 2 * FWHM in diameter.
    npix : int, optional
        The number of connected pixels, each greater than the given threshold,
        that an object must have to be detected. ``npix`` must be a positive
        integer. Passed to ``detect_sources`` function from ``photutils``.
    overlap_threshold : float
        Percentage of overlap a blob has to have with the aperture around an
        injection.
    max_blob_fact : float
        Maximum size of a blob (in multiples of the resolution element) before
        it is considered as "too big" (= non-detection).
    plot : bool, optional
        If True, a final resulting plot summarizing the results will be shown.
    debug : bool, optional
        For showing optional information.

    Returns
    -------
    list_detections : list of int
        List of detection count for each threshold.
    list_fps : list of int
        List of false positives count for each threshold.
    list_binmaps : list of 2d ndarray
        List of binary maps: detection maps thresholded for each threshold
        value.

    """
    def _overlap_injection_blob(injection, fwhm, blob_mask):
        """
        Parameters
        ----------
        injection: tuple (y,x)
        fwhm : float
        blob_mask : 2d bool ndarray

        Returns
        -------
        overlap_fact : float between 0 and 1
            Percentage of the area overlap. If the blob is smaller than the
            resolution element, this is ``intersection_area / blob_area``,
            otherwise ``intersection_area / resolution_element``.

        """
        if len(injections[0]) > 0:
            injection_mask = get_circle(np.ones_like(blob_mask), radius=fwhm,
                                        cy=injection[1], cx=injection[0],
                                        mode="mask")
        else:
            injection_mask = np.zeros_like(blob_mask)
        intersection = injection_mask & blob_mask
        smallest_area = min(blob_mask.sum(), injection_mask.sum())
        return intersection.sum() / smallest_area

    # --------------------------------------------------------------------------
    list_detections = []
    list_fps = []
    list_binmaps = []
    sizey, sizex = frame.shape
    cy, cx = frame_center(frame)
    reselem_mask = get_circle(frame, radius=fwhm, cy=cy, cx=cx, mode="val")
    npix_circ_aperture = reselem_mask.shape[0]

    # normalize injections: accepts combinations of 1d/2d and tuple/list/array.
    injections = np.asarray(injections)
    if injections.ndim == 1:
        injections = np.array([injections])

    for ithr, threshold in enumerate(thresholds):
        if debug:
            print("\nprocessing threshold #{}: {}".format(ithr + 1, threshold))

        segments = detect_sources(frame, threshold, npix, connectivity=4)
        binmap = (segments.data != 0)

        if debug:
            plot_frames((segments.data, binmap), cmap=('tab20b', 'binary'),
                        circle=tuple(tuple(xy) for xy in injections),
                        circle_radius=fwhm, circle_alpha=0.6,
                        label=("segmentation map", "binary map"))

        detections = 0
        fps = 0

        for segment in segments.segments:
            label = segment.label
            blob_mask = (segments.data == label)
            blob_area = segment.area

            if debug:
                lab = "blob #{}, area={}px**2".format(label, blob_area)
                plot_frames(blob_mask, circle_radius=fwhm, circle_alpha=0.6,
                            circle=tuple(tuple(xy) for xy in injections),
                            cmap='binary', label_size=8, label=lab,
                            size_factor=3)

            for iinj, injection in enumerate(injections):
                if len(injections[0]) > 0:  # checking injections is not empty
                    if injection[0] > sizex or injection[1] > sizey:
                        raise ValueError("Wrong coordinates in `injections`")

                    if debug:
                        print("\ttesting injection #{} at {}".format(iinj + 1,
                                                                     injection))

                if blob_area > max_blob_fact * npix_circ_aperture:
                    number_of_apertures_in_blob = blob_area / npix_circ_aperture
                    fps += number_of_apertures_in_blob  # float, rounded at end
                    if debug:
                        print("\tblob is too big (+{:.0f} fps)"
                              "".format(number_of_apertures_in_blob))
                        print("\tskipping all other injections")
                    # continue with next blob, do not check other injections
                    break

                overlap = _overlap_injection_blob(injection, fwhm, blob_mask)
                if overlap > overlap_threshold:
                    if debug:
                        print("\toverlap of {}! (+1 detection)"
                              "".format(overlap))

                    detections += 1
                    # continue with next blob, do not check other injections
                    break

                if debug:
                    print("\toverlap of {} -> do nothing".format(overlap))

            else:
                if debug:
                    print("\tdid not find a matching injection for this "
                          "blob (+1 fps)")
                fps += 1

        if debug:
            print("done with threshold #{}".format(ithr))
            print("result: {} detections, {} fps".format(detections, fps))

        fps = np.round(fps).astype(int).item()  # -> python `int`

        list_detections.append(detections)
        list_binmaps.append(binmap)
        list_fps.append(fps)

    if plot:
        labs = tuple(str(det) + ' detections' + '\n' + str(fps) +
                     ' false positives' for det, fps in zip(list_detections,
                                                            list_fps))
        if len(injections[0]) > 0:
            circles = tuple(tuple(xy) for xy in injections)
        else:
            circles = None
        plot_frames(tuple(list_binmaps), title='Final binary maps', label=labs,
                    label_size=8, cmap='binary', circle_alpha=0.8,
                    circle=circles, circle_radius=fwhm,
                    circle_color='deepskyblue', axis=False)

    return list_detections, list_fps, list_binmaps


def _create_synt_cube(cube, psf, ang, plsc, dist, flux, theta=None,
                      verbose=False):
    """
    """
    centy_fr, centx_fr = frame_center(cube[0])
    if theta is None:
        np.random.seed()
        theta = np.random.randint(0, 360)

    posy = dist * np.sin(np.deg2rad(theta)) + centy_fr
    posx = dist * np.cos(np.deg2rad(theta)) + centx_fr
    if verbose:
        print('Theta:', theta)
        print('Flux_inj:', flux)
    cubefc = cube_inject_companions(cube, psf, ang, flevel=flux, plsc=plsc,
                                    rad_dists=[dist], n_branches=1, theta=theta,
                                    verbose=verbose)
    return cubefc, posx, posy
