import nibabel as nib
import numpy as np
import os
import scipy.ndimage.morphology
import shutil

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import ProbabilisticDirectionGetter

from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.dti import TensorModel, fractional_anisotropy

from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamlinespeed import length


# AnalysisContext documentation: https://docs.qmenta.com/sdk/sdk.html
def run(context):

    ####################################################
    # Get the path to input files  and other parameter #
    ####################################################
    analysis_data = context.fetch_analysis_data()
    settings = analysis_data['settings']
    postprocessing = settings['postprocessing']
    dataset = settings['dataset']

    if dataset == "HCPL":
        dwi_file_handle = context.get_files('input', modality='HARDI')[0]
        dwi_file_path = dwi_file_handle.download('/root/')

        bvalues_file_handle = context.get_files(
            'input', reg_expression='.*prep.bvalues.hcpl.txt')[0]
        bvalues_file_path = bvalues_file_handle.download('/root/')
        bvecs_file_handle = context.get_files(
            'input', reg_expression='.*prep.gradients.hcpl.txt')[0]
        bvecs_file_path = bvecs_file_handle.download('/root/')
    elif dataset == "DSI":
        dwi_file_handle = context.get_files('input', modality='DSI')[0]
        dwi_file_path = dwi_file_handle.download('/root/')
        bvalues_file_handle = context.get_files(
            'input', reg_expression='.*prep.bvalues.txt')[0]
        bvalues_file_path = bvalues_file_handle.download('/root/')
        bvecs_file_handle = context.get_files(
            'input', reg_expression='.*prep.gradients.txt')[0]
        bvecs_file_path = bvecs_file_handle.download('/root/')
    else:
            context.set_progress(message='Wrong dataset parameter')

    inject_file_handle = context.get_files(
        'input', reg_expression='.*prep.inject.nii.gz')[0]
    inject_file_path = inject_file_handle.download('/root/')

    VUMC_ROIs_file_handle = context.get_files(
        'input', reg_expression='.*VUMC_ROIs.nii.gz')[0]
    VUMC_ROIs_file_path = VUMC_ROIs_file_handle.download('/root/')

    ###############################
    # _____ _____ _______     __  #
    # |  __ \_   _|  __ \ \   / / #
    # | |  | || | | |__) \ \_/ /  #
    # | |  | || | |  ___/ \   /   #
    # | |__| || |_| |      | |    #
    # |_____/_____|_|      |_|    #
    #                             #
    ###############################

    ########################################################################################
    #  _______             _          __  __   _______             _     __                #
    # |__   __|           | |        |  \/  | |__   __|           | |   / _|               #
    #    | |_ __ __ _  ___| | ___   _| \  / | ___| |_ __ __ _  ___| | _| |_ __ _  ___ ___  #
    #    | | '__/ _` |/ __| |/ / | | | |\/| |/ __| | '__/ _` |/ __| |/ /  _/ _` |/ __/ _ \ #
    #    | | | | (_| | (__|   <| |_| | |  | | (__| | | | (_| | (__|   <| || (_| | (_|  __/ #
    #    |_|_|  \__,_|\___|_|\_\\__, |_|  |_|\___|_|_|  \__,_|\___|_|\_\_| \__,_|\___\___| #
    #                            __/ |                                                     #
    #                           |___/                                                      #
    #                                                                                      #
    #                                                                                      #
    #                               IronTract Team                                         #
    ########################################################################################

    #################
    # Load the data #
    #################
    dwi_img = nib.load(dwi_file_path)
    bvals, bvecs = read_bvals_bvecs(bvalues_file_path,
                                    bvecs_file_path)
    gtab = gradient_table(bvals, bvecs)

    ############################################
    # Extract the brain mask from the b0 image #
    ############################################
    _, brain_mask = median_otsu(dwi_img.get_data()[:, :, :, 0],
                                median_radius=2, numpass=1)

    ##################################################################
    # Fit the tensor model and compute the fractional anisotropy map #
    ##################################################################
    context.set_progress(message='Processing voxel-wise DTI metrics.')
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(dwi_img.get_data(), mask=brain_mask)
    FA = fractional_anisotropy(tenfit.evals)
    stopping_criterion = ThresholdStoppingCriterion(FA, 0.2)

    sphere = get_sphere("repulsion724")
    seed_mask_img = nib.load(inject_file_path)
    affine = seed_mask_img.affine
    seeds = utils.random_seeds_from_mask(seed_mask_img.get_data(),
                                         affine,
                                         seed_count_per_voxel=True,
                                         seeds_count=5000)

    if dataset == "HCPL":
        ################################################
        # Compute Fiber Orientation Distribution (CSD) #
        ################################################
        context.set_progress(message='Processing voxel-wise FOD estimation.')

        response, _ = auto_response_ssst(gtab, dwi_img.get_data(),
                                         roi_radii=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
        csd_fit = csd_model.fit(dwi_img.get_data(), mask=brain_mask)
        shm = csd_fit.shm_coeff

        prob_dg = ProbabilisticDirectionGetter.from_shcoeff(shm,
                                                            max_angle=20.,
                                                            sphere=sphere,
                                                            pmf_threshold=0.1)
    elif dataset == "DSI":
        context.set_progress(message='Processing voxel-wise DSI estimation.')
        dsmodel = DiffusionSpectrumModel(gtab)
        dsfit = dsmodel.fit(dwi_img.get_data())
        ODFs = dsfit.odf(sphere)
        prob_dg = ProbabilisticDirectionGetter.from_pmf(ODFs,
                                                        max_angle=20.,
                                                        sphere=sphere,
                                                        pmf_threshold=0.01)

    ###########################################
    # Compute DIPY Probabilistic Tractography #
    ###########################################
    context.set_progress(message='Processing tractography.')
    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                         affine, step_size=.2, max_cross=1)
    streamlines = Streamlines(streamline_generator)
    # sft = StatefulTractogram(streamlines, seed_mask_img, Space.RASMM)
    # streamlines_file_path = "/root/streamlines.trk"
    # save_trk(sft, streamlines_file_path)

    ###########################################################################
    # Compute 3D volumes for the IronTract Challenge. For 'EPFL', we only     #
    # keep streamlines with length > 1mm. We compute the visitation  count    #
    # image and apply a small gaussian smoothing. The gaussian smoothing      #
    # is especially usefull to increase voxel coverage of deterministic       #
    # algorithms. The log of the smoothed visitation count map is then        #
    # iteratively thresholded producing 200 volumes/operation points.         #
    # For VUMC, additional streamline filtering is done using anatomical      #
    # priors (keeping only streamlines that intersect with at least one ROI). #
    ###########################################################################
    if postprocessing in ["EPFL", "ALL"]:
        context.set_progress(message='Processing density map (EPFL)')
        volume_folder = "/root/vol_epfl"
        output_epfl_zip_file_path = "/root/TrackyMcTrackface_EPFL_example.zip"
        os.mkdir(volume_folder)
        lengths = length(streamlines)
        streamlines = streamlines[lengths > 1]
        density = utils.density_map(streamlines, affine, seed_mask_img.shape)
        density = scipy.ndimage.gaussian_filter(density.astype("float32"), 0.5)

        log_density = np.log10(density + 1)
        max_density = np.max(log_density)
        for i, t in enumerate(np.arange(0, max_density, max_density / 200)):
            nbr = str(i)
            nbr = nbr.zfill(3)
            mask = log_density >= t
            vol_filename = os.path.join(volume_folder,
                                        "vol" + nbr + "_t" + str(t) + ".nii.gz")
            nib.Nifti1Image(mask.astype("int32"), affine,
                            seed_mask_img.header).to_filename(vol_filename)
        shutil.make_archive(output_epfl_zip_file_path[:-4], 'zip', volume_folder)

    if postprocessing in ["VUMC", "ALL"]:
        context.set_progress(message='Processing density map (VUMC)')
        ROIs_img = nib.load(VUMC_ROIs_file_path)
        volume_folder = "/root/vol_vumc"
        output_vumc_zip_file_path = "/root/TrackyMcTrackface_VUMC_example.zip"
        os.mkdir(volume_folder)
        lengths = length(streamlines)
        streamlines = streamlines[lengths > 1]

        rois = ROIs_img.get_fdata().astype(int)
        _, grouping = utils.connectivity_matrix(streamlines, affine, rois,
                                                inclusive=True,
                                                return_mapping=True,
                                                mapping_as_streamlines=False)
        streamlines = streamlines[grouping[(0, 1)]]

        density = utils.density_map(streamlines, affine, seed_mask_img.shape)
        density = scipy.ndimage.gaussian_filter(density.astype("float32"), 0.5)

        log_density = np.log10(density + 1)
        max_density = np.max(log_density)
        for i, t in enumerate(np.arange(0, max_density, max_density / 200)):
            nbr = str(i)
            nbr = nbr.zfill(3)
            mask = log_density >= t
            vol_filename = os.path.join(volume_folder,
                                        "vol" + nbr + "_t" + str(t) + ".nii.gz")
            nib.Nifti1Image(mask.astype("int32"), affine,
                            seed_mask_img.header).to_filename(vol_filename)
        shutil.make_archive(output_vumc_zip_file_path[:-4], 'zip', volume_folder)

    ###################
    # Upload the data #
    ###################
    context.set_progress(message='Uploading results...')
    #context.upload_file(fa_file_path, 'fa.nii.gz')
    # context.upload_file(fod_file_path, 'fod.nii.gz')
    # context.upload_file(streamlines_file_path, 'streamlines.trk')
    if postprocessing in ["EPFL", "ALL"]:
        context.upload_file(output_epfl_zip_file_path,
                            'TrackyMcTrackface_' + dataset +'_EPFL.zip')
    if postprocessing in ["VUMC", "ALL"]:
        context.upload_file(output_vumc_zip_file_path,
                            'TrackyMcTrackface_' + dataset +'_VUMC.zip')
