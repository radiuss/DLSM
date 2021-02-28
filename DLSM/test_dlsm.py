__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2021, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import sys
import argparse
import os
from keras.models import load_model

from DLSM.utils import clear_folder
from DLSM.collect_stats import StatisticsCalculator


SM_DLSM = 'DLSM'
SM_SOTA = ['NCC', 'PC', 'MI', 'MIND', 'SIFT-OCT', 'HOG', 'HOPC', 'GMI', 'L2Net']
SM_COMPARE = SM_SOTA + [SM_DLSM]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test DLSM model')
    parser.add_argument('modelPath', type=str, help='Path to the DLSM CNN')
    parser.add_argument('--testPath', required=True, type=str, help='Path to test set data')
    parser.add_argument('--sotaPath', required=True, type=str, help='Path to SOTA results for the test set data')
    parser.add_argument('--outDir', required=True, type=str, help='Path to the output folder')
    args = parser.parse_args()

    model_path = args.modelPath
    test_path = args.testPath
    sota_path = args.sotaPath
    out_path = args.outDir

    if not os.path.exists(model_path):
        print('Model does not exist at path {}'.format(model_path))
        sys.exit(0)

    if not os.path.exists(test_path):
        print('Test data does not exist at path {}'.format(test_path))
        sys.exit(0)
    if not os.path.exists(sota_path):
        print('Test/SOTA data does not exist at path {}'.format(sota_path))
        sys.exit(0)

    clear_folder(out_path)

    print("Testing DLSM model {}".format(model_path))
    print("  --  output folder: {}".format(out_path))

    try:
        dlsm_model = load_model(model_path, compile=False)
    except Exception as e:
        print(e)
        sys.exit(0)

    if dlsm_model.name != SM_DLSM:
        print('Unknown model type {}'.format(dlsm_model.name))
        sys.exit(0)
    print("Model successfully loaded")

    sc = StatisticsCalculator(test_path, sota_path, sota_algorithms=SM_SOTA, verbose=1)
    print("Evaluator successfully initialized")

    # ROC + AUC analysis
    sc.process_testset_auc(dlsm_model, sm_name=SM_DLSM)
    sc.print_stats()
    sc.save_roc(sm_names=SM_COMPARE, out_dir=out_path, split_fig=False)
    for sm_name in SM_COMPARE:
        sc.save_hist_by_sm_name(out_path, sm_name=sm_name)

    # SM map analysis. Localization for DLSM
    idx_slice_vec = [0, 18, 40, 148, 226, 398]  # indices from dataset at path args.testPath
    search_zone = 5
    for idx_slice in idx_slice_vec:
        for sm_name in SM_COMPARE:
            sc.save_single_pair(idx=idx_slice, path=out_path, sm_name=sm_name, w=search_zone,
                                model_sm=dlsm_model if sm_name == SM_DLSM else None)

    # Example of sample image pair processing with DLSM
    dxy_true = [1, 3]  # simulate shift by 1 pixel w.r.t. rows axis and 3 pixels w.r.t. columns axis
    patch_ri, patch_ti = sc.get_sample_pair(idx=0, dxy=dxy_true)
    score_raw, score_scalar = sc.process_by_dlsm(patch_ri, patch_ti, dlsm_model)
    di_hat = score_raw[:, :, 1]
    dj_hat = score_raw[:, :, 0]
    dxy_hat, _ = sc.get_position_by_sm_name(score_raw, SM_DLSM)
    sc.display_score(score_scalar, SM_DLSM, dxy_true, dxy_hat=dxy_hat, di_hat=di_hat, dj_hat=dj_hat)

    print("Testing completed")
