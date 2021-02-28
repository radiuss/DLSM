__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2021, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import os
import numpy as np
import h5py
import math
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.io as sio
from scipy.optimize import least_squares
from scipy import interpolate

from DLSM.utils import ImageModality, get_norm_fragment, prepare_input_DLSM,\
                       get_score_by_sm_name, get_cov_elements_by_sm_name


DPI = 300
NON_VALID_SM = -10000
SOTA_NAMES_CONVERTOR = {"dMIND": "MIND", "dSIFT": "SIFT-OCT", 'MIgrad': 'GMI'}
DEFAULT_FIXED_DISPLAY = '--k'
FIXED_DISPLAY = {'DLSM': '.-k',
                 'NCC': '.-r',
                 'PC': '.-g',
                 'MI': '.-b',
                 'MIND': '.-m',
                 'SIFT-OCT': '.--c',
                 'HOG': '.--m',
                 'HOPC': '.--r',
                 'GMI': '.--g',
                 'L2Net': '.--b'}


class StatisticsCalculator:
    """ Class for comparing SOTA SM and DLSM mode: AUC, ROC analysis, localization, visual analysis

    """

    def __init__(self, test_path, path_sota, sota_algorithms=None, verbose=0):
        """ StatisticsCalculator initialization

        :param test_path: path to test RI/TI pairs as *.mat file
        :param path_sota:  path to SOTA results as *.mat file associated with test data in test_path
        :param sota_algorithms: list of SOTA algorithm name for comparison
                                Available SOTA algorithms: NCC, MI, PC, SIFT-OCT, MIND, HOG, HOPC, GMI, L2Net
        :param verbose: if verbose = 1 additional information about test data will be displayed
        """
        sota_algorithms = ['MIND'] if sota_algorithms is None else sota_algorithms
        self.verbose = verbose

        # read RI and TI fragments
        f = sio.loadmat(test_path, mat_dtype=True)
        self.fr_ri_all = f['frRI']
        self.fr_ti_all = f['frTI']
        self.n_fragments = self.fr_ri_all.shape[2]
        if verbose == 1:
            print("Found {} test samples".format(self.n_fragments))

        self.n_ri = self.fr_ri_all.shape[0]
        self.n_ti = self.fr_ti_all.shape[0]
        self.n_bound = int((self.n_ri - self.n_ti) / 2)

        # read true-false correspondences index and modality
        self.is_similar = np.zeros((self.n_fragments,), dtype=int)
        self.modality = np.zeros((self.n_fragments, 2), dtype=ImageModality)

        for i in range(self.n_fragments):
            self.is_similar[i] = f['frDescr'][i]['isSimilar']
            self.modality[i, 0] = ImageModality.from_str(f['frDescr'][i]['modeRI'])
            self.modality[i, 1] = ImageModality.from_str(f['frDescr'][i]['modeTI'])

        # initialize SOTA algorithms
        f_sota = h5py.File(path_sota, 'r')
        n_sota_algs = f_sota['nameAlg'].shape[0]
        self.sm_sota = np.array(f_sota['sm'], dtype='float32')
        self.sm_sota_shift = np.array(f_sota['smShift'], dtype='float32')
        self.shift_gt = np.array(f_sota['dijShift'], dtype='float32').T
        self.search_zone = (self.sm_sota.shape[1] - 1) // 2

        self.alg_name = []
        self.alg_index = []
        for p in range(n_sota_algs):
            alg_name = f_sota[f_sota['nameAlg'][p, 0]][()].tostring().decode(encoding='UTF-16')
            alg_name_converted = SOTA_NAMES_CONVERTOR[alg_name] if alg_name in SOTA_NAMES_CONVERTOR else alg_name
            if alg_name_converted in sota_algorithms:
                number_of_valid = np.sum(self.sm_sota[p, self.search_zone, self.search_zone, :] > NON_VALID_SM)
                if number_of_valid > 0:
                    self.alg_name.append(alg_name_converted)
                    self.alg_index.append(p)

        min_sm_value = np.min(self.sm_sota[self.alg_index, self.search_zone, self.search_zone, :], axis=0)
        self.idx_valid = np.where(min_sm_value > NON_VALID_SM)[0]

        if verbose == 1:
            print("\nFound #{} processed samples".format(self.idx_valid.size))
            self.print_modality_split()

        print("Collecting statistics for SOTA")
        self.alg_stats = {}
        for idx_alg, p in enumerate(self.alg_index):
            print(" -- {}".format(self.alg_name[idx_alg]))
            sm_info = self.calc_auc_modality(self.is_similar[self.idx_valid],
                                             self.sm_sota[p, self.search_zone, self.search_zone, self.idx_valid],
                                             self.modality[self.idx_valid, :])
            self.alg_stats[self.alg_name[idx_alg]] = sm_info

    @staticmethod
    def display_score(score, sm_name, dxy_true, dxy_hat=None, di_hat=None, dj_hat=None, path=None):
        """ Display score map for an SM

        :param score: score map as numpy array
        :param sm_name: name of SM
        :param dxy_true: coordinates of the true shift between RI and TI
        :param dxy_hat: coordinates of the estimated shift between RI and TI
        :param di_hat: map of shift to the SM extrema in i direction. Only for DLSM
        :param dj_hat: map of shift to the SM extrema in j direction. Only for DLSM
        :param path: path to same images. Image will be displayed if path is None
        """
        w = int((score.shape[0] - 1) / 2.)
        figsize = (6.19, 5)
        edgecolor = (0.3, 0.3, 0.3)
        facecolor = 'g'
        head_width = 0.15
        arrow_width = 0.01
        marker_size = 10
        true_pos_marker = '.r'
        hat_pos_marker = '.b'

        f, ax_score = plt.subplots(1, 1, figsize=figsize)
        h = ax_score.imshow(score, aspect='equal', vmin=np.min(score), vmax=np.max(score),
                            extent=(-w - 0.5, w + 0.5, w + 0.5, -w - 0.5))
        f.colorbar(h, ax=ax_score)
        arrow = None
        if di_hat is not None and dj_hat is not None:
            for k1 in range(-w, w + 1):
                for k2 in range(-w, w + 1):
                    arrow = ax_score.arrow(k1, k2, di_hat[k2 + w, k1 + w], dj_hat[k2 + w, k1 + w], width=arrow_width,
                                           head_width=head_width, facecolor=facecolor, edgecolor=edgecolor)
        true_dot, = ax_score.plot(dxy_true[1], dxy_true[0], true_pos_marker, ms=marker_size)
        if dxy_hat is not None:
            hat_dot, = ax_score.plot(dxy_hat[1], dxy_hat[0], hat_pos_marker, ms=marker_size)
        ax_score.set_xlim(-w, w)
        ax_score.set_ylim(-w, w)
        ax_score.set_xlabel("Horizontal translation, pixels")
        ax_score.set_ylabel("Vertical translation, pixels")
        if dxy_hat is not None:
            if arrow is not None:
                ax_score.legend([true_dot, hat_dot, arrow], ['true position', 'estimated position', 'DLSM prediction'])
            else:
                ax_score.legend([true_dot, hat_dot], ['true position', 'estimated position'])
        else:
            ax_score.legend([true_dot], ['true position'])
        plt.title('Score map for SM {}'.format(sm_name))
        plt.tight_layout()

        if path is not None:
            plt.savefig(path, dpi=DPI)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def process_by_dlsm(patch_ri, patch_ti, dlsm):
        """ Process pair of fragments by DLSM model

        :param patch_ri: RI patch of size N by N pixel, where N >= 32
        :param patch_ti: TI patch of size 32 by 32 pixel
        :param dlsm: DLSM model
        :return: DLSM output of size (N-32) by (N-32) by 5 px, and DLSM score of size (N-32) by (N-32)
        """
        size_ti = int(dlsm.get_layer("input_2").get_output_at(0).get_shape()[2])  # get input patch size
        assert(size_ti == patch_ti.shape[0])
        size_ri = patch_ri.shape[0]
        assert(size_ri > size_ti and (size_ri - size_ti) % 2 == 0)
        w = (size_ri - size_ti) // 2
        data = np.zeros(((2 * w + 1) ** 2, 2, size_ti, size_ti), dtype='float32')
        fr_ti_norm = get_norm_fragment(patch_ti)
        if fr_ti_norm is None:
            print("Failed to init TI fragment")
            return

        p = 0
        for k1 in range(0, 2 * w + 1):
            for k2 in range(0, 2 * w + 1):
                fr_ri_norm = get_norm_fragment(patch_ri[k1:k1 + size_ti, k2:k2 + size_ti])
                if fr_ri_norm is None:
                    print("Failed to init RI fragment")
                    return

                data[p, ...] = prepare_input_DLSM(fr_ri_norm, fr_ti_norm)
                p += 1

        score_raw = dlsm.predict(data)
        score_scalar = get_score_by_sm_name(score_raw, "DLSM")
        score_scalar = score_scalar.reshape((2 * w + 1, 2 * w + 1))
        score_raw = score_raw.reshape((2 * w + 1, 2 * w + 1, score_raw.shape[-1]))
        return score_raw, score_scalar

    def save_single_pair(self, idx, path, model_sm=None, sm_name=None, w=5):
        """ Process a single RI-TI pair

        :param idx: pair index
        :param path: path for saving processing results
        :param model_sm: SM model. Only for DLSM
        :param sm_name: name of SM
        :param w: search zone. SM will be calculated in +- w shift range (in both directions)
        """
        di_hat = None
        dj_hat = None
        w = min(self.n_bound, w)

        k = idx
        if k > self.n_fragments:
            print(" -- Sample {} exceeds number {} of available samples".format(k, self.n_fragments))
            return
        if k not in self.idx_valid:
            print(" -- Sample {} in not in available samples".format(k))
            return

        # save RI and TI parches
        fr_ri = self.fr_ri_all[self.n_bound:self.n_bound + self.n_ti, self.n_bound:self.n_bound + self.n_ti, k]
        fr_ri = (fr_ri - np.min(fr_ri)) / (np.max(fr_ri) - np.min(fr_ri))
        fr_ri = (fr_ri * 255.).astype('uint8')
        fr_ti = self.fr_ti_all[:, :, k]
        fr_ti = (fr_ti - np.min(fr_ti)) / (np.max(fr_ti) - np.min(fr_ti))
        fr_ti = (fr_ti * 255.).astype('uint8')
        im = Image.fromarray(fr_ri)
        im.save(os.path.join(path, 'patch_{}_ri.png').format(k))
        im = Image.fromarray(fr_ti)
        im.save(os.path.join(path, 'patch_{}_ti.png').format(k))

        dxy = self.shift_gt[k, :]
        if model_sm is not None:
            patch_ti = self.fr_ti_all[:, :, k]
            size_ti = patch_ti.shape[0]
            patch_ri = self.fr_ri_all[:, :, k]
            f = interpolate.interp2d(np.arange(0, self.n_ri), np.arange(0, self.n_ri), patch_ri, kind='cubic')
            patch_ri = f(np.arange(0, self.n_ri) - dxy[1], np.arange(0, self.n_ri) - dxy[0])
            patch_ri = patch_ri[self.n_bound - w: self.n_bound + size_ti + w,
                                self.n_bound - w: self.n_bound + size_ti + w, ...]
            score_raw, score_scalar = StatisticsCalculator.process_by_dlsm(patch_ri, patch_ti, model_sm)
            di_hat = score_raw[:, :, 1]
            dj_hat = score_raw[:, :, 0]

        elif sm_name is not None:
            if sm_name not in self.alg_name:
                return
            alg_idx = self.alg_name.index(sm_name)
            p = self.alg_index[alg_idx]
            is_inv = self.alg_stats[sm_name]["is_inv"]
            w_sota = min(self.search_zone, w)
            i_min = self.search_zone - w_sota
            i_max = self.search_zone + w_sota + 1
            score_raw = is_inv * self.sm_sota_shift[p:p + 1, :, :, k][0, i_min:i_max, i_min:i_max]
            score_scalar = score_raw
        else:
            raise ValueError("SM name or SM model should be provided")

        fig_path = os.path.join(path, 'sm_profile_img_{}_{}.png'.format(k, sm_name))
        if os.path.exists(fig_path):
            return
        dxy_hat, _ = self.get_position_by_sm_name(score_raw, sm_name)
        self.display_score(score_scalar, sm_name, dxy, dxy_hat, di_hat=di_hat, dj_hat=dj_hat, path=fig_path)

    def print_modality_split(self):
        """ Print number of samples (RI-TI pairs) in each modality combination

        """
        if self.modality is not None:
            for mode1 in ImageModality:
                for mode2 in ImageModality:
                    if int(mode1) <= int(mode2):
                        x1 = np.logical_and(self.modality[self.idx_valid, 0] == mode1,
                                            self.modality[self.idx_valid, 1] == mode2)
                        x2 = np.logical_and(self.modality[self.idx_valid, 0] == mode2,
                                            self.modality[self.idx_valid, 1] == mode1)
                        number_of_samples = np.sum(np.logical_or(x1, x2))
                        if number_of_samples > 0:
                            print("  --  Case {}-to-{}. Found {} samples.".format(mode1, mode2, number_of_samples))

    def print_stats(self):
        """ Print collected AUC statistics

        """
        for alg_name, sm_info in self.alg_stats.items():
            print("\nAUC for {}".format(alg_name))
            for key in sorted(sm_info['auc']):
                print("  --  {}: {:.2f}%".format(key, sm_info['auc'][key]))

    def save_roc(self, out_dir=None, sm_names=None, split_fig=False):
        """ save collected ROC curves

        :param out_dir: output folder
        :param sm_names: list of SM names
        :param split_fig: True if each modality combination should be displayed in separate figure
        """
        sm_info = next(iter(self.alg_stats.values()))
        fig_key_list = sorted(list(sm_info['auc'].keys()))
        fontsize = 10
        line_width = 1
        if split_fig:
            fig_size = (6, 4)
            for fig_key in fig_key_list:
                f, ax = plt.subplots(1, 1, figsize=fig_size)
                for alg_name, sm_info in self.alg_stats.items():
                    if sm_names is not None and alg_name in sm_names:
                        display_type = FIXED_DISPLAY.get(alg_name, DEFAULT_FIXED_DISPLAY)
                        display_type = display_type[1:]
                        for key in sorted(sm_info['auc']):
                            if key == fig_key:
                                label = "AUC for {} = {:4.2f}".format(alg_name, sm_info['auc'][key])
                                ax.plot(sm_info['fpr'][key], sm_info['tpr'][key],
                                        display_type, lw=line_width, label=label)
                ax.set_title(fig_key)
                ax.grid()
                ax.legend(fontsize=fontsize)
                plt.xlabel("False Positive Rate", fontsize=fontsize)
                plt.ylabel("True Positive Rate", fontsize=fontsize)
                plt.tight_layout()
                if out_dir is not None:
                    fig_path = os.path.join(out_dir, "roc_{}.png".format(fig_key))
                    plt.savefig(fig_path, dpi=DPI)
                    plt.close()
                else:
                    plt.show()
        else:
            fig_size = (14, 7)
            n_cases = len(fig_key_list)
            n_x = int(math.floor(math.sqrt(n_cases)))
            n_y = int(math.ceil(n_cases / n_x))
            f, ax = plt.subplots(n_x, n_y, figsize=fig_size)
            for alg_name, sm_info in self.alg_stats.items():
                for k, key in enumerate(sorted(sm_info['auc'])):
                    label = "AUC for {} = {:4.2f}".format(alg_name, sm_info['auc'][key])
                    ax.ravel()[k].plot(sm_info['fpr'][key], sm_info['tpr'][key], label=label)
                    ax.ravel()[k].set_title(key)

            for k in range(n_cases):
                ax.ravel()[k].grid()
                ax.ravel()[k].legend(fontsize=fontsize)
                ax.ravel()[k].set_xlabel("False Positive Rate", fontsize=fontsize)
                ax.ravel()[k].set_ylabel("True Positive Rate", fontsize=fontsize)
            for k in range(n_cases, n_x * n_y):
                ax.ravel()[k].axis('off')
            plt.tight_layout()
            if out_dir is not None:
                fig_path = os.path.join(out_dir, "roc.png")
                plt.savefig(fig_path, dpi=DPI)
                plt.close()
            else:
                plt.show()

    def save_hist_by_sm_name(self, out_dir, sm_name, score_min=None, score_max=None):
        """ Save histogram for processed SMs

        :param out_dir: output folder
        :param sm_name: name of SM
        :param score_min: minimum score value. If None score_min will be calculated from data
        :param score_max: maximum score value. If None score_max will be calculated from data
        """
        if sm_name in self.alg_stats.keys():
            stats = self.alg_stats[sm_name]
            self.disp_hist(stats['labels'], stats['scores'], out_dir, sm_name, x_min=score_min, x_max=score_max)

    @staticmethod
    def disp_hist(labels, score, out_dir, sm_name, x_min=None, x_max=None):
        """ display ROC curve using SM score and labels

        :param labels: true/false correspondence labels. True corresponds to 1, false to 0
        :param score: SM score values
        :param out_dir: output folder
        :param sm_name: name of SM
        :param x_min: minimum score value. If None score_min will be calculated from data
        :param x_max: maximum score value. If None score_max will be calculated from data
        """
        disp_case = {'zero': (0, 'k', 'false correspondence'), 'alt': (1, 'r', 'true correspondence')}
        min_quantile = 0.001
        samples_in_bin = 50.
        min_number_of_bins = 100
        figure_size = (14, 7)

        if x_min is None:
            x_min = np.quantile(score, min_quantile)
        if x_max is None:
            x_max = np.quantile(score, 1 - min_quantile)
        score[score > x_max] = x_max
        score[score < x_min] = x_min

        bins = max(min_number_of_bins, int(score.size / samples_in_bin))
        plt.figure(figsize=figure_size)
        for key in disp_case:
            idx = np.where(labels == disp_case[key][0])[0]
            h, edge_bin = np.histogram(score[idx], range=(x_min, x_max), bins=bins, density=True)
            center_bin = (edge_bin[1:] + edge_bin[0:-1]) / 2
            plt.plot(center_bin, h, disp_case[key][1], label="{}".format(disp_case[key][2]))

        plt.xlim((x_min, x_max))

        plt.xlabel('Score')
        plt.ylabel('Probability distribution')
        plt.title('Score distribution for SM {}'.format(sm_name))
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if out_dir is not None:
            fig_path = os.path.join(out_dir, "hist_{}.png".format(sm_name))
            plt.savefig(fig_path, dpi=DPI)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def calc_auc_modality(labels, score, modality):
        """ Calculate AUC fo general case and each modality combination

        :param labels: true/false correspondence labels. True corresponds to 1, false to 0
        :param score: SM score values
        :param modality: vector of modalities for each sample
        :return: dictionary with fields 'auc', 'fpr', 'tpr', 'samples', 'labels', 'scores', 'is_inv'
                                        'auc' contains dictionary with keys corresponding to modality combination
                                              and value euqal to 100. * auc
                                         'fpr' is vector of false positive rate values
                                         'tpr' is vector of true positive rate values
                                         'samples' contains number samples
                                         'labels' contains labels
                                         'scores' contains SM score values
                                         'is_inv' is 1 is for an SM higher SM values correspond to higher similarity
                                                  and -1 if higher SM values correspond to lower similarity
        """
        min_number_of_samples = 10
        sm_info = {'auc': {}, 'fpr': {}, 'tpr': {}, 'samples': {}}

        idx = np.where(np.logical_and(labels >= 0, abs(score) < np.inf))[0]
        auc = roc_auc_score(labels[idx], score[idx])
        is_inv = 1
        if auc < 0.5:
            auc = 1. - auc
            is_inv = -1
        fpr, tpr, thr = roc_curve(labels[idx], is_inv * score[idx], pos_label=1)
        case_name = 'general'
        sm_info['auc'][case_name] = 100 * auc
        sm_info['fpr'][case_name] = fpr
        sm_info['tpr'][case_name] = tpr
        sm_info['samples'][case_name] = idx.size
        sm_info['labels'] = labels
        sm_info['scores'] = score
        sm_info['is_inv'] = is_inv

        if modality is not None:
            for mode1 in ImageModality:
                for mode2 in ImageModality:
                    if int(mode1) <= int(mode2):
                        x1 = np.logical_and(modality[:, 0] == mode1, modality[:, 1] == mode2)
                        x2 = np.logical_and(modality[:, 0] == mode2, modality[:, 1] == mode1)
                        x = np.logical_or(x1, x2)
                        x = np.logical_and(x, abs(score) < np.inf)
                        idx = np.where(np.logical_and(labels >= 0, x))[0]
                        if idx.size > min_number_of_samples:
                            auc = roc_auc_score(labels[idx], score[idx])
                            if is_inv == -1:
                                auc = 1. - auc
                            fpr, tpr, _ = roc_curve(labels[idx], is_inv * score[idx], pos_label=1)
                            case_name = "-".join((str(mode1), "to", str(mode2)))
                            sm_info['auc'][case_name] = 100 * auc
                            sm_info['fpr'][case_name] = fpr
                            sm_info['tpr'][case_name] = tpr
                            sm_info['samples'][case_name] = idx.size

        return sm_info

    @staticmethod
    def get_sm_maximum(sm_value, x_all=None, y_all=None):
        """ Find coordinates of SM value maximum

        :param sm_value: SM value map as numpy array os size N by N
        :param x_all: x coordinate associated with sm_value. Numpy array of the same size as sm_value
        :param y_all: y coordinate associated with sm_value. Numpy array of the same size as sm_value
        :return: coordinate of SM maximum
        """
        res = np.where(sm_value == np.max(sm_value))
        x_max = res[0][0]
        y_max = res[1][0]
        x_pos = x_all[x_max, y_max] if x_all is not None else x_max
        y_pos = y_all[x_max, y_max] if y_all is not None else y_max
        return x_pos, y_pos

    @staticmethod
    def fun_approx(x, y):
        w_appr = (y.shape[0] - 1)/2
        [k1, k2] = np.mgrid[-w_appr:w_appr + 1, -w_appr:w_appr + 1]
        return (x[0] * k1**2 + 2 * x[1] * k1 * k2 + x[2] * k2**2 + x[3] * k1 + x[4] * k2 + x[5] - y).ravel()

    @staticmethod
    def get_position_by_sm_name(sm_value, sm_name):
        """ Find coordinates of correspondence between RI and TI with subpixel accuracy

        :param sm_value: SM value map as numpy array os size N by N by C px,
                         where N by N is search zone, and C is number of SM value channels
                         C = 5 for DLSM and 1 for the rest of SMs
        :param sm_name: name of SM
        :return: vector of coordinates of correspondence, scalar SM value derived from sm_value
        """
        if 'DLSM' in sm_name and sm_value.ndim != 3:
            print("Wrong number of dimensions {} instead of 3".format(sm_value.ndim))
            return None, None
        if sm_value.shape[0] != sm_value.shape[1]:
            print("Wrong score shape ({}, {}). Not square".format(sm_value.shape[0], sm_value.shape[1]))
            return None, None
        if sm_value.shape[0] % 2 != 1:
            print("Wrong score shape ({}, {}). Not odd".format(sm_value.shape[0], sm_value.shape[1]))
            return None, None

        w = int((sm_value.shape[0] - 1.) / 2.)

        if 'DLSM' in sm_name:

            y_all, x_all = np.meshgrid(np.arange(0, 2 * w + 1, 0.1), np.arange(0, 2 * w + 1, 0.1))
            score_scalar = np.zeros_like(x_all)
            for x in range(0, 2 * w + 1):
                for y in range(0, 2 * w + 1):
                    x_pos = x
                    y_pos = y

                    x_pos_new = int(round(float(x_pos) + sm_value[x_pos, y_pos, 0]))
                    y_pos_new = int(round(float(y_pos) + sm_value[x_pos, y_pos, 1]))

                    if x_pos_new < 0 or y_pos_new < 0 or x_pos_new > 2 * w or y_pos_new > 2 * w:
                        continue

                    x_pos = x_pos_new
                    y_pos = y_pos_new
                    x_hat = float(x_pos) + sm_value[x_pos, y_pos, 0]
                    y_hat = float(y_pos) + sm_value[x_pos, y_pos, 1]

                    sd_x, sd_y, k_xy = get_cov_elements_by_sm_name(sm_value[x_pos, y_pos, :], sm_name)

                    det = sd_x ** 2 * sd_y ** 2 * (1. - k_xy ** 2)

                    form2 = (x_all - x_hat)**2 * sd_y**2 + (y_all - y_hat)**2 * sd_x**2 - \
                            2. * (x_all - x_hat) * (y_all - y_hat) * sd_x * sd_y * k_xy
                    log_lik = -0.5 * (form2 / det + np.log(det))
                    score_scalar += np.exp(log_lik)

            x_pos, y_pos = StatisticsCalculator.get_sm_maximum(score_scalar, x_all, y_all)
            dxy = np.array([x_pos - w, y_pos - w])
            dxy = np.maximum(dxy, -w)
            dxy = np.minimum(dxy, w)
            score_scalar = score_scalar[-1::-1, :]
            dxy = np.array([dxy[0], dxy[1], 0.])
        else:
            score_scalar = sm_value[:, :, 0] if sm_value.ndim == 3 else sm_value
            x_pos, y_pos = StatisticsCalculator.get_sm_maximum(score_scalar)

            if score_scalar[x_pos, y_pos] <= NON_VALID_SM:
                return None, None

            dxy = np.array([x_pos - w, y_pos - w, score_scalar[x_pos, y_pos]])

            w_appr = 1
            if w_appr <= x_pos <= 2 * w - w_appr and w_appr <= y_pos <= 2 * w - w_appr:
                m = score_scalar[x_pos - w_appr:x_pos + w_appr + 1, y_pos - w_appr:y_pos + w_appr + 1]
                m_min = np.min(m)
                m_max = np.max(m)
                x0 = np.array([0., 0., 0., 0., 0., 1])
                if m_min > NON_VALID_SM:
                    m = (m - m_min) / (m_max - m_min)
                    res = least_squares(StatisticsCalculator.fun_approx, x0, args=([m]))
                    a = res.x[0]
                    b = res.x[1]
                    c = res.x[2]
                    d = res.x[3]
                    e = res.x[4]
                    A = 2 * np.array([[a, b], [b, c]])
                    pos = np.linalg.solve(A, np.array([-d, -e]))
                    if abs(pos[0]) < 0.75 and abs(pos[1]) < 0.75:
                        dxy[0] = x_pos + pos[0] - w
                        dxy[1] = y_pos + pos[1] - w

        return dxy, score_scalar

    def process_testset_auc(self, dlsm_model, sm_name='DLSM'):
        """ Calculate AUC and ROC for DLSM model and append to self.alg_stats

        :param dlsm_model: DLSM CNN
        :param sm_name: SM name. Only 'DLSM' value is supported
        """
        n_fragments = self.idx_valid.size
        score = np.zeros((n_fragments, ))
        is_valid = np.zeros((n_fragments,))
        n_fragments_blk = min(n_fragments, 1000)

        delta_bound = self.n_bound
        size_ti = int(dlsm_model.get_layer("input_2").get_output_at(0).get_shape()[2])  # get input patch size
        size_ri = size_ti
        data = np.zeros((n_fragments_blk, 2, size_ti, size_ti), dtype='float32')
        for i_blk in range(0, n_fragments, n_fragments_blk):
            n_blk_this = min(n_fragments_blk, n_fragments - i_blk)
            for k_blk in range(n_blk_this):
                k = self.idx_valid[k_blk + i_blk]

                fr_ri_norm = get_norm_fragment(
                    self.fr_ri_all[delta_bound:delta_bound+size_ri, delta_bound:delta_bound+size_ri, k])
                if fr_ri_norm is None:
                    continue

                fr_ti_norm = get_norm_fragment(self.fr_ti_all[:, :, k])
                if fr_ti_norm is None:
                    continue

                data[k_blk, ...] = prepare_input_DLSM(fr_ri_norm, fr_ti_norm)
                is_valid[k_blk + i_blk] = 1

            out_blk = dlsm_model.predict(data)
            out_blk = out_blk[0:n_blk_this, ...]
            score_blk = get_score_by_sm_name(out_blk, sm_name)
            score[i_blk:i_blk + n_blk_this] = score_blk

        idx = np.where(is_valid > 0)[0]
        labels = self.is_similar[self.idx_valid[idx]]
        score = score[idx]
        modality = self.modality[self.idx_valid[idx], :]

        self.alg_stats[sm_name] = self.calc_auc_modality(labels, score, modality)

    def get_sample_pair(self, idx, dxy):
        """ Get pair of RI and TI fragments

        :param idx: index of fragment pair. Notice that true and false test samples are stored in interleaving order:
                    0 - true, 1 - false, 2 - true, etc.
        :param dxy: translation vector between RI and TI as list [row_shift, col_shift]
        :return: RI fragment, TI fragment
        """
        patch_ti = self.fr_ti_all[:, :, idx]
        size_ti = patch_ti.shape[0]
        patch_ri = self.fr_ri_all[:, :, idx]
        w = self.n_bound - 1 - int(math.ceil(max(map(abs, dxy))))
        f = interpolate.interp2d(np.arange(0, self.n_ri), np.arange(0, self.n_ri), patch_ri, kind='cubic')
        patch_ri = f(np.arange(0, self.n_ri) - dxy[1], np.arange(0, self.n_ri) - dxy[0])
        patch_ri = patch_ri[self.n_bound - w: self.n_bound + size_ti + w,
                            self.n_bound - w: self.n_bound + size_ti + w, ...]

        return patch_ri, patch_ti,
