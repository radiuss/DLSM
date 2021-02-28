__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2021, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import os
from enum import Enum
import numpy as np
import shutil


NORM_CORR = 0.95
SD_MIN = 0.01


def clear_folder(path):
    """ Clear folder

    :param path: path to the folder
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))


class ImageModality(Enum):
    """ Modality of the processed image

    """
    OPTICAL = 0
    RADAR = 1
    DEM = 2

    def __int__(self):
        return self.value

    def __str__(self):
        if self.value == ImageModality.OPTICAL.value:
            return 'optical'
        elif self.value == ImageModality.RADAR.value:
            return 'radar'
        elif self.value == ImageModality.DEM.value:
            return 'DEM'
        else:
            raise NotImplementedError

    @staticmethod
    def from_str(string):
        for mode in ImageModality:
            if string == str(mode):
                return mode
        raise ValueError("Cannot initialize modality from string {}".format(string))


def get_cov_elements_by_sm_name(sm_value, sm_name):
    """ Get elements of the translation vector covariance matrix

    :param sm_value: SM value map as numpy array os size N by N by C px,
                     where N by N is search zone, and C is number of SM value channels
                     C = 5 for DLSM
    :param sm_name: name of SM. Supports only "DLSM" model
    :return: sigma_t, sigma_s, k_ts that form covariance matrix as
             [sigma_t ** 2, sigma_t * sigma_s * k_ts
              sigma_t * sigma_s * k_ts, sigma_s ** 2]
    """
    if 'DLSM' in sm_name:
        sigma_t = sm_value[..., 2] + SD_MIN
        sigma_s = sm_value[..., 3] + SD_MIN
        k_ts = sm_value[..., 4] * NORM_CORR
    else:
        raise ValueError("Unknown SM {}".format(sm_name))
    return sigma_t, sigma_s, k_ts


def get_score_by_sm_name(sm_value, sm_name):
    """ Given SM name, output SM score value

    :param sm_value: sm_value: SM value map as numpy array os size N by N by C px,
                     where N by N is search zone, and C is number of SM value channels. C = 5 for DLSM
    :param sm_name: name of SM. Supports only "DLSM" model
    :return: SM scalar score
    """
    if 'DLSM' in sm_name:
        sigma_t, sigma_s, k_ts = get_cov_elements_by_sm_name(sm_value, sm_name)
        var_t = sigma_t ** 2
        var_s = sigma_s ** 2
        return (var_t * var_s * (1. - k_ts ** 2)) ** 0.25
    else:
        print("Unknown sm {}. Returning default sm value".format(sm_name))
        return sm_value[..., 0]


def prepare_input_DLSM(fr_ri_norm, fr_ti_norm):
    """ Given RI and TI fragments, form input to DLSM model

    :param fr_ri_norm: normalized RI fragment (32 by 32 px for DLSM)
    :param fr_ti_norm:  normalized TI fragment (32 by 32 px for DLSM)
    :return: 32 by 32 by 2 array
    """
    num_channels = 2
    size_ti = fr_ri_norm.shape[0]
    data = np.zeros((num_channels, size_ti, size_ti), dtype='float32')
    data[0, :, :] = fr_ri_norm
    data[1, :, :] = fr_ti_norm
    return data


def is_valid_fragment(fr, fr_std=None, min_std=1):
    """ check that fragment can be used with DLSM model

    :param fr: image fragment
    :param fr_std: fragment std. Could be provided outside
    :param min_std: minimum image fragment STD to consider it valid
    :return: True if fragment is valid, False othervwse
    """
    valid_percentage = 0.25  # fragment should have less than 100. valid_percentage percents of zero values
                             # zero values are considered as data absence
    if fr_std is None:
        fr_std = fr.std()
    if np.sum(fr == 0) > valid_percentage * fr.size or fr_std < min_std:
        return False
    return True


def get_norm_fragment(fr, min_std=1):
    """ normalize image fragment for DLSM model.

    :param fr: image fragment (32 by 32 px for DLSM)
    :param min_std: minimum image fragment STD to consider it valid
    :return: normalized fragment or None if fragment is non-valid
    """
    fr_std = fr.std()
    if not is_valid_fragment(fr, fr_std=fr_std, min_std=min_std):
        return None
    fr_mean = fr.mean()
    return (fr - fr_mean) / fr_std
