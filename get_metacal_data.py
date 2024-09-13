import argparse
import os
import json
import numpy as np
import multiprocessing as mp
import configs_cal
from eval_cal import get_cdf_params
from scipy.interpolate import interp1d


def get_args_parser():
    parser = argparse.ArgumentParser("Obtaining Meta-cal Data")
    parser.add_argument(
        "--scene",
        default="flower",
        help="name of LLFF scene out of {room, fern, flower, fortress, horns, leaves, orchids, trex}",
    )
    parser.add_argument(
        "--gin_config",
        default="configs/llff_flower.gin",
        help="name of Gin config file for pretrained FlipNeRF model",
    )
    parser.add_argument(
        "--test_model_dir",
        default="checkpoints/llff3/flower",
        help="directory containing model checkpoint for obtaining calibration curves",
    )
    parser.add_argument(
        "--data_split_file",
        default="llff_3_view_data_splits.json",
        help="name of JSON file with training and test splits",
    )
    parser.add_argument(
        "--output_dir",
        default="scenes/flower",
        help="directory where to save output",
    )
    parser.add_argument(
        "--num_procs",
        default=48,
        type=int,
        help="number of processes to use for multiprocessing",
    )

    return parser


def adjust_for_quantile(ps, cs):
    """
    Adjusts [ps] and [cs], so that all CDF values in [ps] are unique. Following
    quantile function F^−1_t(p) = inf{c : p ≤ F_t(c)}, for each [p] in [ps],
    the minimum [c] in [cs] such that F_t([c]) = [p] is picked.

    [ps]: **sorted** CDF values for color values in [cs]
    [cs]: **sorted** color values corresponding to CDF values in [ps]

    Returns: ([ps], [cs]) such that interp1d([ps], [cs],
    fill_value="extrapolate") provides the approximated quantile function
    """

    # [inds] are the indices of the first occurrences of the unique values in
    # [ps]. Thus, [inds] provides the indices of the quantiles corresponding to
    # CDF values in [ps]. See: https://numpy.org/doc/stable/reference/generated
    # numpy.unique.html
    (ps, inds) = np.unique(ps, return_index=True)
    return (ps, cs[inds])


def cdf_matrix(pi, mu, beta, c):
    """
    Returns the CDF values for the mixture of Laplacians distribution
    parametrized by mixture coefficients [pi], location parameters [mu], and
    scale parameters [beta] at the color values [c].

    [pi]: mixture coefficients of shape (image height, image width, # of
    mixture components) (e.g., (378, 504, 128))
    [mu]: of shape (image height, image width, # of mixture components, # of
    channels) (e.g., (378, 504, 128, 3))
    [beta]: of shape (image height, image width, # of mixture components, # of
    channels) (e.g., (378, 504, 128, 3))
    [c]: of shape (image height, image width, # of channels) (e.g., (378, 504,
    3))

    Returns: matrix of Laplacian mixture CDF values of shape (image height,
    image width, # of channels) (e.g., (378, 504, 3))
    """

    num_mix_components = pi.shape[-1]
    num_channels = c.shape[-1]

    # Reshape [pi] and [c] to match the shapes of [mu] and [beta].
    pi = np.stack((pi,) * num_channels, axis=-1)
    c = np.stack((c,) * num_mix_components, axis=-2)

    # Compute the cdf parametrized by [mu] and [beta] at the color [c].
    return np.sum(
        pi
        * ((1 / 2) + (1 / 2) * np.sign(c - mu) * (1 - np.exp(-np.abs(c - mu) / beta))),
        axis=-2,
    )


def get_emp_confs(ps):
    """
    Returns the empirical confidence levels ([hat_ps]) corresponding to the
    expected confidence levels [ps].

    [ps]: **sorted** CDF values of colors in calibration set
    For i = 1, ..., len(ps), hat_ps[i - 1] = |{p_t: p_t <= ps[i - 1] and p_t in
    ps}| / len(ps).

    Returns: the empirical confidences [hat_ps] corresponding to the expected
    confidence levels in [ps]
    """

    hat_ps = np.ones(len(ps))
    for ind in range(len(ps) - 1, 0, -1):
        p_front = ps[ind]
        p = ps[ind - 1]
        if p_front == p:
            hat_ps[ind - 1] = hat_ps[ind]
        else:
            hat_ps[ind - 1] = ind / len(ps)
    return hat_ps


def get_uncerts(mus, betas, pis):
    """
    Returns the uncalibrated uncertainty maps for the inferred images with
    mixtures of Laplacians parameterized by location parameters in [mus], scale
    parameters in [betas], and mixture coefficients in [pis].

    Note: the uncertainty is calculated as the average interquartile range over
    the RGB color channels

    [mus]: of shape (# images, image height, image width, # of mixture
    components, # of channels) (e.g., (..., 378, 504, 128, 3))
    [betas]: of shape (# of images, image height, image width, # of mixture
    components, # of channels) (e.g., (..., 378, 504, 128, 3))
    [pis]: mixture coefficients of shape (# of images, image height, image
    width, # of mixture components) (e.g., (..., 378, 504, 128))

    Returns: the uncalibrated uncertainty maps
    """

    global get_uncert
    num_imgs = mus.shape[0]
    img_height = mus.shape[1]
    img_width = mus.shape[2]
    num_mix_comps = mus.shape[-2]
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    uncerts = mp.Array("d", np.zeros(pis.shape[0]), lock=False)

    def cdf(x, mus, betas, pis):
        return np.sum(
            pis
            * (0.5 + 0.5 * np.sign(x - mus) * (1 - np.exp(-np.abs(x - mus) / betas))),
            axis=-1,
        )

    def get_uncert(px_ind):
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        cdf_r = lambda x: cdf(x, mu[:, 0], beta[:, 0], pi)
        cdf_g = lambda x: cdf(x, mu[:, 1], beta[:, 1], pi)
        cdf_b = lambda x: cdf(x, mu[:, 2], beta[:, 2], pi)

        xs = np.linspace(0, 1, 10).reshape((10, 1))
        ys = cdf_r(xs)
        xs = xs[:, 0]
        (ys, xs) = adjust_for_quantile(ys, xs)
        i_cdf_r = interp1d(ys, xs, fill_value="extrapolate")
        interquart_r = i_cdf_r(0.75) - i_cdf_r(0.25)

        xs = np.linspace(0, 1, 10).reshape((10, 1))
        ys = cdf_g(xs)
        xs = xs[:, 0]
        (ys, xs) = adjust_for_quantile(ys, xs)
        i_cdf_g = interp1d(ys, xs, fill_value="extrapolate")
        interquart_g = i_cdf_g(0.75) - i_cdf_g(0.25)

        xs = np.linspace(0, 1, 10).reshape((10, 1))
        ys = cdf_b(xs)
        xs = xs[:, 0]
        (ys, xs) = adjust_for_quantile(ys, xs)
        i_cdf_b = interp1d(ys, xs, fill_value="extrapolate")
        interquart_b = i_cdf_b(0.75) - i_cdf_b(0.25)

        uncerts[px_ind] = (interquart_r + interquart_g + interquart_b) / 3

    proc_pool = mp.Pool(args.num_procs)
    proc_pool.map(get_uncert, range(pis.shape[0]))
    proc_pool.close()
    proc_pool.join()

    return np.array(uncerts).reshape((num_imgs, img_height, img_width))


# Parse commandline arguments.
args = get_args_parser()
args = args.parse_args()

# Make output directory for storing data.
os.makedirs(args.output_dir, exist_ok=True)

# Get train and test indices for the specified scene.
all_data_splits = json.load(open(args.data_split_file))
scene_data_split = all_data_splits[args.scene]
train_inds = scene_data_split["train"]
test_inds = scene_data_split["test"]

# Load config file for the pretrained FlipNeRF model.
config = configs_cal.load_config(args.gin_config, save_config=True)

# Save CDF parameters for test data from FlipNeRF model trained on 3 views.
config.checkpoint_dir = args.test_model_dir
(preds_test, betas_test, mus_test, pis_test, gts_test) = get_cdf_params(
    config, test_inds
)

# Compute expected confidence levels.
p_r = []
p_g = []
p_b = []

for img_ind in range(len(test_inds)):
    expected_cdf_vals = cdf_matrix(
        pis_test[img_ind], mus_test[img_ind], betas_test[img_ind], gts_test[img_ind]
    )
    p_r = p_r + list(expected_cdf_vals[:, :, 0].flatten())
    p_g = p_g + list(expected_cdf_vals[:, :, 1].flatten())
    p_b = p_b + list(expected_cdf_vals[:, :, 2].flatten())

# Compute empirical confidence levels.
print("starting to calculate empirical confidences.")
p_r = np.sort(p_r)
p_g = np.sort(p_g)
p_b = np.sort(p_b)
hat_p_r = get_emp_confs(p_r)
hat_p_g = get_emp_confs(p_g)
hat_p_b = get_emp_confs(p_b)
print("stopped calculating empirical confidences.")

# Save data for training meta-calibrator.
np.save(args.output_dir + "/" + "p_r.npy", p_r)
np.save(args.output_dir + "/" + "p_g.npy", p_g)
np.save(args.output_dir + "/" + "p_b.npy", p_b)
np.save(args.output_dir + "/" + "hat_p_r.npy", hat_p_r)
np.save(args.output_dir + "/" + "hat_p_g.npy", hat_p_g)
np.save(args.output_dir + "/" + "hat_p_b.npy", hat_p_b)

preds = np.array(preds_test)
np.save(args.output_dir + "/preds.npy", preds)

mus_test = np.array(mus_test)
betas_test = np.array(betas_test)
pis_test = np.array(pis_test)

print("starting to calculate uncalibrated uncertainty masks.")
mask = get_uncerts(mus_test, betas_test, pis_test)
print("stopped calculating uncalibrated uncertainty masks.")
np.save(args.output_dir + "/uncal_masks.npy", mask)
