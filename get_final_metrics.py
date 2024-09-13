import argparse
import json
import numpy as np
import multiprocessing as mp
from sklearn.isotonic import IsotonicRegression
import tensorflow as tf
from scipy.misc import derivative

import configs_cal
from eval_cal import get_cdf_params

EPS = 1e-12


def get_args_parser():
    parser = argparse.ArgumentParser("Calibrating NeRF Uncertainties")
    parser.add_argument(
        "--scene",
        default="flower",
        help="name of LLFF scene out of {room, fern, flower, fortress, horns, leaves, orchids, trex}",
    )
    parser.add_argument(
        "--disable_lpips", action="store_true", help="disable lpips for memory reasons"
    )
    parser.add_argument(
        "--gin_config",
        default="configs/llff_flower.gin",
        help="name of Gin config file for pretrained FlipNeRF model",
    )
    parser.add_argument(
        "--test_model_dir",
        default="checkpoints/llff3/flower",
        help="directory containing pretrained FlipNeRF model checkpoint for calibrating",
    )
    parser.add_argument(
        "--data_split_file",
        default="llff_3_view_data_splits.json",
        help="name of JSON file with training and test splits",
    )
    parser.add_argument(
        "--num_procs",
        default=48,
        type=int,
        help="number of processes to use for multiprocessing",
    )

    return parser


def mse_to_psnr(mse):
    """
    Computes PSNR given an MSE (we assume the maximum pixel value is 1).

    [mse]: MSE to compute PSNR for

    Returns: PSNR for given MSE [mse]
    """
    return -10.0 / np.log(10.0) * np.log(mse)


def get_psnr(preds, gts):
    """
    Computes PSNR of inferred images [preds] compared to ground truth images
    [gts].

    [preds]: predicted images
    [gts]: ground truth images corresponding to [preds]

    Returns: PSNR for predicted images [preds]

    -- From FlipNeRF codebase
    """
    return float(mse_to_psnr(((preds - gts) ** 2).mean()))


def load_lpips():
    """
    Gets LPIPS function for evaluating image quality.

    -- From FlipNeRF codebase
    """
    # Make sure tf not using gpu due to memory limits.
    # Set CPU as available physical device
    my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
    graph = tf.compat.v1.Graph()
    session = tf.compat.v1.Session(graph=graph)
    with graph.as_default():
        input1 = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
        input2 = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
        with tf.compat.v1.gfile.Open("alex_net.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

            target = tf.compat.v1.transpose(
                (input1[tf.compat.v1.newaxis] * 2.0) - 1.0, [0, 3, 1, 2]
            )
            pred = tf.compat.v1.transpose(
                (input2[tf.compat.v1.newaxis] * 2.0) - 1.0, [0, 3, 1, 2]
            )
            tf.compat.v1.import_graph_def(
                graph_def, input_map={"0:0": target, "1:0": pred}
            )
            distance = graph.get_operations()[-1].outputs[0]

    def lpips_distance(img1, img2):
        with graph.as_default():
            return session.run(distance, {input1: img1, input2: img2})[0, 0, 0, 0]

    return lpips_distance


def get_lpips(preds, gts):
    """
    Computes LPIPS of inferred images [preds] compared to ground truth images
    [gts].

    [preds]: predicted images
    [gts]: ground truth images corresponding to [preds]

    Returns: LPIPS for predicted images [preds]

    -- From FlipNeRF codebase
    """
    lpips_fn = load_lpips()
    return float(lpips_fn(preds, gts))


def pdf(x, mus, betas, pis):
    """
    Evaluates mixture of Laplacians PDF with location parameters [mus], scale
    parameters [betas] and mixture coefficients [pis] at color value [x].

    [mus]: location parameters for mixture of Laplacians
    [betas]: scale parameters for mixture of Laplacians
    [pis]: mixture coefficients for mixture of Laplacians

    Returns: PDF value for Laplacian mixture evaluated at the color [x]
    """
    return np.sum(pis * (1 / (2 * betas)) * np.exp(-np.abs(x - mus) / betas))


def cdf(x, mus, betas, pis):
    """
    Evaluates mixture of Laplacians CDF with location parameters [mus], scale
    parameters [betas] and mixture coefficients [pis] at color value [x].

    [mus]: location parameters for mixture of Laplacians
    [betas]: scale parameters for mixture of Laplacians
    [pis]: mixture coefficients for mixture of Laplacians

    Returns: CDF value for Laplacian mixture evaluated at the color [x]
    """

    return np.sum(
        pis * (0.5 + 0.5 * np.sign(x - mus) * (1 - np.exp(-np.abs(x - mus) / betas))),
        axis=-1,
    )


def deriv(A, p):
    """
    Evaluates the derivative of the regression function [A] at the input value
    [p] using the method of finite differences.

    [A]: regression model with implemented [predict] method
    [p]: input where to compute the derivative

    Returns: the derivative of [A] at [p] approximated with the method of
    finite differences
    """
    f = lambda x: A.predict([x])[0]
    return derivative(f, p, dx=1e-5)


def get_nll(gts, mus, betas, pis, num_procs):
    """
    Evaluates mean (over pixels and RGB color channels following CF-NeRF)
    negative log-likelihood of ground truth images [gts] assuming the mixture
    of Laplacians parameterized by location parameters, scale parameters, and
    mixture coefficients [mus], [betas], and [pis].

    [gts]: ground truth images
    [mus]: location parameters for mixture of Laplacians
    [betas]: scale parameters for mixture of Laplacians
    [pis]: mixture coefficients for mixture of Laplacians

    Returns: mean negative log-likelihood of ground truth images [gts]
    """

    global get_nll_loc
    num_mix_comps = mus.shape[-2]
    gts = gts.reshape(-1, 3)
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    log_pdf_vals = mp.Array("d", np.zeros(pis.shape[0]))

    def get_nll_loc(px_ind):
        gt = gts[px_ind]
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        log_pdf = np.log(
            pdf(gt[0], mu[:, 0], beta[:, 0], pi)
            * pdf(gt[1], mu[:, 1], beta[:, 1], pi)
            * pdf(gt[2], mu[:, 2], beta[:, 2], pi)
            + EPS
        )
        log_pdf_vals[px_ind] = -log_pdf

    proc_pool = mp.Pool(num_procs)
    proc_pool.map(get_nll_loc, range(pis.shape[0]))
    proc_pool.close()
    proc_pool.join()

    log_pdf_vals = np.array(log_pdf_vals)

    # Average over color channels.
    return np.mean(log_pdf_vals) / 3


def get_nll_chain_rule(gts, mus, betas, pis, A_R, A_G, A_B, num_procs):
    """
    Evaluates mean (over pixels and RGB color channels following CF-NeRF)
    calibrated negative log-likelihood of ground truth images [gts] assuming
    the mixture of Laplacians with location parameters, scale parameters, and
    mixture coefficients [mus], [betas], and [pis] respectively parameterizes
    the uncalibrated distribution, and [A_R], [A_G], and [A_B] are the
    calibration functions.

    Note: uses chain rule to evaluate the calibrated PDF using the calibrated
    CDF. Calibrated CDF is A(CDF(gt)). Corresponding PDF at gt is the
    derivative of A(CDF(gt)). This is: A'(CDF(gt)) * PDF(gt) using the chain
    rule. A' can be approximated using the method of finite differences. The
    PDF of the uncalibrated CDF is just the PDF of the mixture of Laplacians,
    which has a closed form expression.

    [gts]: ground truth images
    [mus]: location parameters for mixture of Laplacians
    [betas]: scale parameters for mixture of Laplacians
    [pis]: mixture coefficients for mixture of Laplacians
    [A_R]: calibration function for red color channel
    [A_G]: calibration function for green color channel
    [A_B]: calibration function for blue color channel

    Returns: mean negative log-likelihood of ground truth images [gts]
    """

    global get_nll_loc
    num_mix_comps = mus.shape[-2]
    gts = gts.reshape(-1, 3)
    mus = mus.reshape(-1, num_mix_comps, 3)
    betas = betas.reshape(-1, num_mix_comps, 3)
    pis = pis.reshape(-1, num_mix_comps)
    log_pdf_vals = mp.Array("d", np.zeros(pis.shape[0]))

    def get_nll_loc(px_ind):
        gt = gts[px_ind]
        mu = mus[px_ind]
        beta = betas[px_ind]
        pi = pis[px_ind]
        p_r = cdf(gt[0], mu[:, 0], beta[:, 0], pi)
        p_g = cdf(gt[1], mu[:, 1], beta[:, 1], pi)
        p_b = cdf(gt[2], mu[:, 2], beta[:, 2], pi)

        # A'(CDF(gt)) * PDF(gt)
        log_pdf = np.log(
            deriv(A_R, p_r)
            * pdf(gt[0], mu[:, 0], beta[:, 0], pi)
            * deriv(A_G, p_g)
            * pdf(gt[1], mu[:, 1], beta[:, 1], pi)
            * deriv(A_B, p_b)
            * pdf(gt[2], mu[:, 2], beta[:, 2], pi)
            + EPS
        )
        
        log_pdf_vals[px_ind] = -log_pdf

    proc_pool = mp.Pool(num_procs)
    proc_pool.map(get_nll_loc, range(pis.shape[0]))
    proc_pool.close()
    proc_pool.join()

    log_pdf_vals = np.array(log_pdf_vals)

    # Average over color channels.
    return np.mean(log_pdf_vals) / 3


def get_cal_err(gts, mus, betas, pis, A_R, A_G, A_B, cal):
    """
    Gets the calibration error of mixture of Laplacians parameterized by [mus],
    [betas], [pis] and calibration functions [A_R], [A_G], and [A_B] (if [cal]
    is True) for ground truth images [gts].

    [gts]: ground truth images
    [mus]: location parameters of mixture of Laplacians
    [betas]: scale parameters of mixture of Laplacians
    [cal]: whether to use the uncalibrated (False) or calibrated (True) model
    to calculate the calibration error

    Returns: the RGB calibration errors of the calibrated model if [cal] is
    True or of the uncalibrated model if [cal] is False as a triple
    """
    ps = []
    for img_ind in range(gts.shape[0]):
        ps.append(cdf_matrix(pis[img_ind], mus[img_ind], betas[img_ind], gts[img_ind]))
    ps = np.array(ps)
    ps = ps.reshape(-1, 3)
    p_r = ps[:, 0]
    p_g = ps[:, 1]
    p_b = ps[:, 2]
    if cal:
        p_r = A_R.predict(p_r)
        p_g = A_G.predict(p_g)
        p_b = A_B.predict(p_b)

    p_r = np.sort(np.array(p_r))
    p_g = np.sort(np.array(p_g))
    p_b = np.sort(np.array(p_b))
    hat_p_r = get_emp_confs(p_r)
    hat_p_g = get_emp_confs(p_g)
    hat_p_b = get_emp_confs(p_b)

    return (
        np.mean((p_r - hat_p_r) ** 2),
        np.mean((p_g - hat_p_g) ** 2),
        np.mean((p_b - hat_p_b) ** 2),
    )


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


# Parse commandline arguments.
args = get_args_parser()
args = args.parse_args()

# Get test indices for the specified scene.
all_data_splits = json.load(open(args.data_split_file))
scene_data_split = all_data_splits[args.scene]
test_inds = scene_data_split["test"]

# Load config file for the pretrained FlipNeRF model.
config = configs_cal.load_config(args.gin_config, save_config=True)

# Save CDF parameters for test data from FlipNeRF model trained on 3 views.
config.checkpoint_dir = args.test_model_dir
(preds_test, betas_test, mus_test, pis_test, gts_test) = get_cdf_params(
    config, test_inds
)

# Load calibration curves inferred from meta-calibrator.
p_r = np.loadtxt("results_r/" + args.scene + "_pred.txt")
p_g = np.loadtxt("results_g/" + args.scene + "_pred.txt")
p_b = np.loadtxt("results_b/" + args.scene + "_pred.txt")
hat_p_r = np.linspace(0, 1, 384)
hat_p_g = hat_p_r
hat_p_b = hat_p_r

D_R = (p_r, hat_p_r)
D_G = (p_g, hat_p_g)
D_B = (p_b, hat_p_b)

# Train auxiliary models A^{R}, A^{G}, and A^{B} on calibration sets D^{R}, D
# {G}, and D^{B}.
A_R = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_R[0], D_R[1]
)
A_G = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_G[0], D_G[1]
)
A_B = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_B[0], D_B[1]
)

# Compute average image quality metrics.
preds = np.array(preds_test)
betas = np.array(betas_test)
mus = np.array(mus_test)
pis = np.array(pis_test)
gts = np.array(gts_test)

avg_psnr = 0
avg_lpips = 0
for image_ind in range(gts.shape[0]):
    avg_psnr += get_psnr(preds[image_ind], gts[image_ind])
    if not args.disable_lpips:
        avg_lpips += get_lpips(preds[image_ind], gts[image_ind])

avg_psnr = avg_psnr / gts.shape[0]
avg_lpips = avg_lpips / gts.shape[0]

print("PSNR:")
print(avg_psnr)
print("LPIPS:")
print(avg_lpips)

# Compute uncertainty metrics.
nll_uncal = get_nll(gts, mus, betas, pis, args.num_procs)
print("NLL (Uncal.):")
print(nll_uncal)
nll_cal = get_nll_chain_rule(gts, mus, betas, pis, A_R, A_G, A_B, args.num_procs)
print("NLL (Cal.):")
print(nll_cal)
cal_err_uncal = get_cal_err(gts, mus, betas, pis, A_R, A_G, A_B, False)
print("Cal. Err. (Uncal.):")
print(cal_err_uncal)
cal_err_cal = get_cal_err(gts, mus, betas, pis, A_R, A_G, A_B, True)
print("Cal. Err. (Cal.):")
print(cal_err_cal)
