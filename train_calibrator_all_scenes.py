import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA

ALL_SCENES = ["chair", "drums", "fern", "ficus", "fortress", "horns", "hotdog", "leaves", "lego", "materials", "mic", "orchids", "room", "scan8", "scan21", "scan30", "scan31", "scan34", "scan38", "scan40", "scan41", "scan45", "scan55", "scan63", "scan82", "scan103", "scan110", "scan114", "ship", "trex", "flower"]

def get_args_parser():
    parser = argparse.ArgumentParser("Train Meta-calibrator")
    parser.add_argument(
        "--test_scene",
        default="flower",
        help="name of LLFF scene to test",
    )
    return parser


class NeuralNet(nn.Module):
    """
    The meta-calibrator is a [NeuralNet] that predicts the PCA coefficients for
    a low-dimensional (3 dims) basis of functions representing the calibration 
    curves.
    """

    def __init__(self, in_features):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(in_features, 256, bias=True)
        self.layer2 = nn.Linear(256, 128, bias=True)
        self.layer3 = nn.Linear(128, 128, bias=True)
        self.layer4 = nn.Linear(128, 3, bias=True)

    def forward(self, x):
        return self.layer4(
            torch.nn.functional.leaky_relu(
                self.layer3(
                    torch.nn.functional.leaky_relu(
                        self.layer2(torch.nn.functional.leaky_relu(self.layer1(x)))
                    )
                )
            )
        )


def extract_pca(training_scenes, color="r"):
    """
    Fits PCA representation with 3 components to training scene calibration
    curves for the color channel [color].

    [training_scenes]: list of training scenes to use for obtaining PCA
    representation
    [color]: color (out of {'r', 'g', 'b'}) channel the calibration curves were
    obtained from

    Returns: ([pca], [basis], [coeff], [mean]), where [pca] is the PCA model,
    [basis] contains the 3 PCA components from [pca] representing the training
    calibration curves, [coeff] contains the PCA coefficients for each of the
    training scenes calculated from [pca], and [mean] is the mean of the
    training calibration curves, subtracted before computing the PCA
    representation
    """

    hat_ps = []
    ps = []
    for scene in training_scenes:
        hat_p = np.load(scene + "/hat_p_" + color + ".npy")
        p = np.load(scene + "/p_" + color + ".npy")
        n = hat_p.shape[0]
        resample_idx = np.linspace(0, n - 1, 384, dtype=np.int32)

        hat_ps.append(hat_p[resample_idx])
        ps.append(p[resample_idx])

    # Convert to numpy arrays.
    hat_ps = np.stack(hat_ps, 0)
    ps = np.stack(ps, 0)
    mean = np.mean(ps, 0)
    ps = ps - mean

    # Compute PCA representation.
    pca = PCA(n_components=3)
    pca.fit(ps)

    basis = pca.components_
    coeff = pca.transform(ps)

    return (pca, coeff, basis, mean)


def load_features(scene):
    """
    Returns DINOv2 features of inferred images and uncalibrated uncertainty
    maps for scene [scene].

    [scene]: scene for which to compute DINOv2 features

    Returns: stacked DINOv2 features of shape N x 768 for inferred images and
    uncalibrated uncertainty maps, where N is the number of inferred images
    """
    features_a = np.load(scene + "/preds_embeddings.npy") / 8.0  # N x 384
    features_b = np.load(scene + "/uncal_masks_embeddings.npy") / 8.0  # N x 384
    features = np.concatenate((features_a, features_b), 1)
    return features


def load_scene(scene, coeff):
    """
    Returns DINOv2 features and PCA coefficients for scene [scene] as pair of
    tensors on GPU.

    [scene]: name of scene to load data for
    [coeff]: coefficients of PCA representation for [scene]'s calibration curves

    Returns: ([x_tensor], [y_tensor]), where [x_tensor] contains the DINOv2
    features and [y_tensor] contains the PCA coefficients for scene [scene]
    """

    # Load DINOv2 features for inferred image and uncalibrated uncertainty map
    # for scene [scene].
    features = load_features(scene)

    # Convert features to torch tensor on GPU.
    x_tensor = torch.tensor(features / 768.0, dtype=torch.float32).cuda()
    # Convert PCA coefficients to torch tensor on GPU.
    y_tensor = torch.tensor(coeff, dtype=torch.float32).reshape(1, -1).cuda()

    return x_tensor, y_tensor

# Parse commandline arguments.
args = get_args_parser()
args = args.parse_args()
test_scenes = [args.test_scene]
train_scenes = ALL_SCENES.copy()

# Remove test scene from training scenes.
train_scenes.remove(args.test_scene)

# Get test scene names.
all_test_scene_names = ["scenes/" + x.strip() for x in test_scenes]

# Get train scene names.
all_train_scene_names = ["scenes/" + x.strip() for x in train_scenes]

for color in ["r", "g", "b"]:
    for test_scene_name in all_test_scene_names:
        train_scene_names = all_train_scene_names.copy()

        test_scene_names = [test_scene_name]
        print("=========")
        print("Test scenes: " + str(test_scene_names))
        print("Train scenes: " + str(train_scene_names))
        print("=========")

        # Compute PCA representation of training scenes.
        (pca, coeffs, basis, mean) = extract_pca(train_scene_names, color)

        # Load the first training sample.
        x_tensor, y_tensor = load_scene(train_scene_names[0], coeffs[0, :])
        # Initialize the meta-calibrator neural network.
        net = NeuralNet(x_tensor.shape[1])
        net = net.cuda()

        # Training setup:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        epochs = 3000

        # Training loop:
        for epoch in range(epochs):
            # Select random scene name.
            scene_name = np.random.choice(train_scene_names)
            scene_idx = train_scene_names.index(scene_name)

            x_tensor, y_tensor = load_scene(scene_name, coeffs[scene_idx, :])

            # Pick random predicted image and uncalibrated uncertainty map.
            feature_idx = np.random.randint(0, x_tensor.shape[0])
            x_tensor = x_tensor[feature_idx, :].reshape(1, -1)

            optimizer.zero_grad()
            outputs = net(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # Test phase:
        for scene in test_scene_names:
            # Load the DINOv2 embeddings of test inferred image and
            # uncalibrated uncertainty map.
            features = load_features(scene)
            # Convert features to torch tensor on GPU.
            x_tensor = torch.tensor(features / 768.0, dtype=torch.float32).cuda()
            # Predict the PCA coefficients for the test scene.
            y_pred = net(x_tensor[0, :].reshape(1, -1)).cpu().detach().numpy()

        y_pred = np.matmul(y_pred, basis)
        
        # Make output directory for storing data.
        os.makedirs("results_" + color, exist_ok=True)
        np.savetxt(
            "results_" + color + "/" + test_scene_names[0].split("/")[-1] + "_pred.txt",
            y_pred.flatten() + mean,
        )
