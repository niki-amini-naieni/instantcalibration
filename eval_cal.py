# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified from original evaluation script for FlipNeRF."""
import functools
import time
import flax
from flax.training import checkpoints
from internal import datasets, models, utils
import jax
from jax import random
import numpy as np
import tensorflow as tf


def get_cdf_params(config, img_inds):

    tf.config.experimental.set_visible_devices([], "GPU")

    # Set to 1 so can select which images (out of all available images for scene) to evaluate model with.
    config.llffhold = 1

    dataset = datasets.load_dataset("test", config.data_dir, config)

    model, init_variables = models.construct_mipnerf(
        random.PRNGKey(20200823), dataset.peek()["rays"], config
    )
    optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables

    # Rendering is forced to be deterministic even if training was randomized, as
    # this eliminates 'speckle' artifacts.
    def render_eval_fn(variables, _, rays):
        return jax.lax.all_gather(
            model.apply(
                variables,
                None,  # Deterministic.
                rays,
                resample_padding=config.resample_padding_final,
                compute_extras=True,
            ),
            axis_name="batch",
        )

    # pmap over only the data input.
    render_eval_pfn = jax.pmap(
        render_eval_fn,
        in_axes=(None, None, 0),
        donate_argnums=2,
        axis_name="batch",
    )

    preds = []
    betas = []
    mus = []
    pis = []
    gts = []

    while True:
        # Fix for loading pre-trained models.
        try:
            state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
        except:  # pylint: disable=bare-except
            print("Using pre-trained model.")
            state_dict = checkpoints.restore_checkpoint(config.checkpoint_dir, None)
            for i in [9, 17]:
                del state_dict["optimizer"]["target"]["params"]["MLP_0"][f"Dense_{i}"]
            state_dict["optimizer"]["target"]["params"]["MLP_0"][
                "Dense_9"
            ] = state_dict["optimizer"]["target"]["params"]["MLP_0"]["Dense_18"]
            state_dict["optimizer"]["target"]["params"]["MLP_0"][
                "Dense_10"
            ] = state_dict["optimizer"]["target"]["params"]["MLP_0"]["Dense_19"]
            state_dict["optimizer"]["target"]["params"]["MLP_0"][
                "Dense_11"
            ] = state_dict["optimizer"]["target"]["params"]["MLP_0"]["Dense_20"]
            del state_dict["optimizerd"]
            state = flax.serialization.from_state_dict(state, state_dict)

        step = int(state.optimizer.state.step)

        print(f"Evaluating checkpoint at step {step}.")

        for idx in range(dataset.size):
            batch = next(dataset)
            # Only evaluate specified images.
            if idx in img_inds:
                print(f"Evaluating image {idx+1}/{dataset.size}")
                eval_start_time = time.time()
                rendering = models.render_image(
                    functools.partial(render_eval_pfn, state.optimizer.target),
                    batch["rays"],
                    None,
                    config,
                )
                print(f"Rendered in {(time.time() - eval_start_time):0.3f}s")

                # Save CDF parameters.
                preds.append(np.array(rendering["rgb"]))
                betas.append(np.array(rendering["gamma"]))
                mus.append(np.array(rendering["mu"]))
                pis.append(np.array(rendering["pi"]))
                gts.append(np.array(batch["rgb"]))

        if config.eval_only_once:
            break

    return (preds, betas, mus, pis, gts)
