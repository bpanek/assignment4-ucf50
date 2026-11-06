# UCF50 Video Action Recognition

This project fixes and extends the supplied HMDB51 video-classification codebase for UCF50. It retains the repo’s LRCN, training loop, and evaluation code, and adds a frozen R(2+1)D-18 video encoder as a second input path. The original Colab experiment and a standalone RunPod rerun both reached **96.90% held-out test accuracy (1,033 of 1,066 clips)**, exceeding the assignment’s 93% target by **3.90 percentage points**. The repo-based notebook runs the integrated implementation and reports its own result.

## Setup and reproduction

1. Clone or download this repository. Open a terminal in the folder containing `run.py`.
2. Download [`UCF50.rar`](https://www.crcv.ucf.edu/data/UCF50.rar) and extract it into `data/`. The action-class folders should be directly inside `data/UCF50/`.
3. Use the checked-in `ucf50_split_seed0.json` at the repo root. It contains the fixed clip assignments and class-to-index mapping used in the experiments.
4. Set up a Python environment with a matching CUDA-enabled PyTorch and TorchVision pair. The standalone RunPod rerun used **PyTorch 2.8.0+cu128**, **CUDA 12.8**, and an **NVIDIA RTX PRO 6000 Blackwell Server Edition**. The frozen video encoder path requires a CUDA GPU with bfloat16 support.
5. Install the remaining dependencies and sign in to Weights & Biases.

   ```bash
   python -m pip install numpy Pillow scikit-learn tqdm opencv-python-headless wandb pytest jupyterlab
   wandb login
   ```

6. Start JupyterLab from the repo root and open `UCF50_Repo_LRCN.ipynb`. The kernel’s working directory should be the folder containing `run.py`. Run the setup, training, and evaluation cells in order.

   ```bash
   python -m jupyterlab
   ```

```text
assignment4-ucf50/
├── UCF50_Repo_LRCN.ipynb
├── run.py
├── models.py
├── video_windows.py
├── ucf50_split_seed0.json
├── tests/
└── data/
    └── UCF50/
        ├── BaseballPitch/
        │   └── v_BaseballPitch_g15_c01.avi
        └── ... other action-class folders
```

The notebook resolves paths relative to the repo root. It uses the original AVI files directly. TorchVision downloads the Kinetics-400 encoder weights on first use if they are absent from the cache. Training writes the best validation checkpoint to `outputs/<training-run-id>/best_model_wts.pt`. The evaluation cell loads that checkpoint and scores the held-out test split.

## Bugs fixed

| Issue | What went wrong | Fix |
| --- | --- | --- |
| **Major bug — Incorrect frame ordering** | The JPEG loader took the first 16 paths from an unsorted glob. The LSTM could receive frames out of chronological order, and alphabetical sorting would still put `frame10.jpg` before `frame2.jpg`. | Sort by numeric frame index **before** applying the frame limit. The regression test presents frames in order `10, 2, 1` and verifies that the loader returns `1, 2` when limited to two frames. |
| **Data leakage** | The original class-stratified split assigned individual clips independently. Related clips from the same UCF50 source group could appear in both training and evaluation. | Assign whole **(class, source group)** pairs to one split. Verify that every clip appears once and no source group crosses splits. |
| **Off-by-one frame sampling** | The sampler requested `n_frames + 1` positions. A real clip produced 17 frames when asked for 16, and truncation could drop the last sampled frame. | Request `min(n_frames, video_length)` positions and use 64-bit frame indices. |
| **Single-frame LSTM failure** | The first recurrent output was discarded. For a one-frame clip, the later loop never ran and the output variable was undefined. | Include the first time step in the recurrent output sequence. The shared LRCN now processes the complete sequence and supports both last-step and mean pooling. |
| **Scheduler compatibility** | `ReduceLROnPlateau(..., verbose=1)` caused training to fail at startup in the Colab environment. | Remove `verbose` from the scheduler calls. The shared trainer also supports `scheduler=None` for the fixed-learning-rate experiment. |

### Fixed UCF50 split

The split manifest is authoritative for comparisons. The train and test proportions were applied to source groups within each class, with the remaining groups assigned to validation.

| Split | Clips | Class/source-group pairs | Classes |
| --- | ---: | ---: | ---: |
| Train | 5,051 | 960 | 50 |
| Validation | 564 | 104 | 50 |
| Test | 1,066 | 200 | 50 |
| **Total** | **6,681** | **1,264** | **50** |

## Model changes

- **Backbone and short-window motion encoding.** The original LRCN extracts one embedding per frame with a 2D CNN. The new path uses a frozen R(2+1)D-18 pretrained on Kinetics-400. It samples **four ordered windows of 16 frames** at approximately **15 FPS**. The CNN captures motion within each window and returns one **512-dimensional embedding**. The LSTM combines the four embeddings across the clip. The backbone, pretraining, and temporal input changed together in the winning experiment, so the reported gain measures their combined effect.
- **Temporal pooling.** The repo’s LRCN supports either the last LSTM output or the mean of all recurrent outputs. We tested both with frozen image backbones. Last-step pooling performed better in the ResNet-50 and ConvNeXt comparisons below and is used in the final model.

The integrated path uses `video_windows.py` to decode and encode the clips, `models.py` for the same LRCN used by the frame-based path, and `train.py` and `test.py` for training and evaluation. It computes the frozen embeddings once per split. The classifier uses one LSTM layer with 100 hidden units, dropout of 0.1, and 50 output classes. Training uses seed 42, AdamW at a fixed learning rate of 0.001, cross-entropy, batch size 128, and 15 epochs. The checkpoint is selected by validation accuracy. The held-out test split is used for final evaluation.

## Experiment history and decisions

All accuracy values below are on the same **564 validation clips** unless explicitly marked as test accuracy.

| Experiment | Result | Why we moved on |
| --- | --- | --- |
| Corrected pretrained ResNet-34 + LSTM, last step | **74.82% best**, epoch 8 of 10. Final validation 73.40% and train 97.27%. | Established the corrected baseline. Validation plateaued as training accuracy rose. [W&B](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/q6eso6q2) |
| ResNet-34 temporal mean-pooling experiment | **75.71% final**. | Tested whether averaging the LSTM outputs helped. The run retained the baseline name, and its recorded pooling configuration and best epoch were not confirmed. [W&B](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/vpql8z2e) |
| Frozen ResNet-50 ImageNet-1K V2 + LSTM | **85.28% last step**, epoch 6, tied at epoch 10. **84.75% mean**, epoch 13. | Caching the 16 frame embeddings per clip made LSTM training faster. Last-step pooling led on this encoder. [Last](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/5uyih9a9) · [Mean](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/2stulaq4) |
| Frozen ConvNeXt-Base ImageNet-1K V1 + LSTM | **89.72% last step**, epoch 7. **89.54% mean**, epoch 10. | Stronger still-frame features improved validation accuracy. Last-step pooling led by one validation clip. Both heads reached 100% training accuracy. [Last](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/r1id2akt) · [Mean](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/ebvqs8e1) |
| Fine-tuned ConvNeXt’s last two blocks and LSTM | **88.65% best** over three epochs. | Updating the encoder from the frozen-model checkpoint did not beat 89.72%. [W&B](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/0oow5ijy) |
| Concatenated normalized ConvNeXt and ResNet-50 frame features into one LSTM | **87.94% best**, epoch 7. | Combining the two sets of still-frame features did not improve validation accuracy. [W&B](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/bs015xhl) |
| Frozen R(2+1)D-18 video CNN + LSTM | **95.21% validation** at epoch 14. **96.90% test**, 1,033 of 1,066 clips. | The CNN captures motion inside each window and the LSTM combines windows across the clip. Validation improved by **5.49 percentage points** over frozen ConvNeXt. [Training](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/nwc0812d) · [Test](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/w5xs39ul) |

## Results and Weights & Biases

| Run | Selected epoch | Validation accuracy | Validation loss | Test accuracy | Test loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Colab experiment | 14 | 95.21% | 0.1943 | **96.90% (1,033/1,066)** | 0.1466 |
| Standalone RunPod reproduction | 14 | 95.04% | 0.1930 | **96.90% (1,033/1,066)** | 0.1477 |

- **Original Colab experiment** — [Train and validation loss and accuracy](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/nwc0812d) · [Held-out test loss and accuracy](https://wandb.ai/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/w5xs39ul)
- **Standalone RunPod reproduction** — [Train and validation loss and accuracy](https://forge.coreweave.com/wandb/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/yiqgpmr1) · [Held-out test loss and accuracy](https://forge.coreweave.com/wandb/bpanek83-johns-hopkins-university/assignment4-ucf50/runs/e4cead0w)

The table reports the validation-selected checkpoints from the two completed standalone experiments. The integrated repo notebook logs training and validation loss and accuracy each epoch. It logs test loss, test accuracy, and the number of correct and total clips in a separate evaluation run linked to the training run. Its metrics and W&B run IDs are generated when the notebook runs.

## Verification

Run the regression tests from the repo root. They use synthetic clips and temporary image files, so they run on CPU without the dataset or pretrained weights.
```bash
python -m pip install pytest
python -m pytest -q tests/
```

- `tests/test_frame_order.py` verifies numeric frame ordering before truncation.
- `tests/test_dataset_split.py` verifies clip coverage and source-group isolation across splits.

Both regression tests passed during development. A separate local smoke check verified that the repo’s LRCN accepts input of shape `(2, 4, 512)` and returns finite logits of shape `(2, 50)`.

Pylint is pending. To run it from the repo root:

```bash
python -m pip install pylint
python -m pylint models.py run.py run_training.py train.py test.py utils.py video_datasets.py video_windows.py tests/test_frame_order.py tests/test_dataset_split.py
```

## References

- [UCF50 Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF50.php)
- [Donahue et al., Long-Term Recurrent Convolutional Networks for Visual Recognition and Description, CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.html)