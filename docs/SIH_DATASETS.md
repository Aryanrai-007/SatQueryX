# SIH2026 dataset usage

SatQueryX maps each dataset in the supplied problem statement to a concrete software role.

| Dataset | Source | Used for |
|---|---|---|
| BigEarthNet.txt | https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt | Remote-sensing adaptation, S1/S2 paired representation, captioning/VQA/grounding |
| VRSBench | https://github.com/lx709/VRSBench | Single-image captioning, grounding and VQA evaluation |
| RSVQA | https://github.com/syvlo/RSVQA | Single-image quantitative VQA evaluation |
| CDVQA | https://github.com/YZHJessica/CDVQA | Bi-temporal change VQA |
| ISRO/SAC | Provided by evaluator | Hidden Cartosat-2S/RISAT final evaluation |

## BigEarthNet.txt

Official dataset: `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`.
Paper: https://arxiv.org/abs/2603.29630

The official dataset contains co-registered Sentinel-1 and Sentinel-2 image pairs and text annotations. Its supplied loader supports `RGB`, `S2-10m20m`, and `S1S2-10m20m` band configurations and provides train/validation/test/bench splits.

## Why binaries are not committed

The dataset image stores are large. SatQueryX stores only loaders, configuration and training/evaluation code. Dataset binaries and model weights stay in the user's environment or model/data hosting service.

## SIH hidden evaluation

The supplied statement says the final evaluation set contains pre-georeferenced, co-registered Cartosat-2S optical and RISAT SAR pairs with task-specific annotations. SatQueryX therefore does not hard-code Sentinel-only model routing at the input-validation layer.
