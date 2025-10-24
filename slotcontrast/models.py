from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchvision.utils import make_grid

from slotcontrast import configuration, losses, modules, optimizers, utils, visualizations
from slotcontrast.data.transforms import Denormalize
from einops import rearrange, repeat

# def _build_with_steve(
#     model_config: configuration.ModelConfig,
#     optimizer_config,
#     train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
#     val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
# ):
#     """Build ObjectCentricModel with STEVE components (dVAE + STEVE decoder)"""
#     # Import STEVE components
#     try:
#         from slotcontrast.modules.steve_components import dVAE, STEVEDecoder
#         from slotcontrast.modules.steve_components.steve_losses import (
#             DVAEReconstructionLoss,
#             STEVECrossEntropyLoss,
#             STEVEGumbelTemperatureScheduler,
#         )
#     except ImportError as e:
#         raise ImportError(
#             f"STEVE components requested but cannot be imported: {e}\n"
#             "Make sure STEVE components are available in slotcontrast/modules/steve_components/"
#         ) from e

#     optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)

#     # Build standard SlotContrast components
#     initializer = modules.build_initializer(model_config.initializer)

#     # Encoder (can use silicon-menagerie ViT or standard encoders)
#     if model_config.encoder.get('use_silicon_vit', False):
#         try:
#             from slotcontrast.modules.silicon_vit import build_silicon_vit
#             backbone = build_silicon_vit(model_config.encoder.backbone)
#             encoder = modules.encoders.FrameEncoder(
#                 backbone=backbone,
#                 pos_embed=None,
#                 output_transform=modules.build_module(model_config.encoder.output_transform)
#                     if model_config.encoder.get('output_transform') else None,
#                 spatial_flatten=model_config.encoder.get('spatial_flatten', False),
#                 main_features_key=model_config.encoder.get('main_features_key', 'vit_block12'),
#             )
#         except ImportError:
#             print("Warning: silicon_vit requested but not available, using standard encoder")
#             encoder_config = deepcopy(model_config.encoder)
#             encoder_config.pop('use_silicon_vit', None)
#             encoder = modules.build_encoder(encoder_config, "FrameEncoder")
#     else:
#         encoder_config = deepcopy(model_config.encoder)
#         encoder_config.pop('use_silicon_vit', None)
#         encoder = modules.build_encoder(encoder_config, "FrameEncoder")

#     grouper = modules.build_grouper(model_config.grouper)
#     decoder = modules.build_decoder(model_config.decoder)

#     # Build STEVE components if enabled
#     dvae_module = None
#     steve_decoder = None

#     if model_config.get('use_dvae', False):
#         dvae_config = model_config.get('dvae', {})
#         dvae_module = dVAE(
#             vocab_size=dvae_config.get('vocab_size', 4096),
#             img_channels=dvae_config.get('img_channels', 3),
#             use_checkpoint=dvae_config.get('use_checkpoint', False),
#         )
#         # Count parameters
#         dvae_params = sum(p.numel() for p in dvae_module.parameters())
#         dvae_trainable = sum(p.numel() for p in dvae_module.parameters() if p.requires_grad)
#         print(f"Added dVAE module:")
#         print(f"  - vocab_size: {dvae_config.get('vocab_size', 4096)}")
#         print(f"  - checkpoint: {dvae_config.get('use_checkpoint', False)}")
#         print(f"  - parameters: {dvae_params:,} ({dvae_params/1e6:.2f}M)")
#         print(f"  - trainable:  {dvae_trainable:,} ({dvae_trainable/1e6:.2f}M)")

#     if model_config.get('use_steve_decoder', False):
#         steve_decoder_config = model_config.get('steve_decoder', {})
#         steve_decoder = STEVEDecoder(
#             vocab_size=steve_decoder_config.get('vocab_size', 4096),
#             d_model=steve_decoder_config.get('d_model', 128),
#             slot_size=model_config.grouper.slot_dim,
#             num_decoder_blocks=steve_decoder_config.get('num_decoder_blocks', 4),
#             num_decoder_heads=steve_decoder_config.get('num_decoder_heads', 4),
#             dropout=steve_decoder_config.get('dropout', 0.1),
#             image_size=steve_decoder_config.get('image_size', 224),
#         )
#         # Count parameters
#         steve_params = sum(p.numel() for p in steve_decoder.parameters())
#         steve_trainable = sum(p.numel() for p in steve_decoder.parameters() if p.requires_grad)
#         print(f"Added STEVE decoder:")
#         print(f"  - vocab_size: {steve_decoder_config.get('vocab_size', 4096)}")
#         print(f"  - d_model: {steve_decoder_config.get('d_model', 128)}")
#         print(f"  - decoder_blocks: {steve_decoder_config.get('num_decoder_blocks', 4)}")
#         print(f"  - parameters: {steve_params:,} ({steve_params/1e6:.2f}M)")
#         print(f"  - trainable:  {steve_trainable:,} ({steve_trainable/1e6:.2f}M)")

#     # Target encoder (optional)
#     target_encoder = None
#     if model_config.target_encoder:
#         target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")

#     # Dynamics predictor (optional)
#     dynamics_predictor = None
#     if model_config.dynamics_predictor:
#         dynamics_predictor = modules.build_dynamics_predictor(model_config.dynamics_predictor)

#     # Process input type
#     input_type = model_config.get("input_type", "image")
#     if input_type == "image":
#         processor = modules.LatentProcessor(grouper, predictor=None)
#     elif input_type == "video":
#         encoder = modules.MapOverTime(encoder)
#         decoder = modules.MapOverTime(decoder)
#         if target_encoder:
#             target_encoder = modules.MapOverTime(target_encoder)
#         if model_config.predictor is not None:
#             predictor = modules.build_module(model_config.predictor)
#         else:
#             predictor = None
#         if model_config.latent_processor:
#             processor = modules.build_video(
#                 model_config.latent_processor,
#                 "LatentProcessor",
#                 corrector=grouper,
#                 predictor=predictor,
#             )
#         else:
#             processor = modules.LatentProcessor(grouper, predictor)
#         processor = modules.ScanOverTime(processor)
#     else:
#         raise ValueError(f"Unknown input type {input_type}")

#     # Build losses
#     target_type = model_config.get("target_type", "features")
#     if target_type == "input":
#         default_target_key = input_type
#     elif target_type == "features":
#         if model_config.target_encoder_input is not None:
#             default_target_key = "target_encoder.backbone_features"
#         else:
#             default_target_key = "encoder.backbone_features"
#     else:
#         raise ValueError(f"Unknown target type {target_type}")

#     loss_defaults = {
#         "pred_key": "decoder.reconstruction",
#         "target_key": default_target_key,
#         "video_inputs": input_type == "video",
#         "patch_inputs": target_type == "features",
#     }

#     # Build all losses (including STEVE losses)
#     if model_config.losses is None:
#         loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
#     else:
#         loss_fns = {}
#         for name, loss_config in model_config.losses.items():
#             # Handle STEVE losses
#             if loss_config.get('name') == 'DVAEReconstructionLoss':
#                 # Skip if dVAE is not enabled
#                 if not dvae_module:
#                     print(f"Skipping {name} because dVAE is not enabled")
#                     continue
#                 loss_fns[name] = DVAEReconstructionLoss(
#                     pred_key=loss_config.get('pred_key', 'dvae.reconstruction'),
#                     target_key=loss_config.get('target_key', input_type),
#                     normalize=loss_config.get('normalize', True),
#                     reduction=loss_config.get('reduction', 'mean'),
#                 )
#             elif loss_config.get('name') == 'STEVECrossEntropyLoss':
#                 # Skip if STEVE decoder is not enabled
#                 if not steve_decoder:
#                     print(f"Skipping {name} because STEVE decoder is not enabled")
#                     continue
#                 loss_fns[name] = STEVECrossEntropyLoss(
#                     pred_key=loss_config.get('pred_key', 'steve_decoder.cross_entropy'),
#                     target_key=loss_config.get('target_key', 'dvae.z_hard'),
#                     reduction=loss_config.get('reduction', 'mean'),
#                 )
#             else:
#                 # Standard SlotContrast losses
#                 loss_fns[name] = losses.build({**loss_defaults, **loss_config})

#     # Mask resizers
#     if model_config.mask_resizers:
#         mask_resizers = {
#             name: modules.build_utils(resizer_config, "Resizer")
#             for name, resizer_config in model_config.mask_resizers.items()
#         }
#     else:
#         mask_resizers = {
#             "decoder": modules.build_utils(
#                 {
#                     "name": "Resizer",
#                     "patch_inputs": target_type == "features",
#                     "video_inputs": input_type == "video",
#                     "resize_mode": "bilinear",
#                 }
#             ),
#             "grouping": modules.build_utils(
#                 {
#                     "name": "Resizer",
#                     "patch_inputs": True,
#                     "video_inputs": input_type == "video",
#                     "resize_mode": "bilinear",
#                 }
#             ),
#         }

#     if model_config.masks_to_visualize:
#         masks_to_visualize = model_config.masks_to_visualize
#     else:
#         masks_to_visualize = "decoder"

#     # Gumbel temperature scheduler
#     gumbel_temp_scheduler = None
#     if dvae_module:
#         gumbel_start_temp = model_config.get('gumbel_start_temp', 1.0)
#         gumbel_final_temp = model_config.get('gumbel_final_temp', 0.1)
#         gumbel_anneal_steps = model_config.get('gumbel_anneal_steps', 10000)
#         gumbel_temp_scheduler = STEVEGumbelTemperatureScheduler(
#             start_temp=gumbel_start_temp,
#             final_temp=gumbel_final_temp,
#             anneal_steps=gumbel_anneal_steps,
#         )

#     # Create model with STEVE components
#     model = ObjectCentricModel(
#         optimizer_builder,
#         initializer,
#         encoder,
#         processor,
#         decoder,
#         loss_fns,
#         loss_weights=model_config.get("loss_weights", None),
#         target_encoder=target_encoder,
#         dynamics_predictor=dynamics_predictor,
#         train_metrics=train_metrics,
#         val_metrics=val_metrics,
#         mask_resizers=mask_resizers,
#         input_type=input_type,
#         target_encoder_input=model_config.get("target_encoder_input", None),
#         visualize=model_config.get("visualize", False),
#         visualize_every_n_steps=model_config.get("visualize_every_n_steps", 1000),
#         masks_to_visualize=masks_to_visualize,
#         dvae=dvae_module,
#         steve_decoder=steve_decoder,
#         gumbel_temp_scheduler=gumbel_temp_scheduler,
#     )

#     if model_config.load_weights:
#         model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

#     return model


def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    # Check if STEVE components are requested
    use_dvae = model_config.get('use_dvae', False)

    dvae_module = None
    if use_dvae:
        from slotcontrast.modules.steve_components import dVAE
        from slotcontrast.modules.steve_components.steve_losses import (
            DVAEReconstructionLoss,
            STEVECrossEntropyLoss,
            STEVEGumbelTemperatureScheduler,
        )
        print("Building model with STEVE dVAE component")
        dvae_config = model_config.get('dvae', {})
        dvae_module = dVAE(
            # enc_size=[2,7],
            # dec_size=[7,2],
            vocab_size=dvae_config.get('vocab_size', 4096),
            img_channels=dvae_config.get('img_channels', 3),
            use_checkpoint=dvae_config.get('use_checkpoint', False),
        )
        # Count parameters
        dvae_params = sum(p.numel() for p in dvae_module.parameters())
        dvae_trainable = sum(p.numel() for p in dvae_module.parameters() if p.requires_grad)
        print(f"Added dVAE module:")
        print(f"  - vocab_size: {dvae_config.get('vocab_size', 4096)}")
        print(f"  - checkpoint: {dvae_config.get('use_checkpoint', False)}")
        print(f"  - parameters: {dvae_params:,} ({dvae_params/1e6:.2f}M)")
        print(f"  - trainable:  {dvae_trainable:,} ({dvae_trainable/1e6:.2f}M)")

    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)

    initializer = modules.build_initializer(model_config.initializer)
    encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder)

    target_encoder = None
    if model_config.target_encoder:
        target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")
        assert (
            model_config.target_encoder_input is not None
        ), "Please specify `target_encoder_input`."

    dynamics_predictor = None
    if model_config.dynamics_predictor:
        dynamics_predictor = modules.build_dynamics_predictor(model_config.dynamics_predictor)

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video":
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder:
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None:
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        if model_config.latent_processor:
            processor = modules.build_video(
                model_config.latent_processor,
                "LatentProcessor",
                corrector=grouper,
                predictor=predictor,
            )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
        processor = modules.ScanOverTime(processor)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    target_type = model_config.get("target_type", "features")
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None:
            default_target_key = "target_encoder.backbone_features"
        else:
            default_target_key = "encoder.backbone_features"
    else:
        raise ValueError(f"Unknown target type {target_type}. Should be `input` or `features`.")

    loss_defaults = {
        "pred_key": "decoder.reconstruction",
        "target_key": default_target_key,
        "video_inputs": input_type == "video",
        "patch_inputs": target_type == "features",
    }
    if model_config.losses is None:
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {}
        for name, loss_config in model_config.losses.items():
            # Handle STEVE losses
            # if loss_config.get('name') == 'DVAEReconstructionLoss':
            #     # Skip if dVAE is not enabled
            #     if not dvae_module:
            #         print(f"Skipping {name} because dVAE is not enabled")
            #         continue
            #     loss_fns[name] = DVAEReconstructionLoss(
            #         pred_key=loss_config.get('pred_key', 'dvae.reconstruction'),
            #         target_key=loss_config.get('target_key', input_type),
            #         normalize=loss_config.get('normalize', True),
            #         reduction=loss_config.get('reduction', 'mean'),
            #     )
            # elif loss_config.get('name') == 'STEVECrossEntropyLoss':
            #     # Skip if STEVE decoder is not enabled
            #     if not dvae_module:
            #         print(f"Skipping {name} because STEVE decoder is not enabled")
            #         continue
            #     loss_fns[name] = STEVECrossEntropyLoss(
            #         pred_key=loss_config.get('pred_key', 'decoder.logits'),
            #         target_key=loss_config.get('target_key', 'dvae.z_hard'),
            #         reduction=loss_config.get('reduction', 'mean'),
            #     )
            # else:
            # Standard SlotContrast losses
            loss_fns[name] = losses.build({**loss_defaults, **loss_config})

        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }

    if model_config.mask_resizers:
        mask_resizers = {
            name: modules.build_utils(resizer_config, "Resizer")
            for name, resizer_config in model_config.mask_resizers.items()
        }
    else:
        mask_resizers = {
            "decoder": modules.build_utils(
                {
                    "name": "Resizer",
                    # When using features as targets, assume patch-shaped outputs. With other
                    # targets, assume spatial outputs.
                    "patch_inputs": target_type == "features",
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
            "grouping": modules.build_utils(
                {
                    "name": "Resizer",
                    "patch_inputs": True,
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
        }

    if model_config.masks_to_visualize:
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = "decoder"

    gumbel_temp_scheduler = None
    if dvae_module:
        gumbel_start_temp = model_config.get('gumbel_start_temp', 1.0)
        gumbel_final_temp = model_config.get('gumbel_final_temp', 0.1)
        gumbel_anneal_steps = model_config.get('gumbel_anneal_steps', 10000)
        gumbel_temp_scheduler = STEVEGumbelTemperatureScheduler(
            start_temp=gumbel_start_temp,
            final_temp=gumbel_final_temp,
            anneal_steps=gumbel_anneal_steps,
        )

    model = ObjectCentricModel(
        optimizer_builder,
        initializer,
        encoder,
        processor,
        decoder,
        loss_fns,
        loss_weights=model_config.get("loss_weights", None),
        target_encoder=target_encoder,
        dynamics_predictor=dynamics_predictor,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        mask_resizers=mask_resizers,
        input_type=input_type,
        target_encoder_input=model_config.get("target_encoder_input", None),
        visualize=model_config.get("visualize", False),
        visualize_every_n_steps=model_config.get("visualize_every_n_steps", 1000),
        masks_to_visualize=masks_to_visualize,
        dvae=dvae_module,
        gumbel_temp_scheduler=gumbel_temp_scheduler
    )

    if model_config.load_weights:
        model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

    return model


class ObjectCentricModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_builder: Callable,
        initializer: nn.Module,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        loss_fns: Dict[str, losses.Loss],
        *,
        loss_weights: Optional[Dict[str, float]] = None,
        target_encoder: Optional[nn.Module] = None,
        dynamics_predictor: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        mask_resizers: Optional[Dict[str, modules.Resizer]] = None,
        input_type: str = "image",
        target_encoder_input: Optional[str] = None,
        visualize: bool = False,
        visualize_every_n_steps: Optional[int] = None,
        masks_to_visualize: Union[str, List[str]] = "decoder",
        dvae: Optional[nn.Module] = None,
        steve_decoder: Optional[nn.Module] = None,
        gumbel_temp_scheduler: Optional[Any] = None,
    ):
        super().__init__()
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.target_encoder = target_encoder
        self.dynamics_predictor = dynamics_predictor

        # STEVE components (optional)
        self.dvae = dvae
        self.steve_decoder = steve_decoder
        self.gumbel_temp_scheduler = gumbel_temp_scheduler

        if loss_weights is not None:
            # Filter out losses that are not used
            
            loss_fns_filtered = {k: loss for k, loss in loss_fns.items() if loss_weights[k] != 0.0}
            loss_weights_filtered = {
                k: loss for k, loss in loss_weights.items() if loss_weights[k] != 0.0
            }
            self.loss_fns = nn.ModuleDict(loss_fns_filtered)
            self.loss_weights = loss_weights_filtered
        else:
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = {}

        self.mask_resizers = mask_resizers if mask_resizers else {}
        self.mask_resizers["segmentation"] = modules.Resizer(
            video_inputs=input_type == "video", resize_mode="nearest-exact"
        )
        self.mask_soft_to_hard = modules.SoftToHardMask()
        self.train_metrics = torch.nn.ModuleDict(train_metrics)
        self.val_metrics = torch.nn.ModuleDict(val_metrics)

        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        if masks_to_visualize is None:
            masks_to_visualize = ["decoder"]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping", "dynamics_predictor"):
                raise ValueError(f"Unknown mask type {key}. Should be `decoder` or `grouping`.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]

        if input_type == "image":
            self.input_key = "image"
            self.expected_input_dims = 4
        elif input_type == "video":
            self.input_key = "video"
            self.expected_input_dims = 5
        else:
            raise ValueError(f"Unknown input type {input_type}. Should be `image` or `video`.")

        self.target_encoder_input_key = (
            target_encoder_input if target_encoder_input else self.input_key
        )

    def configure_optimizers(self):
        modules = {
            "initializer": self.initializer,
            "encoder": self.encoder,
            "processor": self.processor,
            "decoder": self.decoder,
        }
        if self.dynamics_predictor:
            modules["dynamics_predictor"] = self.dynamics_predictor
        if self.dvae:
            modules["dvae"] = self.dvae

        return self.optimizer_builder(modules)

    def forward(self, inputs: Dict[str, Any], verbose_shapes: bool = False) -> Dict[str, Any]:
        import gc

        encoder_input = inputs[self.input_key]  # batch [x n_frames] x n_channels x height x width
        assert encoder_input.ndim == self.expected_input_dims
        batch_size = len(encoder_input)

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]

        slots_initial = self.initializer(batch_size=batch_size)
        processor_output = self.processor(slots_initial, features)
        slots = processor_output["state"]
        decoder_output = self.decoder(slots)
        if verbose_shapes:
            print('features', features.shape)
            print('slots_initial', slots_initial.shape)
            print('slots', slots.shape)
            for key, value in decoder_output.items():
                print(f'\tdecoder_output[{key}]', value.shape)
        


        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }

        if self.dynamics_predictor:
            outputs["dynamics_predictor"] = self.dynamics_predictor(slots)
            predicted_slots = outputs["dynamics_predictor"].get("next_state")
            decoded_predicted_slots = self.decoder(predicted_slots)
            decoded_predicted_slots = {
                f"predicted_{key}": value for key, value in decoded_predicted_slots.items()
            }
            outputs["decoder"].update(decoded_predicted_slots)
            if verbose_shapes:
                print('dynamics_predictor', outputs["dynamics_predictor"].shape)
                print('predicted_slots', predicted_slots.shape)
                for key, value in decoded_predicted_slots.items():
                    print(f'\tdecoded_predicted_slots[{key}]', value.shape)

        # dVAE components processing
        if self.dvae is not None:
            # Clear cache before STEVE components (memory-intensive)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Get Gumbel temperature
            if self.gumbel_temp_scheduler and self.training:
                tau = self.gumbel_temp_scheduler.get_temperature(self.global_step)
            else:
                tau = 0.1  # Low temperature for evaluation

            # Process through dVAE
            if self.input_key == "video":
                # Handle video input
                B, T, C, H, W = encoder_input.shape
                encoder_input_flat = encoder_input.flatten(0, 1)
                dvae_output = self.dvae(encoder_input_flat, tau=tau, hard=True)
                for key, value in dvae_output.items():
                    if value.dim() >= 3:
                        dvae_output[key] = value.view(B, T, *value.shape[1:])
                    if verbose_shapes:
                        print('dvae_output', key, dvae_output[key].shape)
                # Reshape back to video
                dvae_output["z_hard"]= rearrange(dvae_output["z_hard"], 'b t v h w -> b t (h w) v')
                dvae_output["z_soft"]= rearrange(dvae_output["z_soft"], 'b t v h w -> b t (h w) v')
                dvae_output["z_logits"]= rearrange(dvae_output["z_logits"], 'b t v h w -> b t (h w) v')
                
            else:
                dvae_output = self.dvae(encoder_input, tau=tau, hard=True)

            outputs['dvae'] = dvae_output

            # # STEVE decoder if available
            # if self.steve_decoder is not None:
            #     slots = outputs['processor']['state']  # Get slots from processor

            #     # Prepare z_hard tokens
            #     if self.input_key == "video":
            #         B, T = dvae_output['z_hard'].shape[:2]
            #         # Flatten for decoder
            #         z_hard_flat = dvae_output['z_hard'].flatten(0, 1)
            #         # Reshape z_hard to (B*T, seq_len, vocab_size)
            #         z_hard_flat = z_hard_flat.permute(0, 2, 3, 1).flatten(1, 2)

            #         # Flatten slots too
            #         slots_flat = slots.flatten(0, 1)

            #         # Decode
            #         steve_output = self.steve_decoder(slots_flat, z_hard_flat)

            #         # Reshape back
            #         for key, value in steve_output.items():
            #             if value.dim() >= 2:
            #                 steve_output[key] = value.view(B, T, *value.shape[1:])
            #     else:
            #         # Reshape z_hard
            #         z_hard = dvae_output['z_hard'].permute(0, 2, 3, 1).flatten(1, 2)
            #         steve_output = self.steve_decoder(slots, z_hard)

            #     outputs['steve_decoder'] = steve_output

            #     # Clear cache after STEVE decoder (memory cleanup)
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
            #         gc.collect()

        outputs["targets"] = self.get_targets(inputs, outputs)

        # Final memory cleanup
        if self.dvae is not None and torch.cuda.is_available() and self.training:
            torch.cuda.empty_cache()

        return outputs

    def process_masks(
        self,
        masks: torch.Tensor,
        inputs: Dict[str, Any],
        resizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if masks is None:
            return None, None, None

        if resizer is None:
            masks_for_vis = masks
            masks_for_vis_hard = self.mask_soft_to_hard(masks)
            masks_for_metrics_hard = masks_for_vis_hard
        else:
            masks_for_vis = resizer(masks, inputs[self.input_key])
            masks_for_vis_hard = self.mask_soft_to_hard(masks_for_vis)
            target_masks = inputs.get("segmentations")
            if target_masks is not None and masks_for_vis.shape[-2:] != target_masks.shape[-2:]:
                masks_for_metrics = resizer(masks, target_masks)
                masks_for_metrics_hard = self.mask_soft_to_hard(masks_for_metrics)
            else:
                masks_for_metrics_hard = masks_for_vis_hard

        return masks_for_vis, masks_for_vis_hard, masks_for_metrics_hard

    @torch.no_grad()
    def aux_forward(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        decoder_masks = outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard, decoder_masks_metrics_hard = self.process_masks(
            decoder_masks, inputs, self.mask_resizers.get("decoder")
        )

        grouping_masks = outputs["processor"]["corrector"].get("masks")
        grouping_masks, grouping_masks_hard, grouping_masks_metrics_hard = self.process_masks(
            grouping_masks, inputs, self.mask_resizers.get("grouping")
        )

        aux_outputs = {}
        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_vis_hard"] = decoder_masks_hard
        if decoder_masks_metrics_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_metrics_hard
        if grouping_masks is not None:
            aux_outputs["grouping_masks"] = grouping_masks
        if grouping_masks_hard is not None:
            aux_outputs["grouping_masks_vis_hard"] = grouping_masks_hard
        if grouping_masks_metrics_hard is not None:
            aux_outputs["grouping_masks_hard"] = grouping_masks_metrics_hard

        if self.dynamics_predictor:
            dynamics_predictor_masks = outputs["decoder"].get("predicted_masks")
            (
                dynamics_predictor_masks,
                dynamics_predictor_masks_hard,
                dynamics_predictor_masks_metrics_hard,
            ) = self.process_masks(
                dynamics_predictor_masks, inputs, self.mask_resizers.get("decoder")
            )
            if dynamics_predictor_masks is not None:
                aux_outputs["dynamics_predictor_masks"] = dynamics_predictor_masks
            if dynamics_predictor_masks_hard is not None:
                aux_outputs["dynamics_predictor_masks_vis_hard"] = dynamics_predictor_masks_hard
            if dynamics_predictor_masks_metrics_hard is not None:
                aux_outputs["dynamics_predictor_masks_hard"] = dynamics_predictor_masks_metrics_hard

        return aux_outputs

    def get_targets(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.target_encoder:
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            targets[name] = loss_fn.get_target(inputs, outputs)

        return targets

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        for name, loss_fn in self.loss_fns.items():
            prediction = loss_fn.get_prediction(outputs)
            target = outputs["targets"][name]
            losses[name] = loss_fn(prediction, target)

        losses_weighted = [loss * self.loss_weights.get(name, 1.0) for name, loss in losses.items()]
        total_loss = torch.stack(losses_weighted).sum()

        return total_loss, losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self.forward(batch)
        if self.train_metrics or (
            self.visualize and self.trainer.global_step % self.visualize_every_n_steps == 0
        ):
            aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"train/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"train/{name}": loss for name, loss in losses.items()}
            to_log["train/loss"] = total_loss

        if self.train_metrics and self.dynamics_predictor:
            prediction_batch = copy.deepcopy(batch)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                if "predicted" in key.lower():
                    values = metric(**prediction_batch, **outputs, **aux_outputs)
                else:
                    values = metric(**batch, **outputs, **aux_outputs)
                self._add_metric_to_log(to_log, f"train/{key}", values)
                metric.reset()
        self.log_dict(to_log, on_step=True, on_epoch=False, batch_size=outputs["batch_size"])

        del outputs  # Explicitly delete to save memory

        if (
            self.visualize
            and self.trainer.global_step % self.visualize_every_n_steps == 0
            and self.global_rank == 0
        ):
            self._log_inputs(
                batch[self.input_key],
                {key: aux_outputs[f"{key}_hard"] for key in self.mask_keys_to_visualize},
                mode="train",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="train")

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        outputs = self.forward(batch)
        aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss

        if self.dynamics_predictor:
            prediction_batch = deepcopy(batch)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.val_metrics:
            for key, metric in self.val_metrics.items():
                if "predicted" in key.lower():
                    metric.update(**prediction_batch, **outputs, **aux_outputs)
                else:
                    metric.update(**batch, **outputs, **aux_outputs)

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
        )

        if self.visualize and batch_idx == 0 and self.global_rank == 0:
            masks_to_vis = {
                key: aux_outputs[f"{key}_vis_hard"] for key in self.mask_keys_to_visualize
            }
            if batch["segmentations"].shape[-2:] != batch[self.input_key].shape[-2:]:
                masks_to_vis["segmentations"] = self.mask_resizers["segmentation"](
                    batch["segmentations"], batch[self.input_key]
                )
            else:
                masks_to_vis["segmentations"] = batch["segmentations"]
            self._log_inputs(
                batch[self.input_key],
                masks_to_vis,
                mode="val",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="val")

    def on_validation_epoch_end(self):
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)

    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any], name: str, values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        if isinstance(values, dict):
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            log_dict[name] = values

    def _log_inputs(
        self,
        inputs: torch.Tensor,
        masks_by_name: Dict[str, torch.Tensor],
        mode: str,
        step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            self._log_video(f"{mode}/{self.input_key}", video, global_step=step)
            for mask_name, masks in masks_by_name.items():
                if "dynamics_predictor" in mask_name:
                    rollout_length = masks.shape[1]
                    trimmed_video = video[:, -rollout_length:]
                    video_with_masks = visualizations.mix_videos_with_masks(trimmed_video, masks)
                else:
                    video_with_masks = visualizations.mix_videos_with_masks(video, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_name}",
                    video_with_masks,
                    global_step=step,
                )
        elif self.input_key == "image":
            image = denorm(inputs)
            self._log_images(f"{mode}/{self.input_key}", image, global_step=step)
            for mask_name, masks in masks_by_name.items():
                image_with_masks = visualizations.mix_images_with_masks(image, masks)
                self._log_images(
                    f"{mode}/image_with_{mask_name}",
                    image_with_masks,
                    global_step=step,
                )
        else:
            raise ValueError(f"input_type should be 'image' or 'video', but got '{self.input_key}'")

    def _log_masks(
        self,
        aux_outputs,
        mask_keys=("decoder_masks",),
        mode="val",
        types: tuple = ("frames",),
        step: Optional[int] = None,
    ):
        if step is None:
            step = self.trainer.global_step
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]
                if self.input_key == "video":
                    _, f, n_obj, H, W = masks.shape
                    first_masks = masks[0].permute(1, 0, 2, 3)
                    first_masks_inverted = 1 - first_masks.reshape(n_obj, f, 1, H, W)
                    self._log_video(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                        types=types,
                    )
                elif self.input_key == "image":
                    _, n_obj, H, W = masks.shape
                    first_masks_inverted = 1 - masks[0].reshape(n_obj, 1, H, W)
                    self._log_images(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                    )
                else:
                    raise ValueError(
                        f"input_type should be 'image' or 'video', but got '{self.input_key}'"
                    )

    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
        types: tuple = ("frames",),
    ):
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            if "video" in types:
                logger.experiment.add_video(f"{name}/video", data, global_step=global_step)
            if "frames" in types:
                _, num_frames, _, _, _ = data.shape
                num_frames = min(max_frames, num_frames)
                data = data[:, :num_frames]
                data = data.flatten(0, 1)
                logger.experiment.add_image(
                    f"{name}/frames", make_grid(data, nrow=num_frames), global_step=global_step
                )

    def _save_video(self, name: str, data: torch.Tensor, global_step: int):
        assert (
            data.shape[0] == 1
        ), f"Only single videos saving are supported, but shape is: {data.shape}"
        data = data.cpu().numpy()[0].transpose(0, 2, 3, 1)
        data_dir = self.save_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"{global_step}.npy", data)

    def _log_images(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        n_examples = min(n_examples, data.shape[0])
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            logger.experiment.add_image(
                f"{name}/images", make_grid(data, nrow=n_examples), global_step=global_step
            )

    @staticmethod
    def _remove_padding(
        batch: Dict[str, Any], padding_mask: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        if torch.all(padding_mask):
            # Batch consists only of padding
            return None

        mask = ~padding_mask
        mask_as_idxs = torch.arange(len(mask))[mask.cpu()]

        output = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value[mask]
            elif isinstance(value, list):
                output[key] = [value[idx] for idx in mask_as_idxs]

        return output

    def _get_tensorboard_logger(self):
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                return self.logger

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint
        # as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""
        checkpoint = torch.load(checkpoint_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = utils.read_path(self, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
