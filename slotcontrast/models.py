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


def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config) # return a optimizer class with config, callable
    # return:
    # schedule_fn = {'name': 'exp_decay_with_warmup', 'warmup_steps': 2500, 'decay_steps': '${trainer.max_steps}'}
    # optimizer = torch.optim.Adam(itertools.chain.from_iterable(m.parameters() for m in modules.values()), lr=0.0004)
    # sheduler = schedulers.apply_schedule_fn_to_optimizer(optimizer, schedule_fn)
    # {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    # }

    initializer = modules.build_initializer(model_config.initializer)  #build from modules.initializers.build
    # make_build_fn: find name from config['name'], and instantiate that from config
    # return:
    # initializer = FixedLearnedInit(n_slots=NUM_SLOTS, dim=SLOT_DIM) 11/64
    
    encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    print("load encoder success")
    # import pdb; pdb.set_trace
    # images to patch tokens
    # return:
    # model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, checkpoint_path=None, {'dynamic_img_size': True})
    # features = ["blocks.11","blocks.11.attn.qkv"]
    # model = torchvision.models.feature_extraction.create_feature_extractor(model, features)
    # backbone = TimmExtractor(model,features,frozen,pretrained,)

    # output_transform=MLP(FEAT_DIM,SLOT_DIM,2*FEAT_DIM,True,"relu",False,False,"default",False)

    # slotcontrast.moduels.FrameEncoder(
    #     backbone,pos_embed=None
    #     output_transform,
    #     **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
    # )

    grouper = modules.build_grouper(model_config.grouper)
    # slots + token to slots + mask (use mlp)
    # grouper = modules.groupers.SlotAttention(
    #     inp_dim = SLOT_DIM,
    #     slot_dim = SLOT_DIM,
    #     n_iters = 2,
    #     use_mlp = True
    # )

    decoder = modules.build_decoder(model_config.decoder)
    # decoder = modules.decoders.MLPDecoder(inp_dim,outp_dim,hidden_dim,n_patches)
    # {
    #     'name': 'MLPDecoder', 
    #     'inp_dim': '${globals.SLOT_DIM}', 
    #     'outp_dim': '${globals.FEAT_DIM}', 
    #     'hidden_dims': [1024, 1024, 1024], 
    #     'n_patches': '${globals.NUM_PATCHES}'
    # }


    target_encoder = None
    if model_config.target_encoder:  #None
        target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")
        assert (
            model_config.target_encoder_input is not None
        ), "Please specify `target_encoder_input`."

    dynamics_predictor = None   #None
    if model_config.dynamics_predictor:
        dynamics_predictor = modules.build_dynamics_predictor(model_config.dynamics_predictor)

    input_type = model_config.get("input_type", "image")  #video
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video":
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder: #None
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None: # not None
            predictor = modules.build_module(model_config.predictor)
            # {
            #     'name': 'networks.TransformerEncoder', 
            #     'dim': '${globals.SLOT_DIM}', 
            #     'n_blocks': 1, 
            #     'n_heads': 4
            # }
        else:
            predictor = None
        if model_config.latent_processor: # not None
            # {'first_step_corrector_args': {'n_iters': 3}}
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

    target_type = model_config.get("target_type", "features")  #'features'
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None: #None
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
    if model_config.losses is None:  #not None
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }
        # 'losses': {
        #     'loss_featrec': {
        #         'name': 'MSELoss', 
        #         'pred_dims': [0, '${globals.FEAT_DIM}']
        #         }, 
        #     'loss_ss': {
        #         'name': 'Slot_Slot_Contrastive_Loss', 
        #         'pred_key': 'processor.state', 
        #         'temperature': 0.1, 
        #         'batch_contrast': True, 
        #         'patch_inputs': False, 
        #         'keep_input_dim': True}
        # }, 

    if model_config.mask_resizers:  #None
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

    if model_config.masks_to_visualize:  #None
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = "decoder"

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
    )
    # import pdb; pdb.set_trace()
    if model_config.load_weights:  #None
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
    ):
        super().__init__()
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.target_encoder = target_encoder
        self.dynamics_predictor = dynamics_predictor

        if loss_weights is not None:
            # Filter out losses that are not used
            assert (
                loss_weights.keys() == loss_fns.keys()
            ), f"Loss weight keys {loss_weights.keys()} != {loss_fns.keys()}"
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
        return self.optimizer_builder(modules)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # import pdb; pdb.set_trace()
        # {
        #     '__key__':['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'],
        #     '__url__':['./data/movi_c/movi_c-validation-000000.tar',...,'./data/movi_c/movi_c-validation-000000.tar']
        #     'segmentations': [16, 24, 11, 336, 336],
        #     'video': [16, 24, 3, 336, 336],
        #     'batch_padding_mask': [False,...,False],
        # }


        encoder_input = inputs[self.input_key]  # batch, frames, channels, width, height 
        # movi_c: [4, 24, 3, 336, 336] B,frames,channels,W,H
        # saycam: ([32, 3, 3, 224, 224])
        assert encoder_input.ndim == self.expected_input_dims
        batch_size = len(encoder_input)

        encoder_output = self.encoder(encoder_input)
        # features:           [4, 24, 576, 64]     B, frames, tokens, slots_dimension
        # backbone_features:  [4, 24, 576, 384]    B, frames, tokens, feats_dimension
        # 'vit_block_keys12': [4, 24, 576, 384]
        # 'vit_block12':      [4, 24, 576, 384]

        # features:           [32, 3, 256, 64]     
        # backbone_features:  [32, 3, 256, 768]
        # 'vit_block_keys12': [32, 3, 256, 768]
        # 'vit_block12':      [32, 3, 256, 768]

        features = encoder_output["features"] # [4, 24, 576, 64]

        slots_initial = self.initializer(batch_size=batch_size)
        # [4, 11, 64]  B, slos, slots_dimension
        # [32, 11, 64]
        
        processor_output = self.processor(slots_initial, features)
        # state:           [4, 24, 11, 64]   B, frames, slots, slots_dimension
        # state_predicted: [4, 24, 11, 64]   
        # corrector.slots: [4, 24, 11, 64]
        # corrector.masks: [4, 24, 11, 576]  B, frames, slots, patchs (336/14)**2

        # state:           [32, 3, 11, 64]
        # state_predicted: [32, 3, 11, 64]
        # corrector.slots: [32, 3, 11, 64]
        # corrector.masks: [32, 3, 11, 256]  B, frames, slots, patchs (224/14)**2

        slots = processor_output["state"]
        decoder_output = self.decoder(slots)
        # import pdb; pdb.set_trace()
        # reconstruction: [4, 24, 576, 384]   B, frames, patchs, feats_dimension / reconstruction for feats of each patch
        # masks:          [4, 24, 11, 576]    B, frames, slots, patchs / 

        # reconstruction: [32, 3, 256, 768]  shold be 256 instead of 576 !!
        # masks:          [32, 3, 11, 256]

        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }

        if self.dynamics_predictor: # None
            outputs["dynamics_predictor"] = self.dynamics_predictor(slots)
            predicted_slots = outputs["dynamics_predictor"].get("next_state")
            decoded_predicted_slots = self.decoder(predicted_slots)
            decoded_predicted_slots = {
                f"predicted_{key}": value for key, value in decoded_predicted_slots.items()
            }
            outputs["decoder"].update(decoded_predicted_slots)

        # import pdb; pdb.set_trace()
        outputs["targets"] = self.get_targets(inputs, outputs)

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
        # import pdb; pdb.set_trace()
        if self.target_encoder: # None
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            # import pdb; pdb.set_trace()
            targets[name] = loss_fn.get_target(inputs, outputs)
            # loss_featrec, 
            # MSELoss((to_canonical_dims): Rearrange('B F P D -> B (F P) D') (loss_fn): MSELoss())
            # [4, 13824, 384] B, F*P, D_feats  (24*576)
            # [32, 768, 768]  3*256=768
            # loss_ss 
            # Slot_Slot_Contrastive_Loss((to_canonical_dims): Identity() (criterion): CrossEntropyLoss())
            # [4, 24, 576, 384]
            # [32, 3, 256, 768]

        return targets

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        for name, loss_fn in self.loss_fns.items():
            # import pdb; pdb.set_trace()
            # loss_featrec 
            # MSELoss(
            #     (to_canonical_dims): Rearrange('B F P D -> B (F P) D')
            #     (loss_fn): MSELoss()
            # )
            prediction = loss_fn.get_prediction(outputs)
            target = outputs["targets"][name]  
            # something wrong with outputs['target'] shape
            # [4, 13824, 384], [4, 13824, 384]
            # [32, 768, 768], [32, 768, 768]
            
            losses[name] = loss_fn(prediction, target)    # here

        losses_weighted = [loss * self.loss_weights.get(name, 1.0) for name, loss in losses.items()]
        total_loss = torch.stack(losses_weighted).sum()

        return total_loss, losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self.forward(batch)
        # print(f'outputs:{outputs}')
        # import pdb; pdb.set_trace()
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

        # print("training step complete")
        # import pdb; pdb.set_trace()
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        outputs = self.forward(batch)
        # {
        #     'batch_size': 32,
            # 'encoder':{
            #     'features': torch.Size([32, 3, 257, 64]),
            #     'backbone_features':torch.Size([32, 3, 257, 768]),
            #     'vit_block_keys12':torch.Size([32, 3, 257, 768]),
            #     'vit_block12':torch.Size([32, 3, 257, 768])
            # },
        #     'processor':{
        #         'state':torch.Size([32, 3, 11, 64]), 
        #         'state_predicted':torch.Size([32, 3, 11, 64]),
        #         'corrector':{
        #             'slots':torch.Size([32, 3, 11, 64]), 
        #             'masks':torch.Size([32, 3, 11, 257])
        #         }
        #     },
        #     'decoder':{
        #         'reconstruction':torch.Size([32, 3, 576, 768]), 
        #         'masks':torch.Size([32, 3, 11, 576])
        #     },
        #     'targets':{
        #         'loss_featrec':torch.Size([32, 771, 768]),
        #         'loss_ss':torch.Size([32, 3, 257, 768]),
        #     }
        # }

        #batch:{
        #     (['__key__', 'video', 'batch_padding_mask'])
        # }
        # import pdb; pdb.set_trace()
        aux_outputs = self.aux_forward(batch, outputs)     # here
        print("<=== aux_forward Over ===>")
        # import pdb; pdb.set_trace()

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss

        if self.dynamics_predictor: # None
            prediction_batch = deepcopy(batch)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]
        # import pdb; pdb.set_trace()
        # ModuleDict(
        #     (ari): VideoARI()
        #     (image_ari): ImageARI()
        #     (mbo): VideoIoU()
        #     (image_mbo): ImageIoU()
        #     )
        if self.val_metrics:
            for key, metric in self.val_metrics.items():
                if "predicted" in key.lower():
                    metric.update(**prediction_batch, **outputs, **aux_outputs)
                else: # here
                    metric.update(**batch, **outputs, **aux_outputs)
                    # batch:
                    # ['__key__', '__url__', 'segmentations', 'video', 'batch_padding_mask']
                        # batch['__key__'] = ['32', '33', '34', '35']
                        # batch['__url__'] = ['./data/movi_c/movi_c-validation-000001.tar', './data/movi_c/movi_c-validation-000001.tar', './data/movi_c/movi_c-validation-000001.tar', './data/movi_c/movi_c-validation-000001.tar']
                        # batch['segmentations'].shape = [4, 24, 11, 336, 336]   B,frames,slots,H,W
                        # batch['video'].shape = [4, 24, 3, 336, 336]    
                        # batch['batch_padding_mask'] = tensor([False, False, False, False], device='cuda:0')
                    # outputs:
                    # ['batch_size', 'encoder', 'processor', 'decoder', 'targets']
                    # aux_outputs:
                    # ['decoder_masks', 'decoder_masks_vis_hard', 'decoder_masks_hard', 'grouping_masks', 'grouping_masks_vis_hard', 'grouping_masks_hard']

                    # batch:
                    # ['__key__', 'video', 'batch_padding_mask']
                    # outputs:
                    # ['batch_size', 'encoder', 'processor', 'decoder', 'targets']
                    # aux_outputs:
                    # ['decoder_masks', 'decoder_masks_vis_hard', 'decoder_masks_hard', 'grouping_masks', 'grouping_masks_vis_hard', 'grouping_masks_hard']

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
        )

        if self.visualize and batch_idx == 0 and self.global_rank == 0: # False
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
        
        print("<=== Validation Step Over ===>")
        # import pdb; pdb.set_trace()

    def validation_epoch_end(self, outputs):
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
        print(f'\n <===    Use Checkpoint {checkpoint_path}    =>\n')
        import time; time.sleep(3)
