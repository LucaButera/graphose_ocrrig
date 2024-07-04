from abc import ABC
from collections import OrderedDict
from itertools import chain
from logging import getLogger
from typing import Any, Dict, Optional, Type

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.image import FrechetInceptionDistance

from nn.logging import MetricRegistry
from nn.loss import loss_by_name
from nn.models import GeneratorDiscriminatorModel, MaskGeneratorModel


class Predictor(LightningModule, ABC):
    def __init__(
        self,
        model_type: Type[Module],
        model_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 0.0002,
        b1: float = 0.0,
        b2: float = 0.999,
        min_lr_mul: float = 0.01,
        low_lr_mul: float = 0.01,
        peak_lr_mul: float = 10.0,
        ann_period: int = 50,
        loss: str = "BCEWL",
        loss_params: Optional[Dict[str, Any]] = None,
        accumulate_gradient: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.min_lr_mul = min_lr_mul
        self.low_lr_mul = low_lr_mul
        self.peak_lr_mul = peak_lr_mul
        self.ann_period = ann_period
        self.accumulate_gradient = accumulate_gradient
        assert self.accumulate_gradient > 0
        if model_kwargs is None:
            model_kwargs = {}
        self.model = model_type(**model_kwargs)
        self.loss = loss_by_name[loss](**loss_params if loss_params else {})
        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _step_or_accumulate(self, batch_idx, loss, optimizer, scheduler=None):
        if (batch_idx + 1) % self.accumulate_gradient == 0:
            optimizer.zero_grad()
            self.manual_backward(loss / self.accumulate_gradient)
            self.log_dict(grad_norm(self, norm_type="inf"))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            self.manual_backward(loss / self.accumulate_gradient)


class MaskPredictor(Predictor):
    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 0.001,
        b1: float = 0.0,
        b2: float = 0.999,
        min_lr_mul: float = 0.01,
        low_lr_mul: float = 0.01,
        peak_lr_mul: float = 10.0,
        ann_period: int = 50,
        loss: str = "BCEWL",
        loss_params: Optional[Dict[str, Any]] = None,
        accumulate_gradient: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            MaskGeneratorModel,
            model_kwargs,
            lr,
            b1,
            b2,
            min_lr_mul,
            low_lr_mul,
            peak_lr_mul,
            ann_period,
            loss,
            loss_params,
            accumulate_gradient,
            *args,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        metrics = MetricRegistry()
        masks, graphs = batch

        opt = self.optimizers()
        sch = self.lr_schedulers()

        # Generate fake batch
        generated_masks, _ = self.model.sample(graphs)

        loss = self.loss(generated_masks, masks)
        metrics.add("train/loss", loss)
        # Optimize Generator
        self._step_or_accumulate(batch_idx, loss, opt, sch)
        # Log Metrics
        self.log_dict(metrics.retrieve(), on_step=True)

    def validation_step(self, batch, batch_idx):
        masks, graphs = batch
        generated_masks, generated_node_masks = self.model.sample(graphs)
        loss = self.loss(generated_masks, masks)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.ann_period, eta_min=self.lr * self.min_lr_mul
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class GANPredictor(Predictor):
    def __init__(
        self,
        model_type: Type[Module] = GeneratorDiscriminatorModel,
        model_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 0.0002,
        b1: float = 0.0,
        b2: float = 0.999,
        min_lr_mul: float = 0.01,
        low_lr_mul: float = 0.01,
        peak_lr_mul: float = 10.0,
        ann_period: int = 300,
        loss: str = "Hinge",
        loss_params: Optional[Dict[str, Any]] = None,
        accumulate_gradient: int = 1,
        pretrain: Optional[str] = None,
        map_location: Optional[str] = None,
        freeze_pretrained: bool = False,
        ttur: float = 0.25,
        compute_fid_feature: int = 2048,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            model_type,
            model_kwargs,
            lr,
            b1,
            b2,
            min_lr_mul,
            low_lr_mul,
            peak_lr_mul,
            ann_period,
            loss,
            loss_params,
            accumulate_gradient,
            *args,
            **kwargs,
        )
        self.pretrain = pretrain
        self.g_lr = self.lr * self.peak_lr_mul
        if self.pretrain is not None:
            getLogger(self.__class__.__name__).info(
                f"Loading pretrained mask generator from {self.pretrain}"
            )
            d = torch.load(self.pretrain, map_location=map_location)
            extractor_state_dict = OrderedDict()
            for k in d["state_dict"]:
                if k.startswith("model.extractor."):
                    extractor_state_dict[k[16:]] = d["state_dict"][k]
            self.model.extractor.mask_extractor.load_state_dict(extractor_state_dict)
            if freeze_pretrained:
                for param in self.model.extractor.mask_extractor.parameters():
                    param.requires_grad = False
        self.automatic_optimization = False
        self.ttur = ttur
        self.compute_fid_feature = compute_fid_feature
        if self.ttur >= 1:
            self.d_lr = self.g_lr * self.ttur
        else:
            self.d_lr = self.g_lr
            self.g_lr *= self.ttur
        if self.compute_fid_feature > 0:
            self.fid = FrechetInceptionDistance(
                feature=self.compute_fid_feature,
                reset_real_features=False,
                normalize=True,
                compute_on_cpu=True,
            )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _update_fid(self, imgs, generated_imgs):
        if self.compute_fid_feature > 0:
            if self.current_epoch == 0:
                self.fid.update(imgs, real=True)
            self.fid.update(generated_imgs, real=False)

    def training_step(self, batch, batch_idx):
        metrics = MetricRegistry()
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        imgs, graphs = batch
        generated_imgs, layout, fake_masks, masks = self.model.sample(imgs, graphs)

        # Train Img Discriminator
        # Loss on real images
        real_prediction = self.model.discriminate(
            graphs.x,
            graphs.pos,
            graphs.edge_index,
            imgs,
            batch=graphs.batch,
            x_mask=graphs.x_mask if hasattr(graphs, "x_mask") else None,
        )
        d_loss_real = self.loss(real_prediction, mode="D_real")
        # Loss on fake images
        fake_prediction = self.model.discriminate(
            graphs.x,
            graphs.pos,
            graphs.edge_index,
            generated_imgs.detach(),
            batch=graphs.batch,
            x_mask=graphs.x_mask if hasattr(graphs, "x_mask") else None,
        )
        d_loss_fake = self.loss(fake_prediction, mode="D_fake")
        # Compute error of D as sum over the fake and the real batches
        d_loss = d_loss_real + d_loss_fake
        # Registering metrics
        metrics.add("train/discriminator/loss", d_loss)
        metrics.add("train/discriminator/loss/real", d_loss_real)
        metrics.add("train/discriminator/loss/fake", d_loss_fake)
        # Optimize Discriminator
        self._step_or_accumulate(batch_idx, d_loss, opt_d, sch_d)
        # Train Generator
        g_loss = self.loss(
            self.model.discriminate(
                graphs.x,
                graphs.pos,
                graphs.edge_index,
                generated_imgs,
                batch=graphs.batch,
                x_mask=graphs.x_mask if hasattr(graphs, "x_mask") else None,
            ),
            mode="G",
        )
        metrics.add("train/generator/loss", g_loss)
        # Optimize Generator
        self._step_or_accumulate(batch_idx, g_loss, opt_g, sch_g)
        # Log Metrics
        self.log_dict(metrics.retrieve(), on_step=True)

    def validation_step(self, batch, batch_idx):
        # Logging sample images
        if self.compute_fid_feature > 0:
            imgs, graphs = batch
            generated_imgs, layout, fake_masks, masks = self.model.sample(imgs, graphs)
            generated_imgs = generated_imgs.detach()
            self._update_fid(imgs, generated_imgs)

    def validation_epoch_end(self, validation_step_outputs):
        if self.compute_fid_feature > 0:
            self.log("val/fid", self.fid.compute())
            self.fid.reset()

    def configure_optimizers(self):
        if hasattr(self.model, "extractor"):
            g_params = [
                {
                    "params": chain(
                        self.model.extractor.feature_embedding.parameters(),
                        self.model.extractor.graph_feature_extractor.parameters(),
                        self.model.generator.parameters(),
                    )
                }
            ]
            if self.model.extractor.learn_masks:
                mask_params_group = {
                    "params": self.model.extractor.mask_extractor.parameters()
                }
                if self.pretrain is not None:
                    mask_params_group["lr"] = self.g_lr * self.low_lr_mul
                g_params.append(mask_params_group)
        else:
            g_params = self.model.generator.parameters()
        opt_g = torch.optim.Adam(
            g_params,
            lr=self.g_lr,
            betas=(self.b1, self.b2),
        )
        sch_g = CosineAnnealingWarmRestarts(
            opt_g, T_0=50, eta_min=self.g_lr * self.min_lr_mul
        )
        opt_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.d_lr,
            betas=(self.b1, self.b2),
        )
        optimizers = [opt_g, opt_d]
        sch_d = CosineAnnealingWarmRestarts(
            opt_d, T_0=50, eta_min=self.d_lr * self.min_lr_mul
        )
        schedulers = [sch_g, sch_d]
        return optimizers, schedulers
