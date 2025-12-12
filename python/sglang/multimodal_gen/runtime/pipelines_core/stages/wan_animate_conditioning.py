from typing import Any, List

import numpy as np

import PIL
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class WanVideoProcessor:
    def __init__(self, vae_scale_factor=8):
        self.vae_scale_factor = vae_scale_factor

    def preprocess_video(self, video: List[PIL.Image.Image]) -> torch.Tensor:
        if not isinstance(video, list):
            video = [video]

        frames = []
        for img in video:
            arr = np.array(img)
            tensor = torch.from_numpy(arr).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # CHW
            frames.append(tensor)

        video_tensor = torch.stack(frames, dim=1)
        video_tensor = 2.0 * video_tensor - 1.0

        # (1, C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)

        return video_tensor


class WanAnimateConditioningStage(PipelineStage):
    def __init__(self, vae: Any):
        super().__init__()
        self.vae = vae
        self.video_processor = WanVideoProcessor()

    def encode(
        self,
        video_condition: torch.Tensor,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Encode Image
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output: DiagonalGaussianDistribution = self.vae.encode(
                video_condition
            )

        generator = batch.generator

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()

        latent_condition = self.retrieve_latents(
            encoder_output, generator, sample_mode=sample_mode
        )
        latent_condition = server_args.pipeline_config.postprocess_vae_encode(
            latent_condition, self.vae
        )

        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latent_condition.device,
                dtype=latent_condition.dtype,
                vae=self.vae,
            )
        )

        # apply shift & scale if needed
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(latent_condition.device)

        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latent_condition.device)

        latent_condition -= shift_factor
        latent_condition = latent_condition * scaling_factor

        # output = server_args.pipeline_config.postprocess_image_latent(
        #     latent_condition, batch
        # )
        return latent_condition

    def retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self.vae = self.vae.to(get_local_torch_device())

        pose_video = batch.extra.get("pose_video")[:77]
        face_video = batch.extra.get("face_video")[:77]

        pose_video_tensor = self.video_processor.preprocess_video(pose_video).to(
            self.device, dtype=torch.float32
        )
        logger.info(f"[HZ] {pose_video_tensor.shape=}")
        # logger.info(f"[HZ] {pose_video_tensor=}")
        pose_latents_no_ref = self.encode(pose_video_tensor, batch, server_args)
        logger.info(f"[HZ] {pose_latents_no_ref.shape=}")
        # logger.info(f"[HZ] {pose_latents_no_ref=}")
        batch.extra["pose_hidden_states"] = pose_latents_no_ref

        face_video_tensor = self.video_processor.preprocess_video(face_video).to(
            self.device, dtype=torch.float32
        )
        batch.extra["face_pixel_values"] = face_video_tensor

        self.maybe_free_model_hooks()
        self.vae.to("cpu")

        return batch
