from typing import Any, List

import numpy as np

import PIL
import torch
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self.vae = self.vae.to(get_local_torch_device())

        pose_video = batch.extra.get("pose_video")[:77]
        face_video = batch.extra.get("face_video")[:77]

        pose_video_tensor = self.video_processor.preprocess_video(pose_video).to(
            self.device, dtype=torch.float32
        )
        logger.info(f"[HZ] {pose_video_tensor.shape=}")
        pose_latents_no_ref = self.vae.encode(pose_video_tensor).mean.float()
        logger.info(f"[HZ] {pose_latents_no_ref.shape=}")
        batch.extra["pose_hidden_states"] = pose_latents_no_ref

        face_video_tensor = self.video_processor.preprocess_video(face_video).to(
            self.device, dtype=torch.float32
        )
        batch.extra["face_pixel_values"] = face_video_tensor

        self.vae.to("cpu")

        return batch
