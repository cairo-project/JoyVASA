# coding: utf-8

"""
Installable library API for JoyVASA.

This module wraps the CLI flow implemented in ``inference.py`` into a simple,
importable API split into two phases:

    * ``load_models(pretrained_weights_dir, device='cuda', ...)`` -> models
        Builds the inference/crop configs (with all checkpoint paths rooted at
        ``pretrained_weights_dir``) and instantiates the LivePortrait + JoyVASA
        motion-generator pipeline, loading every model into memory once.

    * ``generate(models, reference_image_path, audio_path, output_path, ...)``
        Runs audio-driven animation for a single reference image + audio pair,
        writing the resulting video to ``output_path`` and returning that path.

    * ``main()`` -- argparse based CLI entry point (``generate-joyvasa``).

The heavy lifting (motion generation + warping/decoding) is delegated to the
existing ``LivePortraitPipeline`` / ``LivePortraitPipelineAnimal`` classes; this
module only reorganizes construction so that model loading can be reused across
many ``generate`` calls.
"""

import os
import os.path as osp
import argparse
import shutil
import subprocess

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig


# --------------------------------------------------------------------------- #
# Checkpoint layout relative to a ``pretrained_weights`` directory.
# These mirror the hardcoded defaults in InferenceConfig / CropConfig.
# --------------------------------------------------------------------------- #
_INFERENCE_CKPT_LAYOUT = {
    "checkpoint_MotionGenerator": "JoyVASA/motion_generator/motion_generator_hubert_chinese.pt",
    "checkpoint_AudioEncoder": "hubert-base-ls960",
    "motion_template_path": "JoyVASA/motion_template/motion_template.pkl",
    "checkpoint_F": "liveportrait/base_models/appearance_feature_extractor.pth",
    "checkpoint_M": "liveportrait/base_models/motion_extractor.pth",
    "checkpoint_G": "liveportrait/base_models/spade_generator.pth",
    "checkpoint_W": "liveportrait/base_models/warping_module.pth",
    "checkpoint_S": "liveportrait/retargeting_models/stitching_retargeting_module.pth",
    "checkpoint_F_animal": "liveportrait_animals/base_models/appearance_feature_extractor.pth",
    "checkpoint_M_animal": "liveportrait_animals/base_models/motion_extractor.pth",
    "checkpoint_G_animal": "liveportrait_animals/base_models/spade_generator.pth",
    "checkpoint_W_animal": "liveportrait_animals/base_models/warping_module.pth",
    "checkpoint_S_animal": "liveportrait/retargeting_models/stitching_retargeting_module.pth",
}

_CROP_CKPT_LAYOUT = {
    "insightface_root": "insightface",
    "landmark_ckpt_path": "liveportrait/landmark.onnx",
    "xpose_ckpt_path": "liveportrait_animals/xpose.pth",
}


def _fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _resolve_device(device, inference_cfg, crop_cfg):
    """Translate a torch-style device string into the config's device fields."""
    if device is None:
        return
    device = str(device).strip().lower()
    if device in ("cpu",):
        inference_cfg.flag_force_cpu = True
        crop_cfg.flag_force_cpu = True
        return
    # cuda / cuda:N / gpu
    inference_cfg.flag_force_cpu = False
    crop_cfg.flag_force_cpu = False
    if ":" in device:
        try:
            device_id = int(device.split(":", 1)[1])
        except ValueError:
            device_id = 0
        inference_cfg.device_id = device_id
        crop_cfg.device_id = device_id


class JoyVASAModels(object):
    """Container holding a loaded JoyVASA pipeline and its configs."""

    def __init__(self, pipeline, inference_cfg, crop_cfg, animation_mode, device):
        self.pipeline = pipeline
        self.inference_cfg = inference_cfg
        self.crop_cfg = crop_cfg
        self.animation_mode = animation_mode
        self.device = device

    # dict-style access for convenience / backwards compat
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_models(pretrained_weights_dir, device="cuda", animation_mode="human", **_):
    """Load all JoyVASA / LivePortrait models.

    Args:
        pretrained_weights_dir: path to the ``pretrained_weights`` directory
            that contains ``JoyVASA/``, ``liveportrait/``,
            ``liveportrait_animals/``, ``hubert-base-ls960/``, ``insightface/``.
        device: torch-style device string (``'cuda'``, ``'cuda:0'`` or
            ``'cpu'``).
        animation_mode: ``'human'`` or ``'animal'`` -- selects which pipeline
            (and set of base models) to build.

    Returns:
        JoyVASAModels: an object (also dict-indexable) exposing ``pipeline``,
        ``inference_cfg``, ``crop_cfg``, ``animation_mode`` and ``device``.
    """
    pretrained_weights_dir = osp.abspath(osp.expanduser(pretrained_weights_dir))
    if not osp.isdir(pretrained_weights_dir):
        raise FileNotFoundError(
            f"pretrained_weights_dir not found: {pretrained_weights_dir}"
        )

    inference_cfg = InferenceConfig()
    crop_cfg = CropConfig()

    # Re-root every checkpoint path at the supplied weights directory.
    for attr, rel in _INFERENCE_CKPT_LAYOUT.items():
        setattr(inference_cfg, attr, osp.join(pretrained_weights_dir, rel))
    for attr, rel in _CROP_CKPT_LAYOUT.items():
        setattr(crop_cfg, attr, osp.join(pretrained_weights_dir, rel))

    _resolve_device(device, inference_cfg, crop_cfg)

    animation_mode = (animation_mode or "human").strip().lower()
    if animation_mode == "animal":
        from src.live_portrait_wmg_pipeline_animal import LivePortraitPipelineAnimal
        pipeline = LivePortraitPipelineAnimal(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
        )
    elif animation_mode == "human":
        from src.live_portrait_wmg_pipeline import LivePortraitPipeline
        pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
        )
    else:
        raise ValueError(
            f"Unknown animation_mode: {animation_mode!r} (expected 'human' or 'animal')"
        )

    # Best-effort resolved device string for reporting.
    resolved_device = getattr(
        getattr(pipeline, "live_portrait_wrapper", None)
        or getattr(pipeline, "live_portrait_wrapper_animal", None),
        "device",
        device,
    )

    return JoyVASAModels(pipeline, inference_cfg, crop_cfg, animation_mode, resolved_device)


def _build_args(models, reference_image_path, audio_path, output_dir, overrides):
    """Construct an ArgumentConfig for a single generate call."""
    args = ArgumentConfig()
    args.animation_mode = getattr(models, "animation_mode", "human")
    args.reference = reference_image_path
    args.audio = audio_path
    args.output_dir = output_dir

    inference_cfg = getattr(models, "inference_cfg", None)
    # Apply any user overrides to both the args and the (already-built) config
    # so that flags consumed inside execute() take effect.
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
        if inference_cfg is not None and hasattr(inference_cfg, key):
            setattr(inference_cfg, key, value)
    return args


def generate(models, reference_image_path, audio_path, output_path, **_):
    """Run audio-driven animation for one reference image + audio pair.

    Args:
        models: object returned by :func:`load_models`.
        reference_image_path: path to the source portrait / animal image.
        audio_path: path to the driving audio file.
        output_path: destination path for the generated ``.mp4`` video.
        **_: optional overrides forwarded to the ArgumentConfig / InferenceConfig
            (e.g. ``animation_region``, ``cfg_scale``, ``driving_multiplier``).

    Returns:
        str: the ``output_path`` that was written.
    """
    reference_image_path = osp.abspath(osp.expanduser(reference_image_path))
    audio_path = osp.abspath(osp.expanduser(audio_path))
    output_path = osp.abspath(osp.expanduser(output_path))

    if not osp.exists(reference_image_path):
        raise FileNotFoundError(f"reference image not found: {reference_image_path}")
    if not osp.exists(audio_path):
        raise FileNotFoundError(f"audio not found: {audio_path}")

    # Make bundled ffmpeg discoverable if present, then verify availability.
    ffmpeg_dir = osp.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
    if not _fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and "
            "ffprobe) before running JoyVASA. https://ffmpeg.org/download.html"
        )

    output_dir = osp.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    pipeline = getattr(models, "pipeline", None) if not isinstance(models, dict) else models["pipeline"]
    if pipeline is None:
        raise ValueError("models does not contain a 'pipeline'; call load_models first")

    args = _build_args(models, reference_image_path, audio_path, output_dir, _)

    # execute() writes a video named after the reference/audio basenames and
    # returns its path; move it to the requested output_path.
    produced = pipeline.execute(args)

    if produced and osp.abspath(produced) != output_path:
        shutil.move(produced, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="JoyVASA: audio-driven portrait/animal image animation."
    )
    parser.add_argument(
        "-w", "--pretrained-weights-dir", required=True,
        help="Path to the pretrained_weights directory.",
    )
    parser.add_argument(
        "-r", "--reference", required=True,
        help="Path to the source portrait/animal image.",
    )
    parser.add_argument(
        "-a", "--audio", required=True,
        help="Path to the driving audio file.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Destination path for the generated .mp4 video.",
    )
    parser.add_argument(
        "-m", "--animation-mode", default="human", choices=["human", "animal"],
        help="Animation mode / model set to use (default: human).",
    )
    parser.add_argument(
        "-d", "--device", default="cuda",
        help="Torch device string, e.g. 'cuda', 'cuda:0' or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--animation-region",
        choices=["exp", "pose", "lip", "eyes", "all"],
        default=None,
        help="Region to animate (default: config value).",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=None,
        help="Classifier-free guidance scale for motion generation.",
    )
    parser.add_argument(
        "--driving-multiplier", type=float, default=None,
        help="Driving motion multiplier.",
    )
    args = parser.parse_args()

    models = load_models(
        args.pretrained_weights_dir,
        device=args.device,
        animation_mode=args.animation_mode,
    )

    overrides = {}
    if args.animation_region is not None:
        overrides["animation_region"] = args.animation_region
    if args.cfg_scale is not None:
        overrides["cfg_scale"] = args.cfg_scale
    if args.driving_multiplier is not None:
        overrides["driving_multiplier"] = args.driving_multiplier

    output_path = generate(
        models,
        reference_image_path=args.reference,
        audio_path=args.audio,
        output_path=args.output,
        **overrides,
    )
    print(f"Saved animation to: {output_path}")


if __name__ == "__main__":
    main()
