
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def canny_control(img_rgb: Image.Image, low=80, high=160) -> Image.Image:
    """Return 3ch canny map (RGB) for ControlNet."""
    arr = pil_to_np(img_rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges3 = np.stack([edges, edges, edges], axis=2)
    return np_to_pil(edges3)


def blend_images(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    a_np = pil_to_np(a).astype(np.float32)
    b_np = pil_to_np(b).astype(np.float32)
    m = (1.0 - alpha) * a_np + alpha * b_np
    return np_to_pil(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="keyframe A path (e.g. 0000.png)")
    ap.add_argument("--b", required=True, help="keyframe B path (e.g. 0012.png)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=5, help="inbetween count")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--denoise", type=float, default=0.45)  # img2img strength
    ap.add_argument("--canny_low", type=int, default=80)
    ap.add_argument("--canny_high", type=int, default=160_OPCODE)
    ap.add_argument("--control_scale", type=float, default=1.0)

    # Models (change here if you prefer)
    ap.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--control_model", default="diffusers/controlnet-canny-sdxl-1.0")

    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = load_rgb(a_path)
    img_b = load_rgb(b_path)

    if img_a.size != img_b.size:
        raise ValueError(f"Resolution mismatch: A={img_a.size}, B={img_b.size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load ControlNet + SDXL img2img pipeline
    controlnet = ControlNetModel.from_pretrained(args.control_model, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )

    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention() if device == "cuda" else None

    # Prompt tuned for "line fidelity"
    prompt = (
        "clean anime lineart, thin consistent lines, smooth contour, "
        "accurate silhouette, inbetween frame, high line fidelity, minimal jitter"
    )
    negative = (
        "extra lines, broken lines, double contour, messy sketch, noise, wobble, "
        "deformation, melted, blur, missing parts, low contrast"
    )

    gen = torch.Generator(device=device).manual_seed(args.seed)

    meta = {
        "a": str(a_path),
        "b": str(b_path),
        "n": args.n,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "denoise": args.denoise,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "control_scale": args.control_scale,
        "base_model": args.base_model,
        "control_model": args.control_model,
        "device": device,
        "dtype": str(dtype),
        "frames": [],
    }

    # Generate inbetweens
    for i in range(1, args.n + 1):
        alpha = i / (args.n + 1)

        # init image = linear blend (this acts like "pose/motion hint")
        init_img = blend_images(img_a, img_b, alpha)

        # control image = canny of init (keeps edges stable)
        control_img = canny_control(init_img, low=args.canny_low, high=args.canny_high)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=init_img,
            control_image=control_img,
            strength=float(args.denoise),
            guidance_scale=float(args.cfg),
            num_inference_steps=int(args.steps),
            generator=gen,
            controlnet_conditioning_scale=float(args.control_scale),
        )

        frame = out.images[0]
        out_path = out_dir / f"{i:04d}.png"
        frame.save(out_path)

        meta["frames"].append({"index": i, "alpha": alpha, "path": str(out_path)})

        print(f"[frame {i:04d}] alpha={alpha:.4f} saved={out_path}")

    meta_path = out_dir / "_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
