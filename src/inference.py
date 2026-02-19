import os

import numpy as np
import torch
from PIL import Image
from lightning.fabric import Fabric


def infer(fabric: Fabric, model, infer_loader, save_dir, spatial_dims=2):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(infer_loader):
            inputs = batch["image"]
            output = model(inputs)

            result = torch.softmax(output, dim=1)
            result = torch.argmax(result, dim=1)

            result_np = result.cpu().numpy()[0]

            if spatial_dims == 3:
                import nibabel as nib

                out_path = os.path.join(save_dir, f"prediction_{idx:04d}.nii.gz")
                img = nib.Nifti1Image(result_np.astype(np.uint8), np.eye(4))
                nib.save(img, out_path)
            else:
                if result_np.ndim == 3:
                    result_np = result_np[result_np.shape[0] // 2]
                img = Image.fromarray((result_np * 255).astype("uint8"))
                img.save(os.path.join(save_dir, f"prediction_{idx:04d}.png"))

            print(
                f"Saved prediction_{idx:04d}.png"
                if spatial_dims == 2
                else f"Saved prediction_{idx:04d}.nii.gz"
            )
