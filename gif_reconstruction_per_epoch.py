import torch
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def tensor_to_image(t):
    t = t.detach().cpu().numpy()
    t = np.squeeze(t)  # remove channel dim
    t = (t * 255).clip(0, 255).astype(np.uint8)
    return t


def concat_side_by_side(img_left, img_right):
    w, h = img_left.size
    combined = Image.new("L", (w * 2, h))
    combined.paste(img_left, (0, 0))
    combined.paste(img_right, (w, 0))
    return combined


def add_title(img, title_text):
    w, h = img.size
    title_height = 60

    new_img = Image.new("L", (w, h + title_height), 255)
    new_img.paste(img, (0, title_height))
    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), title_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = draw.textsize(title_text, font=font)

    text_x = (w - text_w) // 2
    text_y = (title_height - text_h) // 2

    draw.text((text_x, text_y), title_text, fill=0, font=font)
    return new_img


def add_under_labels(img, left_label, right_label):
    w, h = img.size
    label_height = 40
    new_img = Image.new("L", (w, h + label_height), 255)
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    def get_text_size(text):
        """Pillow-compatible text size (textbbox for new versions, textsize for old)."""
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            return draw.textsize(text, font=font)

    left_w, left_h = get_text_size(left_label)
    left_x = (w // 4) - (left_w // 2)
    left_y = h + (label_height - left_h) // 2

    right_w, right_h = get_text_size(right_label)
    right_x = (3 * w // 4) - (right_w // 2)
    right_y = h + (label_height - right_h) // 2

    draw.text((left_x, left_y), left_label, fill=0, font=font)
    draw.text((right_x, right_y), right_label, fill=0, font=font)
    return new_img


def create_reconstruction_progress_video(model, sample, device, save_path="reconstruction_progress.gif"):
    """ Creates a GIF/Video showing (GT | Reconstruction) per epoch
    with a title above each frame and labels under each image. """

    if sample.ndim == 3:
        sample = sample.unsqueeze(0)  # (1,1,H,W)
    sample = sample.to(device).float()

    gt = tensor_to_image(sample[0])
    gt_img = Image.fromarray(gt).convert("L")
    gt_img = gt_img.resize((320, 320), Image.NEAREST)

    frames = []

    if not hasattr(model, "epoch_reconstructions"):
        model.epoch_reconstructions = []

    for epoch_idx, recon in enumerate(model.epoch_reconstructions):

        recon_np = tensor_to_image(recon)
        recon_img = Image.fromarray(recon_np).convert("L")
        recon_img = recon_img.resize((320, 320), Image.NEAREST)

        side_by_side = concat_side_by_side(gt_img, recon_img)
        labeled = add_under_labels(side_by_side, "Ground Truth", "Reconstructed")

        end_idx = epoch_idx*10
        title_text = f"Reconstruction after {end_idx} epoch{'s' if epoch_idx != 1 else ''} of training"
        final_frame = add_title(labeled, title_text)

        frames.append(final_frame)

    if save_path.endswith(".gif"):
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=400,
            loop=0)
    else:
        writer = imageio.get_writer(save_path, fps=2)
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

    print(f"Reconstruction progress saved to: {save_path}")


def snapshot_reconstruction(model, sample, device):
    """ Store the reconstruction of `sample` at the current epoch inside the model. """
    model.eval()

    with torch.no_grad():
        if sample.ndim == 3:
            sample = sample.unsqueeze(0)

        sample = sample.to(device).float()
        recon = model(sample)[0]

        if not hasattr(model, "epoch_reconstructions"):
            model.epoch_reconstructions = []

        model.epoch_reconstructions.append(recon.cpu())
