import sys

if "Grounded-SAM" not in sys.path:
    sys.path.append("Grounded-SAM")

import os
from pathlib import Path
import random
# os.system("python -m pip install -e segment_anything")
# os.system("python -m pip install -e GroundingDINO")
# os.system("pip install --upgrade diffusers[torch]")
# os.system("pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel")
# Check if files exist before downloading
if not os.path.exists("groundingdino_swint_ogc.pth"):
    os.system("wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth")
else:
    print("groundingdino_swint_ogc.pth already exists. Skipping download.")

if not os.path.exists("sam_vit_h_4b8939.pth"):
    os.system("wget https://huggingface.co/spaces/mrtlive/segment-anything-model/resolve/main/sam_vit_h_4b8939.pth")
else:
    print("sam_vit_h_4b8939.pth already exists. Skipping download.")
sys.path.append(os.path.join(os.getcwd(), "Grounded-SAM", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-SAM", "segment_anything"))

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np

# diffusers
import torch
from diffusers import StableDiffusionInpaintPipeline


class Segmentor:
    def __init__(self):
        self.config_file = "Grounded-SAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.ckpt_filenmae = "groundingdino_swint_ogc.pth"
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def transform_image(self, image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image
    
    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        with torch.no_grad():
            # Only supports batch size 1 ??
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(
                    pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases
    
    def draw_mask(self, mask, draw, random_color=False):
        if random_color:
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255), 153)
        else:
            color = (30, 144, 255, 153)

        nonzero_coords = np.transpose(np.nonzero(mask))

        for coord in nonzero_coords:
            draw.point(coord[::-1], fill=color)
    
    def create_checkerboard(self, size, square_size=10):
        width, height = size
        image = Image.new('RGBA', size, (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        for x in range(0, width, square_size * 2):
            for y in range(0, height, square_size * 2):
                draw.rectangle([x, y, x + square_size, y + square_size], fill=(200, 200, 200, 255))
                draw.rectangle([x + square_size, y + square_size, x + square_size * 2, y + square_size * 2], fill=(200, 200, 200, 255))
        return image
    
    def run_grounded_sam(self, input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode):
        # load image
        image_pil = input_image.convert("RGB")
        transformed_image = self.transform_image(image_pil)

        groundingdino_model = self.load_model(self.config_file, self.ckpt_filenmae, device=self.device)

        if task_type == 'automatic' or task_type == 'inpainting' or task_type == 'det' or task_type == 'inpainting':
            raise NotImplementedError("Please implement from https://huggingface.co/spaces/linfanluntan/Grounded-SAM/blob/main/app.py")

        # run grounding dino model
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
        )

        size = image_pil.size

        # process boxes
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        # nms
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(
            boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")

        if task_type == 'seg' or task_type == 'inpainting' or task_type == 'automatic':
            # initialize SAM
            assert self.sam_checkpoint, 'sam_checkpoint is not found!'
            sam = build_sam(checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            sam_predictor = SamPredictor(sam)

            image = np.array(image_pil)
            sam_predictor.set_image(image)

            if task_type == 'automatic':
                # use NMS to handle overlapped boxes
                print(f"Revise caption with number: {text_prompt}")

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                boxes_filt, image.shape[:2]).to(self.device)

            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # masks: [1, 1, 512, 512]

        if task_type == 'det':
            image_draw = ImageDraw.Draw(image_pil)
            for box, label in zip(boxes_filt, pred_phrases):
                draw_box(box, image_draw, label)

            return [image_pil]
        elif task_type == 'seg' or task_type == 'automatic':
            mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)
            for mask in masks:
                self.draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
            
            # Create a checkerboard background
            checkerboard = np.array(self.create_checkerboard(size))
            # Create a blank RGBA image for the segmented result
            segmented_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
            # Create a composite mask
            composite_mask = np.zeros(size[::-1], dtype=np.uint8)
            for mask in masks:
                composite_mask |= mask[0].cpu().numpy()
            # Use the composite mask to extract the segmented portion
            image_array = np.array(image_pil)
            for c in range(3):  # RGB channels
                image_array[:,:,c] = image_array[:,:,c] * composite_mask + checkerboard[:,:,c] * (1 - composite_mask)
            # Convert back to PIL Image and paste onto the blank RGBA image
            extracted_segment = Image.fromarray(image_array)
            if extracted_segment.mode != 'RGBA':
                extracted_segment = extracted_segment.convert('RGBA')
            segmented_image.paste(extracted_segment, box=(0, 0), mask=extracted_segment)

            # image_draw = ImageDraw.Draw(image_pil)

            # for box, label in zip(boxes_filt, pred_phrases):
            #     draw_box(box, image_draw, label)

            # if task_type == 'automatic':
            #     image_draw.text((10, 10), text_prompt, fill='black')

            image_pil = image_pil.convert('RGBA')
            image_pil.alpha_composite(mask_image)
            return [image_pil, mask_image, segmented_image]
        elif task_type == 'inpainting':
            assert inpaint_prompt, 'inpaint_prompt is not found!'
            # inpainting pipeline
            if inpaint_mode == 'merge':
                masks = torch.sum(masks, dim=0).unsqueeze(0)
                masks = torch.where(masks > 0, True, False)
            # simply choose the first mask, which will be refine in the future release
            mask = masks[0][0].cpu().numpy()
            mask_pil = Image.fromarray(mask)

            if inpaint_pipeline is None:
                inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
                )
                inpaint_pipeline = inpaint_pipeline.to("cuda")

            image = inpaint_pipeline(prompt=inpaint_prompt, image=image_pil.resize(
                (512, 512)), mask_image=mask_pil.resize((512, 512))).images[0]
            image = image.resize(size)

            return [image, mask_pil]
        else:
            print("task_type:{} error!".format(task_type))


    def iter_images(self, image_folder: Path):
        for root, _, files in os.walk(image_folder):
            for filename in files:
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
                ):
                    image_path = Path(root) / filename
                    yield image_path

    def segment_images(
        self, image_folder: Path):
        for image_path in self.iter_images(image_folder):
            self.segment_image(image_path)

    def segment_image(
        self,
        image_path: Path,
    ):
        image_data = Image.open(image_path)
        image_pil, mask_image, segmented_image = self.run_grounded_sam(image_data, 'clothing. ', 'seg', '', 0.3, 0.25, 0.8, 'merge')
        segmented_image.save(image_path)



if __name__ == "__main__":
    segmentor = Segmentor()
    segmentor.segment_images(Path("/Users/divyanshumund/Downloads/northface_clothing/"))
