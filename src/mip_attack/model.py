import os
from PIL import Image
from typing import Union, List
import logging
import random

import copy
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers import MllamaForConditionalGeneration, AutoProcessor

from collections import defaultdict

from mip_attack.artifacts import (
    RunArtifacts,
    atomic_save_rgb_png,
    atomic_torch_save,
    preemption_requested,
    preemption_signal,
    torch_load,
)
from mip_attack.optimization import (
    backward_model_loss,
    capture_rng_state,
    exact_token_match,
    retain_best_patch,
    restore_rng_state,
    sampled_token_log_likelihoods,
    trim_generated_ids,
)

def normalize_image(image, image_mean, image_std):
    image = (image - image_mean.view(-1, 1, 1).to(image.device)) / image_std.view(-1, 1, 1).to(image.device)
    return image

def unnormalize_image(image, image_mean, image_std):
    image = image * image_std.view(-1, 1, 1).to(image.device) + image_mean.view(-1, 1, 1).to(image.device)
    return image

def tile_image(pixel_values, num_tiles=4):

    assert len(pixel_values.shape) == 3, "Only single images are supported."
    # pixel_values shape: [3, 2H, 2W]

    tile_height = pixel_values.shape[1] // 2
    tile_width = pixel_values.shape[2] // 2

    tiles = []
    for j in range(num_tiles):
        y_start = (j // 2) * tile_height
        y_end = y_start + tile_height
        x_start = (j % 2) * tile_width
        x_end = x_start + tile_width
        tiles.append(pixel_values[:, y_start:y_end, x_start:x_end])

    tiled_image = torch.stack(tiles, dim=0)
    # pixel_values shape: [4, 3, H, W]

    return tiled_image

def untile_image(pixel_values, num_tiles=4):

    assert len(pixel_values.shape) == 4, "Only single images are supported."
    # pixel_values shape: [4, 3, H, W]

    tiles = []
    for j in range(num_tiles):
        rescaled_pixels = pixel_values[j, :, :, :]
        tiles.append(rescaled_pixels)

    combined_image = torch.cat((
            torch.cat((tiles[0], tiles[1]), dim=-1), # top row
            torch.cat((tiles[2], tiles[3]), dim=-1)  # bottom row
        ), dim=-2)
    # pixel_values shape: [3, 2H, 2W]

    return combined_image

def resize_image(pixel_values, target_height, target_width):

    assert len(pixel_values.shape) == 3, "Only single images are supported."
    # pixel_values shape: [3, 2H, 2W]

    # Note: align_corners=False, antialias=True is important to get similar behavior as PIL.Image.resize
    return F.interpolate(pixel_values.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False, antialias=True).squeeze(0)
                
def pad_image(pixel_values, added_height):

    assert len(pixel_values.shape) == 3, "Only single images are supported."
    # pixel_values shape: [3, 2H, 2W]

    return F.pad(pixel_values, (0, 0, 0, added_height), mode='constant', value=0)
    # padding_left, padding_right, padding_top, padding_bottom

def save_image(pixel_values, image_mean=None, image_std=None, image_height=None, image_width=None, file_destination=None, num_tiles=4):

    if file_destination is None:
        raise ValueError("file_destination is required.")

    if num_tiles == 4:
        assert len(pixel_values.shape) == 4, "Only single images are supported."
        # pixel_values shape: [4, 3, H, W]
        combined_image = untile_image(pixel_values, num_tiles).cpu()
        # pixel_values shape: [3, 2H, 2W]
    else:
        combined_image = pixel_values.cpu()

    assert len(combined_image.shape) == 3, "Only single images are supported."

    if image_height is not None and image_width is not None:
        combined_image = combined_image[:, :image_height, :image_width]

    if image_mean is not None and image_std is not None:
        combined_image = unnormalize_image(combined_image, image_mean, image_std)

    if combined_image.min() < 0 or combined_image.max() > 1:
        print(f"Warning: '{file_destination}' is in range [{round(combined_image.min().item(), 3)}, {round(combined_image.max().item(), 3)}]. Clipping pixel values !!")
        combined_image = torch.clip(combined_image, 0, 1)

    atomic_save_rgb_png(file_destination, combined_image)


def get_random_patch_positions(x_area, y_area, patch_size, max_num_patch_positions=1, overlapping=False):

    # return [{"x_range": [0, patch_size[0]], "y_range": [0, patch_size[1]]}, 
    #         {"x_range": [300, 300 + patch_size[0]], "y_range": [0, patch_size[1]]}]

    assert x_area[1] - x_area[0] >= patch_size[1], "Patch width is larger than x_area"
    assert y_area[1] - y_area[0] >= patch_size[0], "Patch height is larger than y_area"

    positions = []

    max_tries = 1000

    for _ in range(max_num_patch_positions):

        tries = 0
        while True:

            # breaking condition
            if tries >= max_tries:
                break
            tries += 1

            x_start = random.randint(x_area[0], x_area[1] - patch_size[1])
            y_start = random.randint(y_area[0], y_area[1] - patch_size[0])

            x_range = [x_start, x_start + patch_size[1]]
            y_range = [y_start, y_start + patch_size[0]]
            
            if not overlapping:
                overlap_found = False
                for pos in positions:
                    if not (
                        x_range[1] <= pos["x_range"][0] or  # No overlap on the left
                        x_range[0] >= pos["x_range"][1] or  # No overlap on the right
                        y_range[1] <= pos["y_range"][0] or  # No overlap above
                        y_range[0] >= pos["y_range"][1]     # No overlap below
                    ):
                        overlap_found = True
                        break
                
                if overlap_found:
                    continue
            
            positions.append({"x_range": x_range, "y_range": y_range})
            break
    
    return positions

def place_patch_on_image(pixel_values, patch, x_range, y_range, num_tiles=4):

    if num_tiles == 4:
        
        assert len(pixel_values.shape) == 4, "Only single images are supported."
        # pixel_values shape: [4, 3, H, W]

        nt, num_channels, tile_h, tile_w = pixel_values.shape
        assert nt == num_tiles, "Number of tiles does not match the image shape."

        pixel_values = untile_image(pixel_values, num_tiles)

        # pixel_values shape: [3, 2H, 2W]
        pixel_values[:, y_range[0]:y_range[1], x_range[0]:x_range[1]] = patch

        pixel_values = tile_image(pixel_values, 4)
        # pixel_values shape: [4, 3, H, W]


    else:
        assert len(pixel_values.shape) == 3, "Only single images are supported."

        pixel_values[:, y_range[0]:y_range[1], x_range[0]:x_range[1]] = patch
        # pixel_values shape: [3, 2H, 2W]

    return pixel_values

def get_patch_from_image(pixel_values, x_range, y_range, num_tiles=4):

    if num_tiles == 4:
        assert len(pixel_values.shape) == 4, "Only single images are supported."
        # pixel_values shape: [4, 3, H, W]

        num_tiles, num_channels, tile_h, tile_w = pixel_values.shape

        patch = untile_image(pixel_values, num_tiles)
        # pixel_values shape: [3, 2H, 2W]
        patch = patch[:, y_range[0]:y_range[1], x_range[0]:x_range[1]]

    else:
        assert len(pixel_values.shape) == 3, "Only single images are supported."
        # pixel_values shape: [3, H, W]

        patch = pixel_values[:, y_range[0]:y_range[1], x_range[0]:x_range[1]]
        # pixel_values shape: [3, H, W]

    return patch

def load_patch(image_path, patch_size):

    image = Image.open(image_path).convert("RGB")
    current_width, current_height = image.size
    current_aspect_ratio = current_width / current_height
    target_width, target_height = patch_size[1], patch_size[0]
    target_aspect_ratio = target_width / target_height
    
    # resize if patch is smaller than target size
    if current_width < target_width:
        image = image.resize((target_width, int(target_width / current_aspect_ratio)), resample=2)
        current_width, current_height = image.size
    if current_height < target_height:
        image = image.resize((int(target_height * current_aspect_ratio), target_height), resample=2)
        current_width, current_height = image.size
    
    # Calculate cropping box (centered)
    if target_aspect_ratio > 1:
        new_width = current_width
        new_height = int(current_width / target_aspect_ratio)
    else:
        new_height = current_height
        new_width = int(current_height * target_aspect_ratio)

    left = (current_width - new_width) // 2
    right = left + new_width
    top = (current_height - new_height) // 2
    bottom = top + new_height

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.crop((left, top, right, bottom))),
        transforms.Resize(patch_size),
        transforms.ToTensor(),
    ])
    adversarial_patch = transform(image)

    return adversarial_patch


class Llama32:
  
    def __init__(
        self,
        models,
        prepare_text_images_individually=False,
        model_revisions=None,
        gpu_memory_reserve_gib=2,
    ):

        self.models = models # list(set(models))
        self.prepare_text_images_individually = prepare_text_images_individually

        self.llama_image_sizes = {
            "Llama-3.2-11B-Vision-Instruct": (1120, 746), 
            "Llama-3.2-11B-Vision": (896, 597),
            "Llama-3.2-90B-Vision-Instruct": (1120, 746),
            "Llama-3.2-90B-Vision": (896, 597), 
        }

        self.vlms = {}
        self.processors = {}
        self.image_means = {}
        self.image_stds = {}

        model_revisions = model_revisions or {}
        for m, model in enumerate(self.models):
            assert "Llama-3.2-" in model, f"{model} not supported !!"

            # model_id = os.path.join("meta-llama", model)
            # revision = model_revisions.get(model)
            # revision_kwargs = {"revision": revision} if revision else {}
            local_model_id = os.path.join("/models-mip", model)
            if os.path.isdir(local_model_id):
                model_id = local_model_id
                revision_kwargs = {}
                print(f"Loading local model: {model_id}")
            else:
                model_id = os.path.join("meta-llama", model)
                revision = model_revisions.get(model)
                revision_kwargs = {"revision": revision} if revision else {}
                print(f"Loading Hugging Face model: {model_id}")

            processor = AutoProcessor.from_pretrained(model_id, **revision_kwargs)

            processor_chat_template = processor_tokenizer_chat_template = "not set"
            if "Instruct" in model: # only Instruction models have chat template
                # confirmed that chat_template and tokenizer.chat_template are the same for 11B and 90B
                processor_chat_template = processor.chat_template
                processor_tokenizer_chat_template = processor.tokenizer.chat_template
            else:
                if processor_chat_template == "not set":
                    instruct_model = "Llama-3.2-11B-Vision-Instruct"
                    instruct_revision = model_revisions.get(instruct_model)
                    temp_processor = AutoProcessor.from_pretrained(
                        os.path.join("meta-llama", instruct_model),
                        **(
                            {"revision": instruct_revision}
                            if instruct_revision
                            else {}
                        ),
                    )
                    processor_chat_template = temp_processor.chat_template
                    processor_tokenizer_chat_template = temp_processor.tokenizer.chat_template
                    del temp_processor

                print(f"Note: {model} processor chat template is empty !!!")
                processor.chat_template = processor_chat_template
                processor.tokenizer.chat_template = processor_tokenizer_chat_template

            self.processors[model] = processor
            self.image_means[model] = torch.tensor(processor.image_processor.image_mean)
            self.image_stds[model] = torch.tensor(processor.image_processor.image_std)
            # processor, image_mean, and image_std are different for each model

            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus} ({[torch.cuda.get_device_name(i) for i in range(num_gpus)]})")
            if num_gpus == 0:
                raise RuntimeError("Llama vision models require at least one CUDA GPU")
            reserve = int(gpu_memory_reserve_gib * 1024**3)
            max_memory = {
                gpu_id: max(1, (torch.cuda.mem_get_info(gpu_id)[0] - reserve) // 1024**3)
                for gpu_id in range(num_gpus)
            }
            max_memory = {
                gpu_id: f"{available_gib}GiB"
                for gpu_id, available_gib in max_memory.items()
            }

            self.vlms[model] = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                **revision_kwargs,
            )
            self.vlms[model].eval()
            # for name, param in self.vlm[model].named_parameters():
            #     print(f"Layer: {name} - Shape: {param.shape} - Device: {param.device} - Requires Grad: {param.requires_grad}")
            #     print("-"*50)

            for param in self.vlms[model].parameters():
                param.requires_grad = False
        
    def prepare_messages(self, system_prompt, num_images, question):

        messages = [
            [
                {  
                    "role": "user", # "system" -> jinja2.exceptions.TemplateError: Prompting with images is incompatible with system messages.
                    "content": [{"type": "text", "text": system_prompt}] + [{"type": "image"}] * num_images + [{"type": "text", "text": question}]  
                }  
            ],
        ]

        return messages

    def adversarial_attack(self, 
                           example_id,
                           system_prompt,
                           agent_input,
                           # question,
                           # images,
                           malicious_generation,
                           load_adversarial_patch,
                           crop_patch_from_image,
                           center_constraint_clipping_factor,
                           keep_specific_color,
                           max_num_patch_positions,
                           resized_values,
                           adv_patch_size,
                           adv_x_area,
                           adv_y_area,
                           adv_optimizer,
                           adv_lr,
                           betas,
                           increased_betas,
                           termination_likelihood,
                           random_pert_intensity,
                           settings_per_update,
                           epsilon,
                           max_optimizer_steps=100000,
                           checkpoint_interval=10,
                           resume=True,
                           reset_betas_on_collapse=False,
                           run_artifacts: RunArtifacts | None = None):

        assert max_num_patch_positions == 1, "Only max_num_patch_positions of 1 is supported !!"
        if max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive")
        if settings_per_update <= 0:
            raise ValueError("settings_per_update must be positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if run_artifacts is None:
            raise ValueError("run_artifacts is required for crafting")

        artifacts = run_artifacts.crafting(example_id)

        malicious_target_ids = self.processors[self.models[0]].tokenizer([malicious_generation], add_special_tokens=False, return_tensors="pt").input_ids
        print(f"Num. Target IDs: {malicious_target_ids.shape[-1]}")

        for i, input_dict in enumerate(agent_input):
            for j, image in enumerate(input_dict["images"]):
                if i == j == 0:
                    screen_width, screen_height = image.size
                    # original image: (3239, 2159)
                else:
                    assert image.size == (screen_width, screen_height), f"Screens have different size !!"
        
        # resize all values by resize_factor
        if resized_values:
            raise NotImplementedError("'resized_values' not supported !!")
            resize_factor = ...
            adv_patch_size = [int(adv_patch_size[0] / resize_factor), int(adv_patch_size[1] / resize_factor)]
            adv_x_area = [int(adv_x_area[0] / resize_factor), int(adv_x_area[1] / resize_factor)]
            adv_y_area = [int(adv_y_area[0] / resize_factor), int(adv_y_area[1] / resize_factor)]            
            

        print(f"adv_patch_size: {adv_patch_size}")
        print(f"adv_x_area: {adv_x_area}")
        print(f"adv_y_area: {adv_y_area}")

        #########################################
        #             Prepare input             #
        #########################################
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

        all_original_content = []
        for input_dict in agent_input:

            num_images = len(input_dict["images"])

            # ----- prepare image pixel values
            all_pixel_values = []
            for n in range(num_images):
                pixel_values = transform(input_dict["images"][n].convert("RGB"))
                all_pixel_values.append(pixel_values)
                assert torch.min(pixel_values) >= 0 and torch.max(pixel_values) <= 1, f"{torch.min(pixel_values)} >= 0 and {torch.max(pixel_values)} <= 1"

            # ----- prepare text input
            for q in input_dict["user_query"]:

                messages = self.prepare_messages(system_prompt=system_prompt, num_images=num_images, question=q)

                model_original_content = {}
                for model in self.models:
                    malicious_target_ids = self.processors[model].tokenizer([malicious_generation], add_special_tokens=False, return_tensors="pt").input_ids # .to(self.vlm.device)        
                    # shape: [batch_size, malicious_seq_len]

                    llama_text = self.processors[model].apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
                    llama_text[0] += malicious_generation
                    
                    if self.prepare_text_images_individually: # <--- encodes images and text differently !!
                        tokenized_text = self.processors[model].tokenizer(
                            text=llama_text,
                            return_tensors="pt"
                        )
                        # dict_keys(['input_ids', 'attention_mask'])

                        processed_images = self.processors[model].image_processor(
                            images=input_dict["images"],          # pixel value range: [0, 255]
                            # do_convert_rgb=True,
                            # do_normalize=True,      # normalized after rescaling based on distribution of dataset 
                            # do_rescale=True,        # 1 / 255
                            # do_pad=True,            # has to be true (quadradic images)
                            # do_resize=True,         # has to be true (quadradic images)
                            # "image_mean": [
                            #     0.48145466,
                            #     0.4578275,
                            #     0.40821073
                            # ],
                            # "image_processor_type": "MllamaImageProcessor",
                            # "image_std": [
                            #     0.26862954,
                            #     0.26130258,
                            #     0.27577711
                            # ],
                            # "max_image_tiles": 4,
                            # "resample": 2,  # PIL.Image.BILINEAR
                            # "rescale_factor": 0.00392156862745098,    # 1 / 255
                            # size={
                            #     "height": 560,    # 2 tiles: 1120 pixel (max)
                            #     "width": 560      # 2 tiles: 1120 pixel (max)
                            # },
                            return_tensors="pt"
                        )
                        # dict_keys(['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask', 'num_tiles'])
                        processed_images.pop("num_tiles")
                        
                        original_content = {k: v.detach() for k, v in {**tokenized_text, **processed_images}.items()}
                        # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask'])

                    else: 
                        original_content = self.processors[model](
                            text=llama_text, 
                            images=input_dict["images"],
                            return_tensors="pt",
                        )
                        # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask', 'cross_attention_mask'])
                        # input_ids: torch.Size([batch_size, prompt_len])
                        # attention_mask: torch.Size([batch_size, prompt_len])
                        # pixel_values: torch.Size([1, num_images, 4, 3, 560, 560])
                        # aspect_ratio_ids: torch.Size([1, batch_size])
                        # aspect_ratio_mask: torch.Size([1, batch_size, 4])
                        # cross_attention_mask <- missing
                        # cross_attention_mask torch.Size([batch_size, prompt_len, 1, 4])
                        
                        assert original_content['input_ids'][:, -malicious_target_ids.shape[1]:].equal(malicious_target_ids), f"{original_content['input_ids'][:, -malicious_target_ids.shape[1]:]} != {malicious_target_ids}"
                        assert original_content['pixel_values'].shape[-1] == original_content['pixel_values'].shape[-2] == max(self.llama_image_sizes[model]) // 2, f"{original_content['pixel_values'].shape[-1]} != {original_content['pixel_values'].shape[-2]} != {max(self.llama_image_sizes[model]) // 2}"

                    model_original_content[model] = {
                        "original_content": original_content, 
                        "pixel_values": all_pixel_values, 
                        "malicious_target_ids": malicious_target_ids, 
                        "num_images": num_images
                    }

                all_original_content.append(model_original_content)

        if not all_original_content:
            raise ValueError("At least one crafting setting is required")

        #########################################
        #             Create patch              #
        #########################################
        if load_adversarial_patch:
            assert not crop_patch_from_image, "Cannot load and crop adversarial patch at the same time !!"
            adversarial_patch = load_patch(
                image_path=load_adversarial_patch, 
                patch_size=adv_patch_size
            )
        elif crop_patch_from_image:
            pxl_values = all_original_content[0][self.models[0]]['pixel_values'][0].clone()
            # any image should have the same patch (create patch from first image in the list)
            if adv_x_area[1] - adv_x_area[0] != adv_patch_size[1] or  adv_y_area[1] - adv_y_area[0] != adv_patch_size[0]:
                print("Patch width and height does not fit adv_x_area and adv_y_area !!")

                x_center = int(adv_x_area[0] / 2 + adv_x_area[1] / 2)
                y_center = int(adv_y_area[0] / 2 + adv_y_area[1] / 2)
                patch_x_area = [x_center - int(adv_patch_size[1] / 2), x_center + int(adv_patch_size[1] / 2)]
                patch_y_area = [y_center - int(adv_patch_size[0] / 2), y_center + int(adv_patch_size[0] / 2)]

                assert patch_x_area[1] - patch_x_area[0] == adv_patch_size[1] and patch_y_area[1] - patch_y_area[0] == adv_patch_size[0]

                adversarial_patch = get_patch_from_image(
                    pixel_values=pxl_values,
                    x_range=patch_x_area,
                    y_range=patch_y_area,
                    num_tiles=1,
                )
            else:
                adversarial_patch = get_patch_from_image(
                    pixel_values=pxl_values,
                    x_range=adv_x_area,
                    y_range=adv_y_area,
                    num_tiles=1,
                )
        else:
            adversarial_patch = torch.ones(
                size=[3]+list(adv_patch_size), 
            )

        assert torch.min(adversarial_patch) >= 0 and torch.max(adversarial_patch) <= 1, f"{torch.min(adversarial_patch)} >= 0 and {torch.max(adversarial_patch)} <= 1"
        print(f"Adversarial Patch Shape: {adversarial_patch.shape}")

        # randomly perturbe pixel values
        if random_pert_intensity > 0:
            random_pertupation = torch.rand_like(adversarial_patch) * random_pert_intensity
            adversarial_patch = adversarial_patch - random_pertupation
            adversarial_patch = torch.clip(adversarial_patch, 0, 1)

        # define epsilon
        epsilon = torch.ones_like(adversarial_patch) * epsilon
        if center_constraint_clipping_factor > 0:
            # create min and max valid pixel ranges
            coords_y, coords_x = torch.meshgrid(
                torch.linspace(-1, 1, adv_patch_size[0]),
                torch.linspace(-1, 1, adv_patch_size[1])
            )
            distance_from_center = torch.sqrt(coords_x**2 + coords_y**2)
            distance_from_center = distance_from_center / distance_from_center.max()
            epsilon = (1 - distance_from_center * center_constraint_clipping_factor) * epsilon

            epsilon = torch.clip(epsilon, 0, 1)

        if keep_specific_color:
            # set epsilon to 0 (False) if the pixel has this specific color
            color_difference = torch.abs(adversarial_patch - torch.tensor(keep_specific_color).view(3, 1, 1))
            is_close_to_color = (color_difference <= 0.001).all(dim=0)
            epsilon = epsilon * (~is_close_to_color).float()

        sum_epsilon = torch.sum(epsilon).item()
        logging.warning(f"INFO: Sum Epsilon: {sum_epsilon}")

        min_initial_adversarial_patch = torch.clip(adversarial_patch - epsilon, 0, 1) #   * 255
        # min_initial_adversarial_patch = min_initial_adversarial_patch.to(torch.uint8).to(torch.float32) / 255

        max_initial_adversarial_patch = torch.clip(adversarial_patch + epsilon, 0, 1) #  * 255
        # max_initial_adversarial_patch = max_initial_adversarial_patch.to(torch.uint8).to(torch.float32) / 255

        # adversarial_patch = adversarial_patch.to(self.vlm.device)
        adversarial_patch.requires_grad = True

        #########################################
        #          Adversarial Search           #
        ######################################### 
        criterion = torch.nn.CrossEntropyLoss()

        if adv_optimizer.lower() == "adam":
            optimizer = torch.optim.Adam([adversarial_patch], lr=adv_lr, betas=betas)
        elif adv_optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD([adversarial_patch], lr=adv_lr)
        elif adv_optimizer.lower() == "sign":
            optimizer = None
        else:
            raise ValueError(f"Unsupported adversarial optimizer: {adv_optimizer}")

        patch_found, malicious_arg_max = [], []
        progress_bar = tqdm(
            desc="Crafting",
            total=max_optimizer_steps,
            unit="step",
            dynamic_ncols=True,
        )
        setting_step, optimizer_step, change = 0, 0, 0
        betas_increased = False
        avg_loss, avg_sequence_likelihood, avg_geometric_mean_likelihoods, avg_arithmetic_mean_likelihoods = [], [], [], []
        last_metrics = {}
        best_adversarial_patch = None
        best_metrics = None
        best_setting_step = None
        best_optimizer_step = None
        best_checkpoint_dirty = False

        old_adversarial_patch = adversarial_patch.clone() # <-- for logging purposes only

        logging.warning(f"INFO: Number different settings (images x prompts): {len(all_original_content)}")

        # --- Optimization loop
        all_ids = list(range(len(all_original_content)))
        setting_cursor = len(all_ids)
        use_random_pos = adv_x_area[1] - adv_x_area[0] != adv_patch_size[1] or adv_y_area[1] - adv_y_area[0] != adv_patch_size[0]
        print(f"Using random positions: {use_random_pos}")

        def update_crafting_progress(metrics, refresh=False):
            evaluations = int(metrics.get("evaluations", 0))
            exact_matches = int(metrics.get("argmax_matches", 0))
            targets_above_threshold = int(metrics.get("targets_above_threshold", 0))
            asr = exact_matches / evaluations if evaluations else 0.0
            progress_bar.set_postfix(
                {
                    "settings": setting_step,
                    "loss": f'{float(metrics.get("loss", 0.0)):.4f}',
                    "argmax": f"{exact_matches}/{evaluations}",
                    "argmax_rate": f"{asr:.4f}",
                    "threshold": f"{targets_above_threshold}/{evaluations}",
                    "sequence_p": f'{float(metrics.get("sequence_likelihood", 0.0)):.4f}',
                    "geometric_p": f'{float(metrics.get("geometric_mean_likelihood", 0.0)):.4f}',
                    "arithmetic_p": f'{float(metrics.get("arithmetic_mean_likelihood", 0.0)):.4f}',
                    "change": f'{float(metrics.get("change", 0.0)):.4f}',
                },
                refresh=refresh,
            )

        def save_best_checkpoint():
            nonlocal best_checkpoint_dirty
            if best_adversarial_patch is None or best_metrics is None:
                return
            checkpoint = {
                "schema_version": 1,
                "status": "best_observed",
                "example_id": example_id,
                "models": self.models,
                "settings_per_update": settings_per_update,
                "setting_step": best_setting_step,
                "optimizer_step": best_optimizer_step,
                "metrics": best_metrics,
                "adversarial_patch": best_adversarial_patch.detach().cpu(),
            }
            atomic_torch_save(artifacts.best_checkpoint_path, checkpoint)
            best_checkpoint_dirty = False
            artifacts.event(
                "best_checkpoint_saved",
                setting_step=best_setting_step,
                optimizer_step=best_optimizer_step,
                metrics=best_metrics,
                path=str(artifacts.best_checkpoint_path.relative_to(run_artifacts.root)),
            )

        def save_checkpoint(status):
            if best_checkpoint_dirty or (
                best_adversarial_patch is not None
                and not artifacts.best_checkpoint_path.exists()
            ):
                save_best_checkpoint()
            checkpoint = {
                "schema_version": 1,
                "status": status,
                "example_id": example_id,
                "models": self.models,
                "settings_per_update": settings_per_update,
                "setting_step": setting_step,
                "optimizer_step": optimizer_step,
                "adversarial_patch": adversarial_patch.detach().cpu(),
                "adversarial_patch_grad": (
                    adversarial_patch.grad.detach().cpu()
                    if status == "preempted" and setting_step % settings_per_update != 0
                    else None
                ),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "min_initial_adversarial_patch": min_initial_adversarial_patch,
                "max_initial_adversarial_patch": max_initial_adversarial_patch,
                "old_adversarial_patch": old_adversarial_patch,
                "change": change,
                "betas_increased": betas_increased,
                "all_ids": all_ids,
                "setting_cursor": setting_cursor,
                "last_metrics": last_metrics,
                "pending_metrics": {
                    "loss": avg_loss,
                    "sequence_likelihood": avg_sequence_likelihood,
                    "geometric_mean_likelihood": avg_geometric_mean_likelihoods,
                    "arithmetic_mean_likelihood": avg_arithmetic_mean_likelihoods,
                    "patch_found": patch_found,
                    "malicious_arg_max": malicious_arg_max,
                },
                "rng_state": capture_rng_state(),
            }
            atomic_torch_save(artifacts.checkpoint_path, checkpoint)
            artifacts.event(
                "checkpoint_saved",
                status=status,
                setting_step=setting_step,
                optimizer_step=optimizer_step,
                path=str(artifacts.checkpoint_path.relative_to(run_artifacts.root)),
            )

        def finish(status, target_reached):
            save_checkpoint(status)
            selected_patch = (
                best_adversarial_patch
                if best_adversarial_patch is not None
                else adversarial_patch
            )
            save_image(
                pixel_values=selected_patch.detach().cpu(),
                file_destination=artifacts.final_patch_path,
                num_tiles=1,
            )
            summary = {
                "target_reached": target_reached,
                "setting_step": setting_step,
                "optimizer_step": optimizer_step,
                "max_optimizer_steps": max_optimizer_steps,
                "settings_per_update": settings_per_update,
                "metrics": last_metrics,
                "best_metrics": best_metrics,
                "best_setting_step": best_setting_step,
                "best_optimizer_step": best_optimizer_step,
                "patch_selection": (
                    "best_observed_boundary"
                    if best_adversarial_patch is not None
                    else "current_patch"
                ),
                "patch": str(artifacts.final_patch_path.relative_to(run_artifacts.root)),
                "checkpoint": str(artifacts.checkpoint_path.relative_to(run_artifacts.root)),
                "best_checkpoint": (
                    str(artifacts.best_checkpoint_path.relative_to(run_artifacts.root))
                    if artifacts.best_checkpoint_path.exists()
                    else None
                ),
            }
            if status == "preempted":
                summary["signal"] = preemption_signal()
            artifacts.write_summary(status, **summary)
            artifacts.event("crafting_finished", status=status, **summary)
            progress_bar.close()
            return selected_patch.detach().cpu()

        if resume and artifacts.checkpoint_path.exists():
            checkpoint = torch_load(artifacts.checkpoint_path)
            if checkpoint.get("schema_version") != 1:
                raise ValueError("Unsupported crafting checkpoint schema")
            if checkpoint["example_id"] != example_id:
                raise ValueError("Crafting checkpoint example mismatch")
            if checkpoint["models"] != self.models:
                raise ValueError("Crafting checkpoint model mismatch")
            if checkpoint["settings_per_update"] != settings_per_update:
                raise ValueError("Crafting checkpoint settings_per_update mismatch")

            with torch.no_grad():
                adversarial_patch.copy_(checkpoint["adversarial_patch"])
            if checkpoint.get("adversarial_patch_grad") is not None:
                adversarial_patch.grad = checkpoint["adversarial_patch_grad"].to(
                    adversarial_patch.device
                )
            if optimizer is not None:
                if checkpoint["optimizer"] is None:
                    raise ValueError("Crafting checkpoint optimizer mismatch")
                optimizer.load_state_dict(checkpoint["optimizer"])
            elif checkpoint["optimizer"] is not None:
                raise ValueError("Crafting checkpoint optimizer mismatch")

            min_initial_adversarial_patch = checkpoint["min_initial_adversarial_patch"]
            max_initial_adversarial_patch = checkpoint["max_initial_adversarial_patch"]
            old_adversarial_patch = checkpoint["old_adversarial_patch"]
            setting_step = checkpoint["setting_step"]
            optimizer_step = checkpoint["optimizer_step"]
            change = checkpoint["change"]
            betas_increased = checkpoint["betas_increased"]
            all_ids = checkpoint["all_ids"]
            setting_cursor = checkpoint["setting_cursor"]
            last_metrics = checkpoint.get("last_metrics", {})
            pending_metrics = checkpoint.get("pending_metrics", {})
            avg_loss = pending_metrics.get("loss", [])
            avg_sequence_likelihood = pending_metrics.get("sequence_likelihood", [])
            avg_geometric_mean_likelihoods = pending_metrics.get("geometric_mean_likelihood", [])
            avg_arithmetic_mean_likelihoods = pending_metrics.get("arithmetic_mean_likelihood", [])
            patch_found = pending_metrics.get("patch_found", [])
            malicious_arg_max = pending_metrics.get("malicious_arg_max", [])
            restore_rng_state(checkpoint["rng_state"])

            if artifacts.best_checkpoint_path.exists():
                best_checkpoint = torch_load(artifacts.best_checkpoint_path)
                if best_checkpoint.get("schema_version") != 1:
                    raise ValueError("Unsupported best-patch checkpoint schema")
                if best_checkpoint["example_id"] != example_id:
                    raise ValueError("Best-patch checkpoint example mismatch")
                if best_checkpoint["models"] != self.models:
                    raise ValueError("Best-patch checkpoint model mismatch")
                if best_checkpoint["settings_per_update"] != settings_per_update:
                    raise ValueError("Best-patch checkpoint settings_per_update mismatch")
                best_adversarial_patch = best_checkpoint["adversarial_patch"].to(
                    adversarial_patch.device
                )
                best_metrics = best_checkpoint["metrics"]
                best_setting_step = best_checkpoint["setting_step"]
                best_optimizer_step = best_checkpoint["optimizer_step"]
                artifacts.event(
                    "best_checkpoint_loaded",
                    setting_step=best_setting_step,
                    optimizer_step=best_optimizer_step,
                    path=str(
                        artifacts.best_checkpoint_path.relative_to(run_artifacts.root)
                    ),
                )
            artifacts.event(
                "crafting_resumed",
                checkpoint_status=checkpoint["status"],
                setting_step=setting_step,
                optimizer_step=optimizer_step,
            )
            progress_bar.n = optimizer_step
            if last_metrics:
                update_crafting_progress(last_metrics)
            progress_bar.refresh()

            if checkpoint["status"] in {"target_reached", "max_optimizer_steps"}:
                return finish(
                    checkpoint["status"],
                    target_reached=checkpoint["status"] == "target_reached",
                )
        else:
            artifacts.event(
                "crafting_started",
                max_optimizer_steps=max_optimizer_steps,
                settings_per_update=settings_per_update,
                checkpoint_interval=checkpoint_interval,
                reset_betas_on_collapse=reset_betas_on_collapse,
            )

        while optimizer_step < max_optimizer_steps:

            if setting_cursor >= len(all_ids):
                random.shuffle(all_ids)
                setting_cursor = 0

            content_dict = all_original_content[all_ids[setting_cursor]]
            setting_cursor += 1
            # random.randint(0, len(all_original_content) - 1)

            # adversarial_patch = adversarial_patch.detach().requires_grad_(True)

            setting_step += 1
            for m, model in enumerate(self.models):

                content = copy.deepcopy(content_dict[model]['original_content']) # .to(self.vlm.device)
                malicious_target_ids = content_dict[model]['malicious_target_ids'].clone() # .to(self.vlm.device)

                # --- add patches to batch
                for n in range(content_dict[model]["num_images"]):

                    if use_random_pos:
                        assert max_num_patch_positions == 1, "Only max_num_patch_positions of 1 is supported !!"
                        rp = get_random_patch_positions(x_area=adv_x_area, 
                                                        y_area=adv_y_area, 
                                                        patch_size=adv_patch_size, 
                                                        max_num_patch_positions=max_num_patch_positions, 
                                                        overlapping=False)[0]
                    else:
                        rp = {"x_range": adv_x_area, "y_range": adv_y_area}

                    resized_image_with_patch = place_patch_on_image(
                        pixel_values=content_dict[model]['pixel_values'][n].clone(), # .to(self.vlm.device), # [0, 1]
                        patch=adversarial_patch, # <-- placing adversarial patch [0, 1]
                        num_tiles=1,
                        **rp,
                    )

                    resized_image_with_patch = resize_image( # <-- does not lead to [0, 1] but [0, 1.000000238418579]
                            pixel_values=resized_image_with_patch,
                            target_height=self.llama_image_sizes[model][1], 
                            target_width=self.llama_image_sizes[model][0],
                        )

                    # logging.warning(f"\nresize_image - min: {torch.min(resized_image_with_patch).item()} max: {torch.max(resized_image_with_patch).item()}")

                    resized_image_with_patch = pad_image(
                        pixel_values=resized_image_with_patch,
                        added_height=self.llama_image_sizes[model][0] - self.llama_image_sizes[model][1],
                    )

                    resized_image_with_patch = tile_image(
                        normalize_image(
                            image=resized_image_with_patch,
                            image_mean=self.image_means[model], 
                            image_std=self.image_stds[model],
                        ), num_tiles=4)

                    assert resized_image_with_patch.shape == content['pixel_values'][0, n, :, :, :, :].shape, f"{resized_image_with_patch.shape} != {content['pixel_values'][0, n, :, :, :, :].shape}"
                    content['pixel_values'][0, n, :, :, :, :] = resized_image_with_patch

                outputs = self.vlms[model](
                    use_cache=False,
                    **content.to(self.vlms[model].device),
                )
                logits = outputs.logits
                vocab_size = logits.shape[-1] # 128256
                # logits shape: torch.Size([batch_size, prompt_len, vocab_size])

                target_start_idx = content['input_ids'].shape[-1] - malicious_target_ids.shape[-1] - 1
                target_logits = logits[:, target_start_idx:-1, :]
                # shape: [batch_size, malicious_seq_len, vocab_size]

                target_ids = malicious_target_ids.to(target_logits.device)
                loss = criterion(
                    target_logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                )
                backward_model_loss(
                    loss,
                    model_count=len(self.models),
                    settings_per_update=settings_per_update,
                )

                with torch.no_grad():
                    detached_logits = target_logits.detach()
                    ll = detached_logits.gather(
                        -1, target_ids.unsqueeze(-1)
                    ).squeeze(-1) - torch.logsumexp(detached_logits, dim=-1)
                    # [batch_size, malicious_seq_len]
                    token_likelihoods = torch.exp(ll)
                    pred_ids = torch.argmax(detached_logits, dim=-1)
                    # shape: [batch_size, malicious_seq_len]

                    avg_loss.append(loss.item())
                    avg_sequence_likelihood.append(torch.exp(torch.sum(ll)).item())
                    avg_geometric_mean_likelihoods.append(torch.mean(torch.exp(torch.mean(ll, dim=-1))).item())
                    avg_arithmetic_mean_likelihoods.append(torch.mean(token_likelihoods).item())

                    if torch.all(pred_ids == target_ids):
                        malicious_arg_max.append(True)
                    else:
                        malicious_arg_max.append(False)

                    # breaking condition
                    if torch.all(token_likelihoods > termination_likelihood):
                        patch_found.append(True)
                    else:
                        patch_found.append(False)

                del content, detached_logits, logits, loss, outputs, target_ids, target_logits


            # A partial accumulation window is resumable because the patch gradient
            # and pending metrics are stored; no model graph is retained.
            if preemption_requested() and setting_step % settings_per_update != 0:
                return finish("preempted", target_reached=False)

            if setting_step % settings_per_update == 0:

                with torch.no_grad():

                    avg_loss = sum(avg_loss) / len(avg_loss)
                    avg_sequence_likelihood = sum(avg_sequence_likelihood) / len(avg_sequence_likelihood)
                    avg_geometric_mean_likelihoods = sum(avg_geometric_mean_likelihoods) / len(avg_geometric_mean_likelihoods)
                    avg_arithmetic_mean_likelihoods = sum(avg_arithmetic_mean_likelihoods) / len(avg_arithmetic_mean_likelihoods)

                    last_metrics = {
                        "sum_epsilon": sum_epsilon,
                        "loss": avg_loss,
                        "argmax_matches": sum(malicious_arg_max),
                        "evaluations": len(malicious_arg_max),
                        "targets_above_threshold": sum(patch_found),
                        "sequence_likelihood": avg_sequence_likelihood,
                        "geometric_mean_likelihood": avg_geometric_mean_likelihoods,
                        "arithmetic_mean_likelihood": avg_arithmetic_mean_likelihoods,
                        "change": change,
                    }
                    target_reached = all(patch_found)
                    (
                        best_adversarial_patch,
                        best_metrics,
                        best_patch_updated,
                    ) = retain_best_patch(
                        adversarial_patch,
                        last_metrics,
                        best_adversarial_patch,
                        best_metrics,
                    )
                    if best_patch_updated:
                        best_setting_step = setting_step
                        best_optimizer_step = optimizer_step
                        best_checkpoint_dirty = True
                    artifacts.event(
                        "optimization_boundary",
                        setting_step=setting_step,
                        optimizer_step=optimizer_step,
                        target_reached=target_reached,
                        best_patch_updated=best_patch_updated,
                        metrics=last_metrics,
                    )

                    if target_reached:
                        update_crafting_progress(last_metrics, refresh=True)
                        return finish("target_reached", target_reached=True)

                    if adv_optimizer.lower() == "sign":
                        updated_values = adversarial_patch - adversarial_patch.grad.sign()
                        adversarial_patch.copy_(updated_values)
                    else:
                        if adv_optimizer.lower() == "adam" and avg_sequence_likelihood > 0.5 and not betas_increased:
                            betas_increased = True
                            for param_group in optimizer.param_groups:
                                param_group['betas'] = increased_betas

                        # Published runs effectively kept ``increased_betas`` here:
                        # the old "decrease" branch assigned the increased values
                        # again. Configs retain that behavior unless reset is enabled.
                        if (
                            adv_optimizer.lower() == "adam"
                            and reset_betas_on_collapse
                            and max(avg_geometric_mean_likelihoods, avg_arithmetic_mean_likelihoods) < 0.01
                            and betas_increased
                        ):
                            betas_increased = False
                            for param_group in optimizer.param_groups:
                                param_group['betas'] = betas

                        optimizer.step()

                    optimizer_step += 1

                    # -- extract patches from batch
                    # all_adversarial_patchs = []
                    # for b in range(model_batch_size):
                    #     all_adversarial_patchs.append(get_patch_from_image(content['pixel_values'][0, b, :, :, :, :], **all_random_positions[b]))
                    # adversarial_patch = torch.mean(torch.stack(all_adversarial_patchs, dim=0), dim=0)
        
                    # clip pixels to epsilon-range and in uint8 quantized form
                    clipped_values = torch.round(torch.clip(adversarial_patch.cpu(), min=min_initial_adversarial_patch, max=max_initial_adversarial_patch) * 255) / 255.0
                    adversarial_patch.copy_(clipped_values)

                    change = torch.sum(torch.abs(old_adversarial_patch - adversarial_patch)).item()
                    last_metrics["change"] = change
                    update_crafting_progress(last_metrics)
                    progress_bar.update(1)

                    old_adversarial_patch = adversarial_patch.clone().detach() # <-- for logging purposes only
                    # old_adversarial_patch_int_gpu = (old_adversarial_patch.clone() * 255).to(torch.uint8).to(torch.float32) / 255
                    # logging.warning(f"\ndelta float int GPU: {torch.sum(torch.abs(old_adversarial_patch - old_adversarial_patch_int_gpu)).item()}")
                    # old_adversarial_patch_int_cpu = (old_adversarial_patch.clone().cpu() * 255).to(torch.uint8).to(torch.float32) / 255
                    # logging.warning(f"delta float int CPU: {torch.sum(torch.abs(old_adversarial_patch.clone().cpu() - old_adversarial_patch_int_cpu)).item()}")

                    if adv_optimizer.lower() == "sign":
                        adversarial_patch.grad.zero_()
                    else:
                        optimizer.zero_grad()

                    avg_loss, avg_sequence_likelihood, avg_geometric_mean_likelihoods, avg_arithmetic_mean_likelihoods = [], [], [], []
                    patch_found, malicious_arg_max = [], []

                    if preemption_requested():
                        return finish("preempted", target_reached=False)
                    if optimizer_step >= max_optimizer_steps:
                        return finish("max_optimizer_steps", target_reached=False)
                    if optimizer_step % checkpoint_interval == 0:
                        save_checkpoint("running")


        return finish("max_optimizer_steps", target_reached=False)

    
    @torch.no_grad()
    def process_images(self, 
                       system_prompt: str, 
                       question: str, 
                       images: Union[str, Image.Image, List[Union[str, Image.Image]]], 
                       temperatures,
                       num_generations,
                       malicious_generation) -> dict:

        assert len(self.models) == 1,  "Only single model is supported for evaluation !!"
        model = self.models[0]

        # --- Prepare input ---
        malicious_target_ids = self.processors[model].tokenizer([malicious_generation], add_special_tokens=False, return_tensors="pt").input_ids
        malicious_target_ids = malicious_target_ids.to(self.vlms[model].device)
        logging.debug("Number of target tokens: %d", malicious_target_ids.shape[-1])
        # shape: [batch_size, malicious_seq_len]

        # traget_image_width, traget_image_height = images[0].size
        # image: (3239, 2159)

        messages = self.prepare_messages(system_prompt=system_prompt, num_images=len(images), question=question)

        llama_text = self.processors[model].apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)

        if not isinstance(images, list):  
            images = [images]

        if self.prepare_text_images_individually: # <--- encodes images and text differently !!
            tokenized_text = self.processors[model].tokenizer(
                text=llama_text,
                return_tensors="pt"
            )
            processed_images = self.processors[model].image_processor(
                images=images,
                return_tensors="pt"
            )
            processed_images.pop("num_tiles")
            content = {k: v for k, v in {**tokenized_text, **processed_images}.items()}
        else:
            content = self.processors[model](
                text=llama_text, 
                images=images,
                return_tensors="pt"
            )
            
        # generate
        content = content.to(self.vlms[model].device)
        

        output_all_temps = {}
        for temp in temperatures:
            output_all_temps[temp] = defaultdict(list)
            generation_count = 1 if temp == 0 else num_generations
            generation_options = {
                "max_new_tokens": int(malicious_target_ids.shape[-1]),
                "do_sample": temp != 0,
                "top_p": None,
                "output_scores": True,
                "return_dict_in_generate": True,
                "num_return_sequences": generation_count,
            }
            if temp != 0:
                generation_options["temperature"] = temp
            else:
                generation_options["temperature"] = None
            output = self.vlms[model].generate(**content, **generation_options)

            prompt_len = content['input_ids'].shape[-1]
            generated_ids = output["sequences"][:, prompt_len:]
            if generated_ids.shape[0] != generation_count:
                raise RuntimeError("Generation returned an unexpected batch size")
            eos_token_id = self.vlms[model].generation_config.eos_token_id
            for sequence_index, output_ids in enumerate(generated_ids):
                output_ids = trim_generated_ids(output_ids, eos_token_id)
                sampled_log_likelihoods = sampled_token_log_likelihoods(
                    output["scores"], output_ids, sequence_index=sequence_index
                )
                sampled_token_likelihoods = torch.exp(sampled_log_likelihoods)
                sampled_arithmetic_mean_likelihood = torch.mean(
                    sampled_token_likelihoods
                ).item()
                sampled_geometric_mean_likelihood = torch.exp(
                    torch.mean(sampled_log_likelihoods)
                ).item()

                output_str = self.processors[model].decode(output_ids)
                sample_matches_target = exact_token_match(
                    output_ids, malicious_target_ids[0]
                )
                output_all_temps[temp]["output_ids"].append(output_ids)
                output_all_temps[temp]["output_str"].append(output_str)
                output_all_temps[temp]["sample_matches_target"].append(sample_matches_target)
                output_all_temps[temp]["sampled_token_likelihoods"].append(sampled_token_likelihoods)
                output_all_temps[temp]["sampled_arithmetic_mean_likelihoods"].append(sampled_arithmetic_mean_likelihood)
                output_all_temps[temp]["sampled_geometric_mean_likelihoods"].append(sampled_geometric_mean_likelihood)
            del output

        return output_all_temps

    @torch.no_grad() 
    def predict(self, 
                system_prompt: str, 
                question: str, 
                images: Union[str, Image.Image, List[Union[str, Image.Image]]], 
                temperatures):

        assert len(self.models) == 1,  "Only single model is supported for evaluation !!"
        model = self.models[0]

        messages = self.prepare_messages(system_prompt=system_prompt, num_images=len(images), question=question)

        llama_text = self.processors[model].apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)

        if not isinstance(images, list):  
            images = [images]

        if self.prepare_text_images_individually: # <--- encodes images and text differently !!
            tokenized_text = self.processors[model].tokenizer(
                text=llama_text,
                return_tensors="pt"
            )
            processed_images = self.processors[model].image_processor(
                images=images,
                return_tensors="pt"
            )
            processed_images.pop("num_tiles")
            content = {k: v for k, v in {**tokenized_text, **processed_images}.items()}
        else:
            content = self.processors[model](
                text=llama_text, 
                images=images,
                return_tensors="pt"
            )
            
        # print(f"Input IDs Shape: {content['input_ids'].shape}")

        content = content.to(self.vlms[model].device)

        output_all_temps = {}
        for temp in temperatures:

            output_all_temps[temp] = defaultdict(list)

            if temp == 0:
                output = self.vlms[model].generate(**content, max_new_tokens=1000, do_sample=False, top_p=None)
            else:
                output = self.vlms[model].generate(**content, max_new_tokens=1000, do_sample=True, temperature=temp, top_p=None)

            prompt_len = content['input_ids'].shape[-1]
            output_ids = output[0, prompt_len:]

            output_str = self.processors[model].decode(output_ids)

            output_all_temps[temp]["output_ids"].append(output_ids)
            output_all_temps[temp]["output_str"].append(output_str)

        return output_all_temps
