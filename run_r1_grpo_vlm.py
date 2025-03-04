import logging
import os
from dataclasses import dataclass
from datetime import datetime
import random
import re
import torch
import yaml

from transformers.trainer_utils import get_last_checkpoint

from grpo_config import GRPOConfig

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

import datasets
from datasets import load_dataset
from torch.utils.data import Dataset

import sys
import math
from typing import Optional, Tuple

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
import torch
from typing import Tuple


# FlashAttention fix from https://github.com/om-ai-lab/VLM-R1/blob/main/src/open-r1-multimodal/src/open_r1/grpo_rec.py
def custom_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
    else:
        cos, sin = position_embeddings
        # Add this
        cos = cos.to(torch.float)
        sin = sin.to(torch.float)
    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output


Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from grpo_qwen2vl import Qwen2VLGRPOTrainer
from PIL import Image

try:
    from math_verify import parse, verify
except ImportError:

    def parse(x):
        return x

    def verify(x, y):
        return float(x == y)


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Optional: wandb environment
os.environ["WANDB_PROJECT"] = "grpo-qwen-2-2B"
os.environ["WANDB_ENTITY"] = ""
os.environ["WANDB_API_KEY"] = "3d726fd76bb1ed0c15a7004731707d54572acef0"
os.environ["WANDB_NAME"] = "grpo-qwen-2-2B-v-8gpus-one_reward-synthetic-data-check"


@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    max_pixels: int = 12845056
    min_pixels: int = 3136
    image_root: str = ""
    config_path: str = "config.yaml"
    json_data_path: str = ""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def relaxed_bbox_iou(pred_bbox, gt_bbox, threshold=0.5):
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = pred_area + gt_area - intersection
    iou = intersection / union if union > 0 else 0

    return iou >= threshold


def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> bool:
    def extract_first_number(text: str):
        match = re.search(r"[-]?\d[\d,]*(\.\d+)?%?", text)
        return match.group(0) if match else text

    def to_float(text: str):
        text = text.strip().lower()
        text = text.replace(",", "")
        if text.endswith("%"):
            try:
                val = float(text.rstrip("%"))
                return val / 100.0
            except ValueError:
                return None
        else:
            try:
                return float(text)
            except ValueError:
                return None

    pred_num_str = extract_first_number(prediction)
    tgt_num_str = extract_first_number(target)
    pred_float = to_float(pred_num_str)
    tgt_float = to_float(tgt_num_str)
    if pred_float is not None and tgt_float is not None:
        if abs(tgt_float) < 1e-12:
            return abs(pred_float - tgt_float) < 1e-12
        relative_change = abs(pred_float - tgt_float) / abs(tgt_float)
        return relative_change <= max_relative_change
    return prediction.strip().lower() == target.strip().lower()


def grounding_reward(completions, target, **kwargs):
    """
    Reward function that checks bounding boxes. We keep your original logic.
    """
    completions = [c[0]["content"] for c in completions]
    rewards = []
    for completion, gt_bbox_list in zip(completions, target):
        try:
            bbox_match = re.search(r"<answer>.*?\[(.*?)\].*?</answer>", completion, re.DOTALL)
            if bbox_match:
                pred_bbox = [float(x.strip()) for x in bbox_match.group(1).split(",")]
                gt_bbox = [float(x) for x in gt_bbox_list[0]]  # given your data format
                reward = 1.0 if relaxed_bbox_iou(pred_bbox, gt_bbox) else 0.0
            else:
                reward = 0.0
        except Exception:
            reward = 0.0

        rewards.append(reward)
        # Log a small fraction
        if random.random() < 0.1:
            os.makedirs("completion_samples", exist_ok=True)
            log_file = os.path.join("completion_samples", "grounding_samples.txt")
            with open(log_file, "a") as f:
                f.write(f"------------- Grounding reward: {reward} -------------\n")
                f.write(f"Content: {completion}\n")
                f.write(f"GT bbox: {gt_bbox_list}\n")

    return rewards


def format_reward(completions, **kwargs):
    """A simpler check for <think>... and <answer>... (not checking correctness)."""
    completions = [c[0]["content"] for c in completions]
    completions = ["<think>" + c if c.startswith("<think>") else c for c in completions]
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.fullmatch(pattern, c, re.DOTALL) for c in completions]
    return [1.0 if m else 0.0 for m in matches]


def get_checkpoint(training_args):
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


class CustomDataset:
    def __init__(self, list_data_dict=None, script_args=None, processor=None, json_data_path=None):
        super(CustomDataset, self).__init__()
        self.script_args = script_args
        self.processor = processor
        self.SYSTEM_PROMPT = (
            "You are a Vision Language Model specialized in visual grounding in <answer>[x1, y1, x2, y2]</answer>."
        )
        self.QUESTION_TEMPLATE = "Provide bounding box for the region of the image relevant to the asked question: {Question}. First output the thinking process in <think> </think> tags and then output the final answer in <answer>[x1, y1, x2, y2]</answer> tags."

        # Load data from JSON file if path is provided
        if json_data_path and os.path.exists(json_data_path):
            import json

            with open(json_data_path, "r") as f:
                data = json.load(f)
            self.list_data_dict = data
        else:
            self.list_data_dict = list_data_dict or []

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                ],
            }

        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": self.QUESTION_TEMPLATE.format(Question=example["question"])
                                + '<answer> {"bbox": [x1, y1, x2, y2]} </answer> tags, '
                                + "where x1, y1, x2, y2 are integers representing the coordinates.",
                            },
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if "image" in example:
            image_path = os.path.join(image_root, example["image"])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example["image"])
            image = Image.open(image_path).convert("RGB")

            # Resize image if needed to meet min/max size requirements
            w, h = image.size

            if w < 28 or h < 28:
                # Calculate new dimensions maintaining aspect ratio for small images
                if w < h:
                    new_w = 28
                    new_h = int(h * (28 / w))
                else:
                    new_h = 28
                    new_w = int(w * (28 / h))
                image = image.resize((new_w, new_h), Image.LANCZOS)
            elif w > 512 or h > 512:
                # Calculate new dimensions maintaining aspect ratio for large images
                if w > h:
                    new_w = 512
                    new_h = int(h * (512 / w))
                else:
                    new_h = 512
                    new_w = int(w * (512 / h))
                image = image.resize((new_w, new_h), Image.LANCZOS)
            else:
                # Image is within acceptable dimensions, no resize needed
                new_w, new_h = w, h
        else:
            image = None
        # print("Image size", image.size)
        return {
            "image": image,
            "question": example["question"],
            "target": example["bboxs"],
            "prompt": (
                make_conversation_image(example)["prompt"]
                if "image" in example
                else make_conversation(example)["prompt"]
            ),
        }


def grpo_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    model_name = model_args.model_name_or_path
    tokenizer_path = script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_name

    # Load the appropriate model and processor based on model name
    if "Qwen2.5-VL" in model_name:
        processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )
    else:  # Default to Qwen2-VL
        processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )

    # Create CustomDataset instances
    train_dataset = CustomDataset(
        script_args=script_args,
        processor=processor,
        json_data_path=script_args.json_data_path,
    )
    print(f"Created datasets with {len(train_dataset)} training examples")
    print(f"Sample example: {train_dataset[0]}")

    # Choose your reward functions
    chosen_reward_funcs = [grounding_reward, format_reward]

    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=chosen_reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Resuming from checkpoint at {last_checkpoint}.")

    logger.info("*** Starting training ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Processor saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "tutorial", "philschmid"]})

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Done ***")


def main():
    from trl import TrlParser

    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Load config from config file if not set directly via command line
    if not script_args.json_data_path and script_args.config_path and os.path.exists(script_args.config_path):
        with open(script_args.config_path, "r") as f:
            config = yaml.safe_load(f)
            if config and "json_data_path" in config:
                script_args.json_data_path = config.get("json_data_path", "")

    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
