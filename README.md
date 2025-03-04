# Vision GRPO

# Training Vision Language Models with GRPO for Visual Grounding

Based on the recent advances in RL for reasoning enhance, we'll explore how to fine-tune Vision Language Models (VLMs) using ***Group Relative Policy Optimization*** (***GRPO***). We'll walk through a complete training pipeline, from dataset preparation to evaluating results.

## 1. Modified GRPO for Vision Language Models

### Adapting GRPO for Vision Language Models

Based on the great tutorial [mini-R1](https://www.philschmid.de/mini-deepseek-r1) tutorial, we provided the modified version of the approach for training vision language models using the same reasoning approach. To adapt it for Vision Language Models, we need to:

1. **Handle Multimodal Inputs**: Process both images and text in the same framework
2. **Custom Reward Functions**: Create vision-specific rewards that evaluate how well the model identifies regions in images
3. **Specialized Architecture**: Use a vision-language model architecture (like Qwen2.5-VL) that can process both modalities

Due to the fact that each Vision-Language model follows its own architecture, we are not able to use unified abstraction such as AutoModelforCausalLM for language models, so, our tutorial covers two common multimodal architectures for Qwen-VL (2 and 2.5).

The modified `Qwen2VLGRPOTrainer` enables:

- Processing of image-text pairs
- Evaluation of visual grounding capabilities
- Optimization of both text generation and region identification

```python
# Example of the modified GRPO trainer integration
trainer = Qwen2VLGRPOTrainer(
    model=model,
    reward_funcs=[grounding_reward, format_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_args)
)

```

## 2. The Visual Grounding Task for CoT Reasoning

### What is Visual Grounding?

Basically, vision grounding task is defined as a task to provide bounding boxes for the object defined in the input request. We look into the visual grounding as a suplementary task to help the model to provide correct answer on some complex question. This approach was investigated in [Visual CoT](https://arxiv.org/abs/2403.16999), where authors proposed to use visual grounding task to zoom into specific part of the image, where the answer is kept. In our tutorial, we use subsample of the textVQA dataset to show, whether we can teach the model to zoom in to the relevant parts of the image via RL.

### Task Formulation

The task is structured as follows:

1. The model receives an image and a text query about a specific visual element
2. The model must:
    - Reason through the visual content (in `<think>...</think>` tags)
    - Output precise bounding box coordinates for the relevant region (in `<answer>[x1, y1, x2, y2]</answer>` format)

### Example Query

```
Image: [Image of a living room]
Query: Where is the red vase in this image? Show your reasoning in <think> thinking process </think> tags. Return bounding box in <answer> [x1, y1, x2, y2] </answer> tags.
```

### Expected Output

```
Let me analyze this image.
<think>
I can see a living room with various furniture. Looking for a red vase...
I can see a red vase on the coffee table in the center of the image.
It appears to be located approximately at the coordinates [220, 150, 260, 210].
</think>
<answer>{"bbox": [220, 150, 260, 210]}</answer>

```

## 3. Dataset Preparation

### Dataset Structure

For this tutorial, we use a vision chain-of-thought dataset specifically designed for visual grounding tasks:

```python
import json
import math
from PIL import Image
import os

def process_jsonl_data(jsonl_file, train_path, output_file=None, max_size=512, maintain_aspect_ratio=True):
    """
    Process a JSONL file containing image metadata, resize images, and rescale bounding boxes.
    
    Parameters:
    -----------
    jsonl_file: str
        Path to the JSONL file
    train_path: str
        Path to the directory containing training images
    output_file: str, optional
        Path to save the processed dataset (if None, just returns the data)
    max_size: int, default=512
        Maximum dimension for resized images
    maintain_aspect_ratio: bool, default=True
        Whether to maintain aspect ratio when resizing
        
    Returns:
    --------
    list: Processed dataset
    """
    dataset = []
    
    # Count for statistics
    total_entries = 0
    skipped_entries = 0
    processed_entries = 0
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                # Skip any empty lines if present
                continue
                
            total_entries += 1
            
            try:
                data = json.loads(line)
                
                # Skip entries with multiple bounding boxes
                if len(data['bboxs']) > 1:
                    skipped_entries += 1
                    continue
                
                # Ensure image path is complete
                if not data['image'].startswith(train_path):
                    data['image'] = os.path.join(train_path, data['image'])
                
                # Check if image exists
                if not os.path.exists(data['image']):
                    print(f"Warning: Image not found at {data['image']}")
                    skipped_entries += 1
                    continue
                
                # Open and get dimensions of the image
                try:
                    image = Image.open(data['image'])
                    original_width, original_height = image.size
                except Exception as e:
                    print(f"Error opening image {data['image']}: {e}")
                    skipped_entries += 1
                    continue
                
                # Determine new dimensions
                if maintain_aspect_ratio:
                    if original_width > max_size or original_height > max_size:
                        # Calculate new dimensions maintaining aspect ratio
                        if original_width > original_height:
                            new_width = max_size
                            new_height = int(original_height * (max_size / original_width))
                        else:
                            new_height = max_size
                            new_width = int(original_width * (max_size / original_height))
                    else:
                        # Image is within acceptable dimensions, no resize needed
                        new_width, new_height = original_width, original_height
                else:
                    # Fixed size without maintaining aspect ratio
                    new_width, new_height = max_size, max_size
                
                # Only rescale bounding boxes if dimensions changed
                if new_width != original_width or new_height != original_height:
                    # Calculate the scaling factors
                    scale_x = new_width / original_width
                    scale_y = new_height / original_height
                    
                    # Rescale all bounding boxes
                    new_bboxs = []
                    for original_bbox in data['bboxs']:
                        # Adjust the bounding box coordinates
                        new_bbox = [
                            math.ceil(original_bbox[0] * scale_x),
                            math.ceil(original_bbox[1] * scale_y),
                            math.ceil(original_bbox[2] * scale_x),
                            math.ceil(original_bbox[3] * scale_y)
                        ]
                        new_bboxs.append(new_bbox)
                    
                    # Update bounding boxes in the data
                    data['bboxs'] = new_bboxs.copy()
                
                # Store the new dimensions in the data
                data['width'] = new_width
                data['height'] = new_height
                
                # Append processed data to the dataset
                dataset.append(data)
                processed_entries += 1
                
                # Print progress every 1000 entries
                if processed_entries % 1000 == 0:
                    print(f"Processed {processed_entries} entries...")
                
            except Exception as e:
                print(f"Error processing line: {e}")
                skipped_entries += 1
    
    # Print statistics
    print(f"Total entries: {total_entries}")
    print(f"Processed entries: {processed_entries}")
    print(f"Skipped entries: {skipped_entries}")
    
    # Save processed dataset if output file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        print(f"Saved processed dataset to {output_file}")
    
    return dataset

# Example usage:
if __name__ == "__main__":
    TRAIN_PATH = "./train_images/"
    JSONL_FILE = "./metadata/textvqa_cot_train.jsonl"
    OUTPUT_FILE = "processed_textvqa_train.jsonl"
    
    # Process the JSONL file
    processed_data = process_jsonl_data(
        jsonl_file=JSONL_FILE,
        train_path=TRAIN_PATH,
        output_file=OUTPUT_FILE,
        max_size=512,
        maintain_aspect_ratio=True
    )
    
    print(f"Processed dataset contains {len(processed_data)} entries")
    
    # Show a sample entry if available
    if processed_data:
        sample = processed_data[0]
        print("\nSample entry:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Image: {sample['image']}")
        print(f"Dimensions: {sample['width']}x{sample['height']}")
        print(f"Bounding boxes: {sample['bboxs']}")

```

### Generating Prompts for Training

We format each example into a chat template for Qwen2.5-VL, using a system message that specifies the visual grounding task:

```python
system_message = "You are a Vision Language Model specialized in visual grounding. Provide bounding box in <answer> [x1, y1, x2, y2] </answer>."

def generate_r1_prompt(sample):
    prefix = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {
                    "type": "text",
                    "text": (
                        sample["question"] + " Show your reasoning in <think> thinking process </think> tags. "
                        "Return bounding box in <answer> [x1, y1, x2, y2] </answer> tags."
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me analyze this image.\n<think>"}],
        },
    ]
    encoded_prompt = processor.apply_chat_template(prefix, tokenize=False, continue_final_message=True)
    return {"prompt": encoded_prompt, "target": sample["bboxs"]}

# Apply prompt generation to dataset
dataset = dataset.map(generate_r1_prompt)

# Create train/test split
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

```

## 4. Launching Training

### Setting Up Reward Functions

A key component of GRPO is the definition of reward functions. For visual grounding, we define multiple reward functions to evaluate different aspects of the model's output:

```python
def grounding_reward(completions, target, **kwargs):
    """Reward function that checks bounding boxes."""
    rewards = []
    for completion, gt_bbox in zip(completions, target):
        try:
            bbox_match = re.search(r"<answer>\[(.*?)\]</answer>", completion)
            if bbox_match:
                pred_bbox = [float(x.strip()) for x in bbox_match.group(1).split(",")]
                gt_bbox = [float(x) for x in gt_bbox[0].strip("[]").split(",")]

                # Check IoU between predicted and ground truth bounding boxes
                reward = 1.0 if relaxed_bbox_iou(pred_bbox, gt_bbox) else 0.0
            else:
                reward = 0.0
        except Exception:
            reward = 0.0
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """Check that completions follow the required format."""
    completions = ["<think>" + c for c in completions]
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.fullmatch(pattern, c, re.DOTALL) for c in completions]
    return [1.0 if m else 0.0 for m in matches]

# Select reward functions for training
chosen_reward_funcs = [grounding_reward, format_reward]

```

### Training Configuration

We configure the training process with appropriate hyperparameters:

```python
# Training arguments example
training_args = GRPOConfig(
    output_dir="./qwen_vl_grpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    report_to="wandb",
    logging_first_step=True
)

```

### Initializing Model and Trainer

We load the Qwen2.5-VL model and set up the GRPO trainer:

```python
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

# Load model and processor
processor = Qwen2_5_VLProcessor.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=True
)

# Initialize GRPO trainer
trainer = Qwen2VLGRPOTrainer(
    model=model,
    reward_funcs=chosen_reward_funcs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_args)
)

# Start training
train_result = trainer.train()

```

### Saving and Logging

After training completes, we save the model and metrics:

```python
# Save metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Save model
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

# Optional: Push to Hugging Face Hub
if training_args.push_to_hub:
    trainer.push_to_hub()

```

## 5. Training Metrics

<img width="756" alt="image" src="https://github.com/user-attachments/assets/a8d03dd6-7290-49c6-8700-46014b3e3235" />

## 6. Example Results

Let's look at some examples of the model's performance after training:

### Example 1: Successful Grounding

<img width="414" alt="image" src="https://github.com/user-attachments/assets/b46d9923-921c-449b-8bcd-3df1246d5576" />

**Query:**

```
What is the comment? Show your reasoning in <think> thinking process </think> tags. Return bounding box in <answer> [x1, y1, x2, y2] </answer> tags.

```

**Model Output:**


```
Let me analyze this image.
<think>
The comment on the book is located near the bottom of the image, just above the comment input field.
</think>
<answer>{"bbox": [508, 467, 593, 487]}</answer>

```

Qwen2.5 VL initially performs well on grounding tasks; however, the results vary across different examples.

<img width="414" alt="DUNE" src="https://github.com/user-attachments/assets/e1d66ce3-23c9-49be-a420-3337b307b2aa" />

## Conclusion

In this tutorial, we've walked through the complete process of training a Vision Language Model for visual grounding using GRPO:

1. We adapted GRPO for vision-language tasks by implementing custom reward functions for bounding box evaluation
2. Prepared a specialized dataset for visual grounding with formatted prompts
3. Configured and launched training with the modified `Qwen2VLGRPOTrainer`
4. Examined examples showing the model's ability to perform visual grounding tasks

This approach demonstrates how reinforcement learning techniques can be applied to multimodal models, helping them learn to connect textual and visual information more effectively. While the example is not for real-life applications, and smaller models can benefit more from SFT-reasoning, this is a good starting point.

### Next Steps

- Experiment with different reward functions to further improve performance
- Explore more complex visual grounding tasks (e.g., multiple object identification)
- Combine with other vision-language tasks like visual question answering or image captioning

### Resources

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Qwen2.5-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
- [Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning](https://arxiv.org/abs/2403.16999)
- [Mini-R1: Reproduce Deepseek R1 „aha moment“ a RL tutorial](https://www.philschmid.de/mini-deepseek-r1)
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1/tree/main)
- [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
