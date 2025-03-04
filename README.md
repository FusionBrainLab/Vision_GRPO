# Vision_GRPO

# Training Vision Language Models with GRPO for Visual Grounding

Based on the recent advances in RL for reasoning enhance, we'll explore how to fine-tune Vision Language Models (VLMs) using ***Group Relative Policy Optimization*** (***GRPO***). We'll walk through a complete training pipeline, from dataset preparation to evaluating results.

## 1. Modified GRPO for Vision Language Models

### Adapting GRPO for Vision Language Models

Based on the great tutorial by Phill Schmidt \cite{}, we provided the modified version of the approach for training vision language models using the same reasoning approach. To adapt it for Vision Language Models, we need to:

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

Basically, vision grounding task is defined as a task to provide bounding boxes for the object defined in the input request. We look into the visual grounding as a suplementary task to help the model to provide correct answer on some complex question. This approach was investigated in \cite{vision chain of thoughts}, where authors proposed to use visual grounding task to zoom into specific part of the image, where the answer is kept. In our tutorial, we use subsample of the textVQA dataset to show, whether we can teach the model to zoom in to the relevant parts of the image via RL.

![image.png](attachment:18c6cd9c-20aa-40d2-8f37-73b9641cd828:image.png)

### Task Formulation

In our implementation, the task is structured as follows:

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
# Load the dataset
dataset = datasets.load_from_disk("/path/to/vision_cot")

# Define features for dataset
my_features = Features({
    "image": Image(decode=True),
    "question": Value("string"),
    "answer": Value("string"),
    "bboxs": Sequence(Sequence(Value("float")))
})

# Process dataset to keep relevant fields
def keep_fields(example):
    return {
        "image": example["image"],
        "question": example["question"],
        "answer": example["answer"],
        "bboxs": example["bboxs"]
    }

dataset = dataset.map(
    keep_fields,
    features=my_features,
    remove_columns=["width", "height", "dataset", "split"]
)

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
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    fp16=True,
    report_to="wandb",
    logging_first_step=True
)

```

### Initializing Model and Trainer

We load the Qwen2.5-VL model and set up the GRPO trainer:

```python
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

# Load model and processor
processor = Qwen2VLProcessor.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=True
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
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

During training, we monitor several key metrics to evaluate the model's performance:

### Loss Tracking

![image.png](attachment:353a9fef-171f-4562-9bf7-ecbb08ad2862:image.png)

The training process shows how the loss decreases over time, indicating the model is learning to:

- Generate proper reasoning in `<think>` tags
- Accurately predict bounding boxes in `<answer>` tags

## 6. Example Results

Let's look at some examples of the model's performance after training:

### Example 1: Successful Grounding

![image.png](attachment:9226dbdd-8c25-4224-b561-c1d4204ba806:image.png)

![image.png](attachment:e7fefc98-945d-4537-8dbd-728972d40daf:image.png)

**Query:**

```
Who is the author of the book? Show your reasoning in <think> thinking process </think> tags. Return bounding box in <answer> [x1, y1, x2, y2] </answer> tags.

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

### Example 2: Challenging Case

**Query:**

```
Where is the coffee mug in this kitchen scene? Show your reasoning in <think> thinking process </think> tags. Return bounding box in <answer> [x1, y1, x2, y2] </answer> tags.

```

**Model Output:**

```
Let me analyze this image.
<think>
This is a kitchen scene with various items on the counter.
I need to locate a coffee mug.
I can see what appears to be a white coffee mug on the counter near the microwave.
There's also what might be another mug partially visible behind some other objects.
The white mug is more clearly visible, so I'll focus on that one.
Based on the image dimensions, I estimate the bounding box coordinates to be around [180, 220, 210, 250].
</think>
<answer>[180, 220, 210, 250]</answer>

```

## Conclusion

In this tutorial, we've walked through the complete process of training a Vision Language Model for visual grounding using GRPO:

1. We adapted GRPO for vision-language tasks by implementing custom reward functions for bounding box evaluation
2. We prepared a specialized dataset for visual grounding with formatted prompts
3. We configured and launched training with the modified `Qwen2VLGRPOTrainer`
4. We tracked key metrics to monitor training progress
5. We examined examples showing the model's ability to perform visual grounding tasks

This approach demonstrates how reinforcement learning techniques can be applied to multimodal models, helping them learn to connect textual and visual information more effectively. The trained model can accurately identify regions in images based on textual descriptions, providing explanations for its reasoning process.

### Next Steps

- Experiment with different reward functions to further improve performance
- Explore more complex visual grounding tasks (e.g., multiple object identification)
- Combine with other vision-language tasks like visual question answering or image captioning

### Resources

- GRPO Paper
- Qwen2.5-VL Model Documentation
- Visual Grounding Benchmark Datasets