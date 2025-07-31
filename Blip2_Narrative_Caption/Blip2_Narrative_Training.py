import os
import json
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    Seq2SeqTrainer
)
import evaluate
import numpy as np
from transformers import AutoProcessor, CLIPModel

#Script that was being worked on while the BLIP-2 script for training and validation was being ran.
#Focus on fine-tuning the BLIP-2 Model to produce first-person captioning 
#Incoproated EgoCap and EgoFormer datasets

# ========== Config ==========
image_root_dir = "static"          # Root directory containing images
json_path = "EgoCap.json"          # JSON file with image-caption pairs
output_dir = "./results"           # Directory to save training results
os.makedirs(output_dir, exist_ok=True)  # Ensure results directory exists

# ========== Dataset Definition ==========
class EgoCapDataset(Dataset):
    """
    Custom dataset for egocentric image captioning.
    Reads image-caption pairs from a JSON file and processes them for BLIP-2.
    """
    def __init__(self, json_file, image_dir, processor, max_length=64):
        # Load annotations from JSON
        with open(json_file, "r") as f:
            self.annotations = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        # Flatten the data into (image, caption) pairs
        self.pairs = [
            (img_name, caption)
            for img_name, meta in self.annotations.items()
            for caption in meta.get("captions", [])
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get image name and caption
        image_name, caption = self.pairs[idx]

        # Locate the image file recursively
        for root, _, files in os.walk(self.image_dir):
            if image_name in files:
                image_path = os.path.join(root, image_name)
                break
        else:
            raise FileNotFoundError(f"{image_name} not found in {self.image_dir}")

        # Open and convert image to RGB
        image = Image.open(image_path).convert("RGB")

        # Add a first-person POV prompt
        prompt = "Describe this image from my point of view."
        input_text = f"{prompt} {caption}"

        # Process the image and text for BLIP-2
        encoding = self.processor(
            images=image, 
            text=input_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        )
        pixel_values = encoding["pixel_values"].squeeze(0)
        input_ids = encoding["input_ids"].squeeze(0)

        # Prepare labels for training (ignore padding)
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": image_name
        }

# ========== Prepare Dataset ==========
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
dataset = EgoCapDataset(json_path, image_root_dir, processor)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# ========== Load Model ==========
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b")

# Freeze the first visual encoder layer to stabilize training
for name, param in model.vision_model.named_parameters():
    if "encoder.layers.0" in name or "embeddings" in name:
        param.requires_grad = False
model.train()

# ========== Metrics Setup ==========
bertscore = evaluate.load("bertscore")  # Load BERTScore metric
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize CIDEr metric scorer
from pycocoevalcap.cider.cider import Cider
cider_scorer = Cider()

def compute_metrics(eval_pred):
    """
    Compute CIDEr, CLIPScore, and BERTScore for evaluation.
    """
    predictions, labels = eval_pred

    # Replace ignored tokens (-100) with PAD for decoding
    predictions = np.where(predictions != -100, predictions, processor.tokenizer.pad_token_id)

    # Decode to text
    pred_texts = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    pred_texts = [p.strip() for p in pred_texts]
    label_texts = [l.strip() for l in label_texts]

    # Compute BERTScore F1
    bert = bertscore.compute(predictions=pred_texts, references=label_texts, lang="en")
    bert_f1 = float(np.mean(bert["f1"]))

    # Compute CLIPScore for each prediction
    clip_scores = []
    for example, caption in zip(val_dataset, pred_texts):
        for root, _, files in os.walk(image_root_dir):
            if example["image_path"] in files:
                image_path = os.path.join(root, example["image_path"])
                break
        else:
            continue

        image_tensor = clip_processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")["pixel_values"]
        inputs = clip_processor(text=[caption], images=image_tensor, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = clip_model(**inputs).logits_per_image[0, 0].item()
        clip_scores.append(logits)

    avg_clip = float(np.mean(clip_scores)) if clip_scores else 0.0

    # Compute CIDEr score
    res = {i: [pred] for i, pred in enumerate(pred_texts)}
    gts = {i: [ref] for i, ref in enumerate(label_texts)}
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return {
        "CIDEr": cider_score,
        "CLIPScore": avg_clip,
        "BERTScore": bert_f1
    }

# ========== Training Configuration ==========
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=50,
    generation_num_beams=5,
    report_to="none",
    fp16=True  # Mixed precision for faster training on GPU
)

# ========== CSV Logger Callback ==========
class CSVLoggerCallback(TrainerCallback):
    """
    Custom callback to log metrics to a CSV file after each evaluation.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "CIDEr", "CLIPScore", "BERTScore", "train_loss"])
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Capture training loss from logs
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Append evaluation metrics to CSV
        if metrics is None:
            return
        with open(self.csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get("epoch", state.epoch),
                metrics.get("eval_CIDEr"),
                metrics.get("eval_CLIPScore"),
                metrics.get("eval_BERTScore"),
                metrics.get("train_loss") or self.last_train_loss
            ])

# ========== Data Collator ==========
class Blip2DataCollator:
    """
    Prepares batches of data for BLIP-2 by stacking pixel values and padding labels.
    """
    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# ========== Custom Trainer ==========
class Blip2Trainer(Seq2SeqTrainer):
    """
    Custom trainer to handle BLIP-2 forward pass.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(pixel_values=inputs["pixel_values"], labels=inputs["labels"])
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Initialize Trainer
trainer = Blip2Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    data_collator=Blip2DataCollator(),
    compute_metrics=compute_metrics
)

# Add CSV logging callback
trainer.add_callback(CSVLoggerCallback(os.path.join(output_dir, "metrics.csv")))

# Train the model
trainer.train()

# Save the fine-tuned model and processor
model.save_pretrained(os.path.join(output_dir, "blip2_egocap_finetuned"))
processor.save_pretrained(os.path.join(output_dir, "blip2_egocap_finetuned"))
