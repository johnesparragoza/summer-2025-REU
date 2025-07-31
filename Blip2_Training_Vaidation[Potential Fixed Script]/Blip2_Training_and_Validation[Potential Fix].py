# ===== Import Required Libraries =====
import os, torch, csv, io, shutil                # Core Python & PyTorch utilities
import pandas as pd                             # Data handling
from PIL import Image                           # Image loading and processing
from tqdm import tqdm                            # Progress bars for loops
from torch.utils.data import Dataset, DataLoader # Dataset & batching utilities
from torchvision import transforms               # Image transformations
from transformers import (                       # Hugging Face Transformers
    AutoProcessor, AutoModelForVision2Seq, AutoTokenizer,
    CLIPProcessor, CLIPModel
)
from peft import LoraConfig, get_peft_model      # LoRA for parameter-efficient fine-tuning
from sklearn.model_selection import train_test_split  # Train/val split
from pycocoevalcap.cider.cider import Cider      # CIDEr evaluation metric
from matplotlib import pyplot as plt             # Plotting for training curves
from bert_score import score as bertscore        # BERTScore for caption evaluation
from pathlib import Path                         # File path handling

# ===== Configuration =====
captions_file = "Flicker30k_Captions/captions.txt"  # Captions file path
images_dir = "Flicker30k_Dataset"                   # Image dataset folder
model_name = "Salesforce/blip2-opt-6.7b"            # Base BLIP-2 model
batch_size, accumulation_steps = 8, 4               # Batch & gradient accumulation
learning_rate = 1e-4                                # Training learning rate
num_epochs = 3                                      # Training epochs
max_token_len = 800                                 # Max token length for captions
checkpoint_dir = "checkpoints_run2"                 # Checkpoint directory
csv_log_path = os.path.join(checkpoint_dir, "training_metrics_run2.csv")  # CSV log path
os.makedirs(checkpoint_dir, exist_ok=True)          # Ensure checkpoint folder exists

#Have not evalauted the script effectivness due to time constraints
#Added a frozen layer and ensure that the 5 captions are partnered with each image.
#BERTScore was added for future data analysis discussions.

# ===== Load Captions into DataFrame =====
df = pd.read_csv(captions_file, header=None, names=["image_name", "idx", "caption"], sep=",")
df["image_path"] = df["image_name"].str.strip().apply(lambda x: os.path.join(images_dir, x))
df["caption"] = df["caption"].astype(str).str.strip()
df = df[df["image_path"].apply(os.path.exists) & df["caption"].str.len().gt(0)]  # Filter missing or empty captions

# ===== Split Dataset (80% Train / 20% Validation) =====
unique_images = df["image_name"].unique()                       # Unique image names
train_imgs, val_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
train_df = df[df["image_name"].isin(train_imgs)].reset_index(drop=True)
val_df = df[df["image_name"].isin(val_imgs)].reset_index(drop=True)

# ===== Initialize Model & Tokenizer =====
processor = AutoProcessor.from_pretrained(model_name)           # Image+Text processor
processor.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForVision2Seq.from_pretrained(model_name)      # BLIP-2 Vision-to-Seq
lora_cfg = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], lora_alpha=32, lora_dropout=0.05)
model = get_peft_model(model, lora_cfg)                        # Apply LoRA adaptation

# 🔒 Freeze the first layer of the vision encoder to stabilize training
for name, param in model.vision_model.encoder.layers[0].named_parameters():
    param.requires_grad = False
    print(f"🔒 Frozen: vision_model.encoder.layers[0].{name}")

# ===== Custom Dataset Class =====
class Flickr30KDataset(Dataset):
    def __init__(self, df, proc):
        self.df, self.proc = df, proc
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, i):
        # Load image and caption
        row = self.df.iloc[i]
        img = Image.open(row.image_path).convert("RGB")
        cap = row.caption
        
        # Process image+caption for model input
        enc = self.proc(images=img, text=cap,
                        return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_token_len)
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # Remove batch dimension
        enc["labels"] = enc["input_ids"]                 # Labels for LM loss
        return enc

# Collate function to stack batches properly
def collate_fn(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}

# Create DataLoaders for training and validation
train_loader = DataLoader(Flickr30KDataset(train_df, processor), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Flickr30KDataset(val_df, processor), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ===== Training Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW optimizer
cider_scorer = Cider()                                               # Initialize CIDEr
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize metric storage
loss_vals, cider_vals, clip_vals, bert_vals = [], [], [], []
best_cider, start_epoch = 0.0, 0

# Create CSV file header for logging
with open(csv_log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "loss", "cider", "clipscore", "bertscore"])

# ===== Training Loop =====
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    # ----- Training Phase -----
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss / accumulation_steps  # Gradient accumulation
        loss.backward()
        
        # Update model after accumulated steps
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    avg_loss = total_loss / len(train_loader)
    loss_vals.append(avg_loss)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # ----- Validation Phase -----
    model.eval()
    all_preds, all_refs, all_imgs = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            gen = model.generate(pixel_values=batch["pixel_values"], max_new_tokens=32, num_beams=5)
            all_preds += processor.batch_decode(gen, skip_special_tokens=True)
            all_refs += processor.batch_decode(batch["labels"], skip_special_tokens=True)
            all_imgs += batch["pixel_values"].cpu()

    # Filter out blank predictions for metrics
    filtered_preds, filtered_refs = [], []
    for pred, ref in zip(all_preds, all_refs):
        if pred.strip() and ref.strip():
            filtered_preds.append(pred.strip())
            filtered_refs.append(ref.strip())

    num_blank_preds = len(all_preds) - len(filtered_preds)
    if num_blank_preds > 0:
        print(f"⚠️ Skipped {num_blank_preds} blank predictions during scoring")

    # ----- Compute CIDEr Score -----
    if filtered_preds:
        gts = {i: [ref] for i, ref in enumerate(filtered_refs)}
        res = {i: [pred] for i, pred in enumerate(filtered_preds)}
        cider, _ = cider_scorer.compute_score(gts, res)
    else:
        cider = 0.0
        print(f"⚠️ [Epoch {epoch+1}] All predictions/references were empty. CIDEr set to 0.")
    cider_vals.append(cider)

    # ----- Compute CLIPScore -----
    clip_scores = []
    valid_clip_pairs = 0
    for img_t, cap in zip(all_imgs, all_preds):
        if cap.strip():
            img_t = img_t.clamp(0, 1)
            pil = transforms.ToPILImage()(img_t)
            try:
                inp = clip_processor(images=pil, text=cap.strip(), return_tensors="pt").to(device)
                out = clip_model(**inp)
                sim = torch.cosine_similarity(out.image_embeds, out.text_embeds).item()
                clip_scores.append(sim)
                valid_clip_pairs += 1
            except Exception as e:
                print(f"⚠️ CLIPScore skipped for one sample due to: {e}")
    avg_clip = sum(clip_scores) / valid_clip_pairs if valid_clip_pairs > 0 else 0.0
    clip_vals.append(avg_clip)

    # ----- Compute BERTScore (F1) -----
    P, R, F1 = bertscore(all_preds, all_refs, lang="en", rescale_with_baseline=True)
    avg_bert = F1.mean().item()
    bert_vals.append(avg_bert)

    print(f"[Epoch {epoch+1}] CIDEr: {cider:.4f} | CLIPScore: {avg_clip:.4f} | BERTScore: {avg_bert:.4f}")

    # Log metrics to CSV
    with open(csv_log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, avg_loss, cider, avg_clip, avg_bert])

    # Save best model based on CIDEr
    if cider > best_cider:
        best_cider = cider
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "cider": best_cider
        }, os.path.join(checkpoint_dir, "best_model_run2.ckpt"))
        print(f"✅ New best saved (CIDEr {best_cider:.4f})")

# ===== Plotting and Saving Metrics =====
# Training Loss Curve
buf = io.BytesIO()
plt.figure(); plt.plot(loss_vals, marker="o"); plt.title("Training Loss"); plt.grid(True)
plt.savefig(buf, format="png"); buf.seek(0); Image.open(buf).save("training_loss_plot_run2.png"); plt.close(); buf.close()

# Validation Metrics Curve (CIDEr & CLIPScore)
buf = io.BytesIO()
plt.figure(); plt.plot(cider_vals, label="CIDEr", marker="o"); plt.plot(clip_vals, label="CLIPScore", marker="s")
plt.legend(); plt.title("Validation Scores"); plt.grid(True)
plt.savefig(buf, format="png"); buf.seek(0); Image.open(buf).save("validation_metrics_plot_run2.png")

print("✅ Training complete; metrics logged at", csv_log_path)
