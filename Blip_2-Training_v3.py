import os, csv, io, torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoProcessor, AutoModelForVision2Seq, AutoTokenizer,
    CLIPProcessor, CLIPModel
)
from peft import LoraConfig, get_peft_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pycocoevalcap.cider.cider import Cider

# Config
captions_file = "Flicker30k_Captions/captions.txt"
images_dir = "Flicker30k_Dataset"
model_name = "Salesforce/blip2-opt-2.7b"
batch_size, accumulation_steps = 8, 4
resume = False
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
csv_log_path = os.path.join(checkpoint_dir, "training_metrics.csv")
max_to_keep = 3

# Load data
df = pd.read_csv(captions_file, header=None, names=["image_name","idx","caption"], sep=",")
df["image_path"] = df["image_name"].str.strip().apply(lambda x: os.path.join(images_dir, x))
df["caption"] = df["caption"].astype(str).str.strip()
df = df[df["image_path"].apply(os.path.exists) & df["caption"].str.len().gt(0)]
print(f"Loaded {len(df)} valid pairs.")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Model & LoRA
processor = AutoProcessor.from_pretrained(model_name)
processor.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForVision2Seq.from_pretrained(model_name)
lora_cfg = LoraConfig(r=8, target_modules=["q_proj","v_proj"], lora_alpha=32, lora_dropout=0.05)
model = get_peft_model(model, lora_cfg)

class Flickr30KDataset(Dataset):
    def __init__(self, df, proc):
        self.df, self.proc = df.reset_index(drop=True), proc
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.image_path).convert("RGB")
        cap = row.caption
        enc = self.proc(images=img, text=cap,
                        return_tensors="pt", padding="max_length",
                        truncation=True, max_length=32)
        enc = {k: v.squeeze(0) for k,v in enc.items()}
        enc["labels"] = enc["input_ids"]
        return enc

def collate_fn(batch): return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}

train_loader = DataLoader(Flickr30KDataset(train_df, processor),
                          batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Flickr30KDataset(val_df, processor),
                        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Metrics and scorers
loss_vals, cider_vals, clip_vals = [], [], []
cider_scorer = Cider()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

best_cider, start_epoch = 0.0, 0

with open(csv_log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "loss", "cider", "clipscore"])

# Resume from checkpoint
if resume:
    epochs = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pt")])
    if epochs:
        ckpt = torch.load(os.path.join(checkpoint_dir, epochs[-1]))
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_cider = ckpt.get("cider", 0.0)
        print(f"Resumed from epoch {start_epoch}")

# Training loop
for epoch in range(start_epoch, 5):
    model.train(); total_loss=0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss / accumulation_steps
        loss.backward()
        if (step + 1) % accumulation_steps == 0 or step+1 == len(train_loader):
            optimizer.step(); optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    avg_loss = total_loss / len(train_loader)
    loss_vals.append(avg_loss)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # Save and cleanup checkpoints
    ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss, "cider": best_cider}, ckpt_path)
    ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")])
    for old in ckpts[:-max_to_keep]:
        os.remove(os.path.join(checkpoint_dir, old))

    # Validation
    model.eval()
    all_preds, all_refs, all_imgs = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            gen = model.generate(pixel_values=batch["pixel_values"], max_new_tokens=32, num_beams=5)
            all_preds += processor.batch_decode(gen, skip_special_tokens=True)
            all_refs += processor.batch_decode(batch["labels"], skip_special_tokens=True)
            all_imgs += batch["pixel_values"].cpu()

    # Compute CIDEr
    gts = {i: [ref] for i, ref in enumerate(all_refs)}
    res = {i: [pred] for i, pred in enumerate(all_preds)}
    cider, _ = cider_scorer.compute_score(gts, res)
    cider_vals.append(cider)

    # Compute CLIPScore with safe PIL conversion
    clip_scores=[]
    for img_t, cap in zip(all_imgs, all_preds):
        img_t = img_t.clamp(0,1)
        pil = transforms.ToPILImage()(img_t)
        inp = clip_processor(images=pil, text=cap, return_tensors="pt").to(device)
        out = clip_model(**inp)
        clip_scores.append(torch.cosine_similarity(out.image_embeds, out.text_embeds).item())
    avg_clip = sum(clip_scores)/len(clip_scores)
    clip_vals.append(avg_clip)
    print(f"[Epoch {epoch+1}] CIDEr: {cider:.4f} | CLIPScore: {avg_clip:.4f}")

    with open(csv_log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, avg_loss, cider, avg_clip])

    if cider > best_cider:
        best_cider = cider
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss, "cider": best_cider}, best_model_path)
        print(f"✅ New best saved (CIDEr {best_cider:.4f})")

# Plotting
buf = io.BytesIO()
plt.figure(); plt.plot(loss_vals, marker="o"); plt.title("Training Loss"); plt.grid(True)
plt.savefig(buf, format="png"); buf.seek(0); Image.open(buf).save("training_loss_plot.png"); plt.close(); buf.close()
buf = io.BytesIO()
plt.figure(); plt.plot(cider_vals, label="CIDEr", marker="o"); plt.plot(clip_vals, label="CLIPScore", marker="s")
plt.legend(); plt.title("Validation Scores"); plt.grid(True)
plt.savefig(buf, format="png"); buf.seek(0); Image.open(buf).save("validation_metrics_plot.png")
print("✅ Training complete; metrics logged at", csv_log_path)
