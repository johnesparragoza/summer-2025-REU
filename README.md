# summer-2025-REU TXST
# Image to Narrative with multi modal architecture.

This project implements an image captioning model using (mostly) BLIP-2. It is designed as an assistive tool to help individuals with Autism Spectrum Disorder (ASD) interpret and communicate visual scenes. The model generates descriptive captions for input images and is structured to allow future integration with a language model for narrative generation.

---

## Key Features

- BLIP-2 (Pretrain or Mini version): Lightweight*, pretrained image-to-text model.
- MC-LLaVA : Multimodal, pre-trained, already fine tuned.
- Assistive Design: Supports ASD individuals by translating visual input into readable context.
- Inline Visualization: Displays both the image and its generated caption.
- CPU-Compatible: Optimized to run without a GPU for accessible prototyping.
- Modular Architecture: Built to easily add a language model for expanded functionality.

---

## Project Structure
```bash
.
├── blip2.ipynb                  # Jupyter Notebook with BLIP-2 captioning pipeline
├── images/                      # Folder containing input images
├── requirements.txt             # Python dependencies
├── README.md                    # Project description and usage
├──
