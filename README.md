# summer-2025-REU
# Visual Understanding with BLIP-2 for ASD Support

This project implements an image captioning model using the lightweight version of BLIP-2. It is designed as an assistive tool to help individuals with Autism Spectrum Disorder (ASD) interpret and communicate visual scenes. The model generates descriptive captions for input images and is structured to allow future integration with a language model for narrative generation.

---

## Key Features

- BLIP-2 (Pretrain or Mini version): Lightweight, pretrained image-to-text model.
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
└── README.md                    # Project description and usage
