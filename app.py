import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import re

# Fading function
def get_faded_prompt(words, fade_level):
    # Return narrative with the last 'fade_level' words replaced with blanks
    if fade_level == 0:
        return " ".join(words)
    faded_words = [
        word if i < len(words) - fade_level else "___"
        for i, word in enumerate(words)
    ]
    return " ".join(faded_words)

# APP UI
st.title("Image to Narrative")
st.markdown("Upload an image and get a short narrative with interactive fading.")

# Tooltips & Help
with st.expander("What is this app?"):
    st.write("""
    This app helps you describe images in a simple way.
    You can upload a photo of your choosing, get a short narrative,
    and practice filling in the blanks as the words are faded out!
    """)

# User upload image
uploaded_file = st.file_uploader(
    "Choose an image file (jpg, jpeg, png)...",
    type = ["jpg", "jpeg", "png"],
    help = "Upload a clear photo (jpg, jpeg, png). Max size: 5MB"
)

# loading the model & processor (MC_LLaVA) as of now, witch cache
@st.cache_resource
def load_model():
    model_id = "visheratin/MC-LLaVA-3b"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model
processor, model = load_model()

# Open & display uploaded image
if uploaded_file is not None: 
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    #Caption input (tooltip)
    caption = st.text_input(
        "Optional: add your own description for this image",
        "",
        help="Add a short description, or leave blank for automatic caption"
    )
    if not caption:
        caption = " " #avoid empty caption in prompt
    
    # Building/editing prompt
    prompt = (
        "<|im_start|>user\n"
        f"<image>\nDescribe this image in a single sentence, short and simple, no more than 10 words.\nCaption: {caption}\nNarrative:\n"
        "<|im_end|>\n"
        "<|im_start|>assisstant\n"
    )
    #Prepare inputs for model
    inputs = processor(prompt, [image], model, return_tensors="pt")
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    # Generate narrative
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the narrative part
    narrative_parts = re.split(r"<\|im_start\|>assistant", generated_text)
    narrative = narrative_parts[-1] if len(narrative_parts) > 1 else generated_text
    narrative = narrative.replace("<|im_end|>", "").strip()
    words = narrative.split()


    # Fading level 
    if "fade_level" not in st.session_state:
        st.session_state.fade_level = 0
    max_fade = len(words)
    #Buttons to control fading
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("Fade Less", help="Decrease fading level (remove more blanks)"):
            if st.session_state.fade_level > 0:
                st.session_state.fade_level -= 1

    with col3: 
        if st.button("Fade More", help="Increase fading level (add more blanks)"):
            if st.session_state.fade_level < max_fade:
                st.session_state.fade_level += 1
    # Display faded narrative
    faded_prompt = get_faded_prompt(words, st.session_state.fade_level)
    st.markdown(f"**Your Faded Prompt:** {faded_prompt}")        
    st.success(faded_prompt)

    #optional: Show full narrative
    with st.expander("Show Full Narrative"):
        st.info(narrative)
else:
    st.info("Please upload an image to get started.")
st.markdown("---") 
st.caption("Powered by MC-LLaVA-3B & Streamlit. For Best Results, use clear images, with obvious subjects.")   