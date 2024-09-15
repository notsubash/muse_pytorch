import streamlit as st
import torch
from muse_maskgit_pytorch import VQGanVAE, MaskGitTransformer, MaskGit, Muse
from PIL import Image
import io
import os
import logging

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_models():
    device = torch.device('cpu') ## Change to 'cuda' if gpu is available
    logging.info(f"Using device: {device}")

    try:
        vae = VQGanVAE(dim=256, codebook_size=65536).to(device)
        state_dict = torch.load("/Users/subash/Desktop/Scopic/Task_1/models/vae_model.pt", map_location=device, weights_only=True)
        vae.load_state_dict(state_dict)
        vae.eval()
        logging.info("VAE model loaded successfully")

        base_transformer = MaskGitTransformer(
            num_tokens = 65536,
            seq_len = 256,
            dim = 512,
            depth = 8,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            t5_name = 't5-small',
        )
        base_maskgit = MaskGit(
            vae = vae,
            transformer = base_transformer,
            image_size = 256,
            cond_drop_prob = 0.25,
        ).to(device)
        state_dict = torch.load("/Users/subash/Desktop/Scopic/Task_1/models/maskgit_model.pt", map_location=device, weights_only=True)
        base_maskgit.load_state_dict(state_dict)
        base_maskgit.eval()
        logging.info("MaskGit model loaded successfully")

        superres_transformer = MaskGitTransformer(
            num_tokens = 65536,
            seq_len = 1024,
            dim = 512,
            depth = 2,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            t5_name = 't5-small',
        )
        superres_maskgit = MaskGit(
            vae = vae,
            transformer = superres_transformer,
            image_size = 512,
            cond_drop_prob = 0.25,
            cond_image_size = 256,
        ).to(device)

        state_dict = torch.load("/Users/subash/Desktop/Scopic/Task_1/models/superres_maskgit_model.pt", map_location=device, weights_only=True)
        superres_maskgit.load_state_dict(state_dict)
        superres_maskgit.eval()
        logging.info("Superres MaskGit model loaded successfully")

        return Muse(base=base_maskgit, superres=superres_maskgit)

    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

st.title("MUSE Text-to-Image Generation")

muse = load_models()

prompt = st.text_input("Enter your text prompt:")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            generated_images = muse([prompt])
            img = generated_images[0]
            
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            st.image(img_byte_arr, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a text prompt.")

st.markdown("---")
st.write("This app uses the MUSE model to generate images from text prompts.")
