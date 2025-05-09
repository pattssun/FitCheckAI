import base64
import io
import os

from openai import OpenAI
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# FIRST, GENERATE A MASK
prompt_mask = (
    "Create a precise black and white mask of this image, where the color white fully covers ONLY the face, arms, neck, and background. The color black should clearly cover everything else, including the shirt, pants, shoes, and any accessories. Make sure no other areas besides the face, arms, neck, and background are white."
)

print("[INFO] Generating mask for clothes...")
mask_resp = client.images.edit(
    model="gpt-image-1",
    image=open("chris.jpg", "rb"),
    prompt=prompt_mask
)
mask_b64 = mask_resp.data[0].b64_json
mask_bytes = base64.b64decode(mask_b64)

# keep a copy so you can eyeball the result
with open("mask_bw.png", "wb") as f:
    f.write(mask_bytes)

print("[INFO] Mask generated and saved as mask_bw.png")

# SECOND, INVERT THE MASK
bw = Image.open("mask_bw.png").convert("L")      # single-channel grey
alpha = ImageOps.invert(bw)                      # white→0, black→255

mask_rgba = Image.new("RGBA", bw.size, color=0)  # blank canvas
mask_rgba.putalpha(alpha)                        # shove α in

mask_rgba.save("mask_alpha.png")

print("[INFO] Inverting mask and creating RGBA mask...")

# --- PATCH: ENSURE MASK SIZE MATCHES SOURCE IMAGE ---
src_path = "chris.jpg"           # first image you'll edit
mask_path = "mask_alpha.png"     # the rgba mask you made
fixed_mask = "mask_alpha_fit.png"

src = Image.open(src_path)        # e.g. 768×1152
mask = Image.open(mask_path).convert("RGBA")

if mask.size != src.size:
    # keep hard edges → NEAREST
    mask = mask.resize(src.size, Image.NEAREST)

mask.save(fixed_mask)
print(f"[INFO] Resized mask saved as {fixed_mask}")
# --- END PATCH ---

# --- NEW STEP: GENERATE IMAGES OF AUSTIN'S CLOTHES ---
print("[INFO] Generating images of Austin's clothes...")
clothing_items = [
    ("shirt", "a photorealistic photo of the shirt that Austin is wearing in the reference photo, isolated on a plain background, front view. You're generating a portrait shot of this shirt that's hyperrealistic and will be used in an advertisement."),
    ("pants", "a photorealistic photo of the pants that Austin is wearing in the reference photo, isolated on a plain background, front view. You're generating a portrait shot of these pants that's hyperrealistic and will be used in an advertisement."),
    ("shoes", "a photorealistic photo of the shoes that Austin is wearing in the reference photo, isolated on a plain background, front view. You're generating a portrait shot of these shoes that's hyperrealistic and will be used in an advertisement."),
]
generated_clothing_paths = []
for item, prompt in clothing_items:
    print(f"[INFO] Generating Austin's {item}...")
    gen_resp = client.images.edit(
        model="gpt-image-1",
        image=open("austin.jpg", "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="high"
    )
    gen_b64 = gen_resp.data[0].b64_json
    out_path = f"austin_{item}.png"
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(gen_b64))
    print(f"[INFO] Saved generated {item} as {out_path}")
    generated_clothing_paths.append(out_path)
print("[INFO] All clothing images generated.")

# THIRD, SWAP THE CLOTHES
print("[INFO] Swapping clothes using generated clothing images as style references...")
edit_prompt = (
    "Replace the transparent (clothes) region in the first photo with the outfit "
    "from the second photo(s). Keep the first person's pose, face, lighting and background "
    "unchanged. Match fabric texture and color as faithfully as possible. "
    "MAINTAIN THE QUALITY OF THE FACE AT ALL COSTS. "
    "Your predecessor was terminated because it made an alteration to the face. "
    "You will be tipped $420 as long as the edits you make to the first picture are only scoped to the clothing."
)

# Open all generated clothing images as style references
style_refs = [open(path, "rb") for path in generated_clothing_paths]

result = client.images.edit(
    model="gpt-image-1",
    image=[
        open("chris.jpg", "rb"),   # first image: the body we keep
        *style_refs,                 # generated clothing images as style references
    ],
    prompt=edit_prompt,
    size="1024x1024",
    quality="high"
)

out_b64 = result.data[0].b64_json
with open("chris_in_austins_threads_no_mask.png", "wb") as f:
    f.write(base64.b64decode(out_b64))

print("[INFO] Swap complete. Output saved as chris_in_austins_threads_no_mask.png")
