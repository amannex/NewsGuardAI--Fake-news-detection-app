from PIL import Image, ImageDraw, ImageFont
import os

sizes = [16, 48, 128]
out_dir = "extension/icons"

for size in sizes:
    img = Image.new('RGB', (size, size), color = (37, 99, 235))
    d = ImageDraw.Draw(img)
    # Just a simple text 'N' for NewsGuard
    # Since we might not have a TTF handy, we just draw a rectangle or use default font
    d.text((size/4, size/4), "N", fill=(255,255,255))
    img.save(f"{out_dir}/icon{size}.png")

print("Icons generated!")
