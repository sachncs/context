#!/usr/bin/env python3
"""Generate the social/OG card for the ceng site (1200x630)."""
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1200, 630
BG = (4, 5, 10, 255)

img = Image.new("RGBA", (W, H), BG)
draw = ImageDraw.Draw(img)

# Aurora glows
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
g = ImageDraw.Draw(glow)
for cx, cy, r, col in [
    (860, 120, 460, (61, 102, 255, 46)),
    (220, 560, 400, (122, 78, 255, 40)),
    (1080, 540, 300, (34, 211, 238, 26)),
]:
    g.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
glow = glow.filter(ImageFilter.GaussianBlur(90))
img.alpha_composite(glow)
draw = ImageDraw.Draw(img)

# Subtle grid
for x in range(0, W, 56):
    draw.line([(x, 0), (x, H)], fill=(255, 255, 255, 10), width=1)
for y in range(0, H, 56):
    draw.line([(0, y), (W, y)], fill=(255, 255, 255, 10), width=1)

# Fade to dark at bottom
fade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
fd = ImageDraw.Draw(fade)
for i in range(H):
    a = int(120 * (i / H) ** 2)
    fd.line([(0, i), (W, i)], fill=(4, 5, 10, a))
img.alpha_composite(fade)
draw = ImageDraw.Draw(img)

# Logo mark
def rounded_logo(x, y, s, pad=0.16 * 0):
    r = int(s * 0.28)
    grad = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    top = (93, 139, 255)
    mid = (111, 107, 255)
    bot = (154, 99, 255)
    gd.rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=top)
    overlay = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i, c in enumerate([mid, bot]):
        for yy in range(int(s * 0.66), s):
            od.line([(0, yy), (s, yy)], fill=(c[0], c[1], c[2], int(255 * (yy - s * 0.66) / (s * 0.34))))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=255)
    grad.paste(overlay, (0, 0), mask)
    img.paste(grad, (x, y))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x, y, x + s - 1, y + s - 1], radius=r, outline=(255, 255, 255, 70), width=2)
    # bars
    bar_w = int(s * 0.7)
    bar_h = int(s * 0.13)
    for i, (bw, op) in enumerate([(bar_w, 247), (int(bar_w * 0.72), 199), (bar_w, 148)]):
        bx = x + (s - bw) // 2
        by = y + int(s * 0.28) + i * int(s * 0.19)
        d.rounded_rectangle([bx, by, bx + bw, by + bar_h], radius=bar_h // 2, fill=(255, 255, 255, op))

rounded_logo(96, 96, 176)

# Typography
def font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:
        return ImageFont.load_default()

head = font("/System/Library/Fonts/Supplemental/Arial.ttf", 76)
sub = font("/System/Library/Fonts/Helvetica.ttc", 34)
tag = font("/System/Library/Fonts/Helvetica.ttc", 25)
mono = font("/System/Library/Fonts/Supplemental/Arial.ttf", 30)

# Headline with gradient: "Context," white, " engineered." gradient
base_y = 120
draw.text((330, base_y), "Context,", font=head, fill=(246, 248, 255, 255))
w = draw.textlength("Context,", font=head)

def gradient_text(x, y, word, fp, size):
    step = 1
    gx = x
    for ch in word:
        wch = draw.textlength(ch, font=fp)
        p = (gx - x) / draw.textlength(word, font=fp)
        r = int(93 + (154 - 93) * p)
        gg = int(139 + (99 - 139) * p)
        b = int(255 + (255 - 255) * p)
        draw.text((gx, y), ch, font=fp, fill=(r, gg, b, 255))
        gx += wch

gradient_text(330 + w + 22, base_y, "engineered.", head, 76)

# Subline
subline = "Research-grade context engineering for LLMs."
draw.text((330, 226), subline, font=sub, fill=(168, 180, 214, 255))

# tag chips
draw.text((330, 430), "ppa_compress", font=mono, fill=(146, 176, 255, 255))
draw.text((565, 430), "ppa_check", font=mono, fill=(146, 176, 255, 255))
draw.text((790, 430), "Evolver", font=mono, fill=(146, 176, 255, 255))
draw.text((1010, 430), "OKF", font=mono, fill=(146, 176, 255, 255))

# footer
draw.text((330, 520), "v0.4.0  ·  Apache-2.0  ·  Python 3.10–3.12", font=tag, fill=(120, 132, 164, 255))
draw.text((960, 520), "sachncs/context", font=tag, fill=(120, 132, 164, 255))

img.convert("RGB").save("/Users/sachin/repo/mygit/context/site/public/og.png", "PNG")
print("og.png written")