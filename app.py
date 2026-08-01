import base64
import io
import json
import os
import subprocess
import tempfile
import time

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont

from google import genai
from google.genai import types

app = FastAPI(title="AdGen AI Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AI Analysis ────────────────────────────────────────────────────────────────

def analyze_product_and_generate_script(
    image_bytes: bytes, tone: str, duration: int, extra_info: str, api_key: str
) -> dict:
    client = genai.Client(api_key=api_key)

    tone_map = {
        "Luxury & Premium": "sophisticated, elegant, aspirational — think Apple or Rolex",
        "Fun & Energetic": "upbeat, youthful, vibrant — think Red Bull or Skittles",
        "Trustworthy & Professional": "confident, clear, reliable — think IBM or Dove",
        "Emotional & Storytelling": "warm, human, narrative-driven — think Nike or Coca-Cola",
    }

    num_frames = max(3, duration // 3)

    prompt = f"""You are an expert advertising creative director. Analyze the product in this image and create a complete advertisement video script.

Tone style: {tone_map.get(tone, tone)}
Video duration: {duration} seconds
Extra context from user: {extra_info or 'None provided'}

Return ONLY a valid JSON object (no markdown fences, no extra text) with this exact structure:
{{
  "product_name": "detected product name",
  "product_description": "1-sentence description of what you see",
  "tagline": "punchy 5-8 word tagline for the ad",
  "frames": [
    {{
      "title": "short frame title",
      "headline": "bold headline text (max 6 words)",
      "subtext": "supporting copy (max 12 words)",
      "visual_note": "brief note on what this frame shows visually",
      "duration_seconds": 3
    }}
  ],
  "color_palette": {{
    "primary": "#hexcode",
    "secondary": "#hexcode",
    "accent": "#hexcode",
    "text": "#hexcode"
  }},
  "full_script": "Complete voiceover script for the full ad"
}}

Generate exactly {num_frames} frames with a compelling story arc: Hook → Problem/Desire → Solution → Call to Action.
Make the tagline memorable and copy punchy. Color palette should match the product aesthetic."""

    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    text_part = types.Part.from_text(text=prompt)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[image_part, text_part])],
        config=types.GenerateContentConfig(max_output_tokens=2048, temperature=0.7),
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Frame Rendering ────────────────────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def create_ad_frame(
    product_img: Image.Image,
    headline: str,
    subtext: str,
    tagline: str,
    palette: dict,
    frame_index: int,
    total_frames: int,
    product_name: str,
) -> Image.Image:
    W, H = 1080, 720
    canvas = Image.new("RGB", (W, H), color=(10, 10, 20))

    primary_rgb = hex_to_rgb(palette.get("primary", "#16a34a"))
    secondary_rgb = hex_to_rgb(palette.get("secondary", "#2563eb"))
    accent_rgb = hex_to_rgb(palette.get("accent", "#4ade80"))

    for y in range(H):
        t = y / H
        r = max(0, int(primary_rgb[0] * (1 - t) + secondary_rgb[0] * t) - 180)
        g = max(0, int(primary_rgb[1] * (1 - t) + secondary_rgb[1] * t) - 180)
        b = max(0, int(primary_rgb[2] * (1 - t) + secondary_rgb[2] * t) - 180)
        for x in range(W):
            canvas.putpixel((x, y), (r, g, b))

    draw = ImageDraw.Draw(canvas)

    for x in range(W):
        t = x / W
        r = int(accent_rgb[0] * t + primary_rgb[0] * (1 - t))
        g = int(accent_rgb[1] * t + primary_rgb[1] * (1 - t))
        b = int(accent_rgb[2] * t + primary_rgb[2] * (1 - t))
        for dy in range(4):
            canvas.putpixel((x, dy), (r, g, b))

    img_area_w, img_area_h = 420, 460
    img_x = W - img_area_w - 40
    img_y = (H - img_area_h) // 2

    product_copy = product_img.copy()
    product_copy.thumbnail((img_area_w, img_area_h), Image.LANCZOS)

    glow = Image.new("RGBA", (img_area_w + 80, img_area_h + 80), (0, 0, 0, 0))
    glow_d = ImageDraw.Draw(glow)
    ar, ag, ab = accent_rgb
    for i in range(40, 0, -1):
        glow_d.ellipse(
            [40 - i, 40 - i, img_area_w + 40 + i, img_area_h + 40 + i],
            fill=(ar, ag, ab, int(6 * i)),
        )
    canvas.paste(glow.convert("RGB"), (img_x - 40, img_y - 40), mask=glow.split()[3])

    pw, ph = product_copy.size
    px = img_x + (img_area_w - pw) // 2
    py = img_y + (img_area_h - ph) // 2
    if product_copy.mode in ("RGBA", "LA"):
        canvas.paste(product_copy, (px, py), mask=product_copy.split()[-1])
    else:
        canvas.paste(product_copy, (px, py))

    text_x = 60
    text_w = W - img_area_w - 100

    def load_font(size, bold=False):
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    font_brand = load_font(18, bold=True)
    font_headline = load_font(52, bold=True)
    font_sub = load_font(22)
    font_tagline = load_font(20, bold=True)
    font_frame = load_font(14)

    text_color = (240, 240, 255)
    muted_color = (160, 165, 195)

    draw.text((text_x, 55), product_name.upper(), font=font_brand, fill=accent_rgb)

    dot_y = 85
    for i in range(total_frames):
        dot_x = text_x + i * 18
        color = accent_rgb if i == frame_index else (60, 65, 80)
        draw.ellipse([dot_x, dot_y, dot_x + 8, dot_y + 8], fill=color)

    headline_y = dot_y + 35
    words = headline.split()
    lines, cur = [], []
    for word in words:
        test = " ".join(cur + [word])
        if draw.textbbox((0, 0), test, font=font_headline)[2] > text_w and cur:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))

    for i, line in enumerate(lines[:3]):
        draw.text((text_x, headline_y + i * 62), line, font=font_headline, fill=text_color)

    hl_end_y = headline_y + len(lines[:3]) * 62 + 10
    draw.rectangle([text_x, hl_end_y, text_x + 50, hl_end_y + 3], fill=accent_rgb)

    sub_y = hl_end_y + 20
    sub_words = subtext.split()
    sub_lines, sub_cur = [], []
    for word in sub_words:
        test = " ".join(sub_cur + [word])
        if draw.textbbox((0, 0), test, font=font_sub)[2] > text_w and sub_cur:
            sub_lines.append(" ".join(sub_cur))
            sub_cur = [word]
        else:
            sub_cur.append(word)
    if sub_cur:
        sub_lines.append(" ".join(sub_cur))

    for i, line in enumerate(sub_lines[:3]):
        draw.text((text_x, sub_y + i * 32), line, font=font_sub, fill=muted_color)

    if frame_index == total_frames - 1:
        tag_y = H - 100
        tw = draw.textbbox((0, 0), tagline, font=font_tagline)[2]
        draw.rectangle(
            [text_x - 5, tag_y - 8, text_x + tw + 15, tag_y + 30],
            fill=(*accent_rgb, 30),
        )
        draw.text((text_x, tag_y), f'"{tagline}"', font=font_tagline, fill=accent_rgb)

    draw.text(
        (text_x, H - 30),
        f"Frame {frame_index+1} of {total_frames}",
        font=font_frame,
        fill=(80, 85, 100),
    )
    draw.text((W - 180, H - 30), "Powered by Gemini AI", font=font_frame, fill=(60, 65, 80))

    return canvas


def create_video_from_frames(
    frames: list, output_path: str, fps: int = 24, frame_duration: int = 3
):
    with tempfile.TemporaryDirectory() as tmp:
        idx = 0
        for frame_img in frames:
            buf = io.BytesIO()
            frame_img.convert("RGB").save(buf, format="JPEG", quality=95)
            buf.seek(0)
            reloaded = Image.open(buf)
            for _ in range(fps * frame_duration):
                reloaded.save(f"{tmp}/frame_{idx:06d}.jpg", quality=95)
                idx += 1

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{tmp}/frame_%06d.jpg",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "22",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


GEMINI_API_KEY = "AQ.Ab8RN6KR1HSCkGVcx6SRFoKLxEt_yvdcNCAeL08rhQyKuZxNVA"


@app.post("/api/generate")
async def generate_ad(
    image: UploadFile = File(...),
    tone: str = Form("Luxury & Premium"),
    duration: int = Form(15),
    extra_info: str = Form(""),
):
    """
    Accepts a product image + settings, returns:
      - script data (JSON)
      - base64-encoded frame previews
      - base64-encoded final MP4
    """
    api_key = GEMINI_API_KEY

    # Read & normalize image
    raw_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(raw_bytes))
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()

    # Step 1 — AI analysis
    try:
        script_data = analyze_product_and_generate_script(
            jpeg_bytes, tone, duration, extra_info, api_key
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")

    palette = script_data.get(
        "color_palette",
        {"primary": "#16a34a", "secondary": "#2563eb", "accent": "#4ade80", "text": "#ffffff"},
    )
    product_name = script_data["product_name"]
    tagline = script_data["tagline"]
    ad_frames = script_data["frames"]
    total = len(ad_frames)

    # Step 2 — Render frames
    product_pil = Image.open(io.BytesIO(raw_bytes))
    frames_rendered = []
    frame_previews_b64 = []

    for i, frame_data in enumerate(ad_frames):
        frame_img = create_ad_frame(
            product_img=product_pil,
            headline=frame_data["headline"],
            subtext=frame_data["subtext"],
            tagline=tagline,
            palette=palette,
            frame_index=i,
            total_frames=total,
            product_name=product_name,
        )
        frames_rendered.append(frame_img)

        preview_buf = io.BytesIO()
        frame_img.save(preview_buf, format="JPEG", quality=80)
        frame_previews_b64.append(base64.b64encode(preview_buf.getvalue()).decode())

    # Step 3 — Encode video
    frame_dur = max(2, duration // total)
    out_path = f"/tmp/adgen_{int(time.time())}.mp4"

    try:
        create_video_from_frames(frames_rendered, out_path, fps=24, frame_duration=frame_dur)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg error: {str(e)}")

    with open(out_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    os.remove(out_path)

    return JSONResponse({
        "script": script_data,
        "frame_previews": frame_previews_b64,
        "video_b64": video_b64,
        "meta": {
            "total_frames": total,
            "frame_duration": frame_dur,
            "estimated_duration": total * frame_dur,
            "tone": tone,
        },
    })