import streamlit as st
from google import genai
from google.genai import types
import base64
import json
import os
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
import io
import subprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdGen AI — Product Ad Maker",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Space+Grotesk:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #111827 50%, #0d1117 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4ade80, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.hero-sub {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.05em;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(12px);
}

.card-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.5rem;
}

.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #16a34a, #2563eb);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 0.6rem;
}

.step-row {
    display: flex;
    align-items: center;
    color: #e5e7eb;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.75rem;
}

.stFileUploader label, .stSelectbox label,
.stTextInput label, .stTextArea label, .stSlider label {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #16a34a 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(22,163,74,0.4);
}

.script-box {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 12px;
    padding: 1.25rem;
    color: #d1d5db;
    font-size: 0.88rem;
    line-height: 1.7;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

.info-text {
    color: #6b7280;
    font-size: 0.82rem;
    text-align: center;
    margin-top: 0.5rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

.api-box {
    background: rgba(22,163,74,0.06);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def analyze_product_and_generate_script(
    image_bytes: bytes, tone: str, duration: int, extra_info: str, api_key: str
) -> dict:
    """Use Gemini vision to analyze the product and generate ad script + frames."""

    client = genai.Client(api_key=api_key)

    tone_map = {
        "Luxury & Premium":          "sophisticated, elegant, aspirational — think Apple or Rolex",
        "Fun & Energetic":           "upbeat, youthful, vibrant — think Red Bull or Skittles",
        "Trustworthy & Professional":"confident, clear, reliable — think IBM or Dove",
        "Emotional & Storytelling":  "warm, human, narrative-driven — think Nike or Coca-Cola",
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

    # Build multimodal request: image + text
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    text_part  = types.Part.from_text(text=prompt)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[image_part, text_part])],
        config=types.GenerateContentConfig(
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if Gemini adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


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
    """Render a single advertisement frame (1080×720) with PIL."""
    W, H = 1080, 720
    canvas = Image.new("RGB", (W, H), color=(10, 10, 20))

    primary_rgb   = hex_to_rgb(palette.get("primary",   "#16a34a"))
    secondary_rgb = hex_to_rgb(palette.get("secondary", "#2563eb"))
    accent_rgb    = hex_to_rgb(palette.get("accent",    "#4ade80"))

    # Gradient background
    for y in range(H):
        t = y / H
        r = max(0, int(primary_rgb[0]*(1-t) + secondary_rgb[0]*t) - 180)
        g = max(0, int(primary_rgb[1]*(1-t) + secondary_rgb[1]*t) - 180)
        b = max(0, int(primary_rgb[2]*(1-t) + secondary_rgb[2]*t) - 180)
        for x in range(W):
            canvas.putpixel((x, y), (r, g, b))

    draw = ImageDraw.Draw(canvas)

    # Top accent stripe
    for x in range(W):
        t = x / W
        r = int(accent_rgb[0]*t + primary_rgb[0]*(1-t))
        g = int(accent_rgb[1]*t + primary_rgb[1]*(1-t))
        b = int(accent_rgb[2]*t + primary_rgb[2]*(1-t))
        for dy in range(4):
            canvas.putpixel((x, dy), (r, g, b))

    # Product image (right panel)
    img_area_w, img_area_h = 420, 460
    img_x = W - img_area_w - 40
    img_y = (H - img_area_h) // 2

    product_copy = product_img.copy()
    product_copy.thumbnail((img_area_w, img_area_h), Image.LANCZOS)

    # Glow halo
    glow = Image.new("RGBA", (img_area_w + 80, img_area_h + 80), (0, 0, 0, 0))
    glow_d = ImageDraw.Draw(glow)
    ar, ag, ab = accent_rgb
    for i in range(40, 0, -1):
        glow_d.ellipse(
            [40-i, 40-i, img_area_w+40+i, img_area_h+40+i],
            fill=(ar, ag, ab, int(6*i))
        )
    canvas.paste(glow.convert("RGB"), (img_x-40, img_y-40), mask=glow.split()[3])

    pw, ph = product_copy.size
    px = img_x + (img_area_w - pw) // 2
    py = img_y + (img_area_h - ph) // 2
    if product_copy.mode in ("RGBA", "LA"):
        canvas.paste(product_copy, (px, py), mask=product_copy.split()[-1])
    else:
        canvas.paste(product_copy, (px, py))

    # Text area (left panel)
    text_x = 60
    text_w  = W - img_area_w - 100

    def load_font(size, bold=False):
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    font_brand    = load_font(18, bold=True)
    font_headline = load_font(52, bold=True)
    font_sub      = load_font(22)
    font_tagline  = load_font(20, bold=True)
    font_frame    = load_font(14)

    text_color  = (240, 240, 255)
    muted_color = (160, 165, 195)

    # Brand name
    draw.text((text_x, 55), product_name.upper(), font=font_brand, fill=accent_rgb)

    # Progress dots
    dot_y = 85
    for i in range(total_frames):
        dot_x = text_x + i * 18
        color = accent_rgb if i == frame_index else (60, 65, 80)
        draw.ellipse([dot_x, dot_y, dot_x+8, dot_y+8], fill=color)

    # Headline (word-wrapped)
    headline_y = dot_y + 35
    words = headline.split()
    lines, cur = [], []
    for word in words:
        test = " ".join(cur + [word])
        if draw.textbbox((0,0), test, font=font_headline)[2] > text_w and cur:
            lines.append(" ".join(cur)); cur = [word]
        else:
            cur.append(word)
    if cur: lines.append(" ".join(cur))

    for i, line in enumerate(lines[:3]):
        draw.text((text_x, headline_y + i*62), line, font=font_headline, fill=text_color)

    hl_end_y = headline_y + len(lines[:3])*62 + 10
    draw.rectangle([text_x, hl_end_y, text_x+50, hl_end_y+3], fill=accent_rgb)

    # Subtext
    sub_y = hl_end_y + 20
    sub_words = subtext.split()
    sub_lines, sub_cur = [], []
    for word in sub_words:
        test = " ".join(sub_cur + [word])
        if draw.textbbox((0,0), test, font=font_sub)[2] > text_w and sub_cur:
            sub_lines.append(" ".join(sub_cur)); sub_cur = [word]
        else:
            sub_cur.append(word)
    if sub_cur: sub_lines.append(" ".join(sub_cur))

    for i, line in enumerate(sub_lines[:3]):
        draw.text((text_x, sub_y + i*32), line, font=font_sub, fill=muted_color)

    # Tagline on last frame
    if frame_index == total_frames - 1:
        tag_y = H - 100
        tw = draw.textbbox((0,0), tagline, font=font_tagline)[2]
        draw.rectangle([text_x-5, tag_y-8, text_x+tw+15, tag_y+30],
                       fill=(*accent_rgb, 30))
        draw.text((text_x, tag_y), f'"{tagline}"', font=font_tagline, fill=accent_rgb)

    draw.text((text_x, H-30), f"Frame {frame_index+1} of {total_frames}",
              font=font_frame, fill=(80, 85, 100))

    # Gemini badge (bottom right)
    draw.text((W-180, H-30), "Powered by Gemini AI",
              font=font_frame, fill=(60, 65, 80))

    return canvas


def create_video_from_frames(
    frames: list, output_path: str, fps: int = 24, frame_duration: int = 3
):
    """Stitch PIL frames into MP4 via ffmpeg."""
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


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🎬 AdGen AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">TURN ANY PRODUCT PHOTO INTO A POLISHED ADVERTISEMENT VIDEO · POWERED BY GEMINI</div>',
    unsafe_allow_html=True,
)

# ── API Key input ─────────────────────────────────────────────────────────────
st.markdown('<div class="api-box">', unsafe_allow_html=True)
api_key = st.text_input(
    "🔑 Google Gemini API Key",
    type="password",
    placeholder="Paste your API key from aistudio.google.com …",
    help="Your key is never stored — it only lives in memory for this session.",
)
st.markdown(
    '<div class="info-text">Get a free key at <strong>aistudio.google.com</strong> → Get API key</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div class="card-label">Step 1</div>
  <div class="step-row"><span class="step-badge">1</span> Upload your product photo</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a product image here (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file:
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(uploaded_file, use_container_width=True, caption="Your product")
    with col_info:
        st.markdown(f"""
        <div style="padding:1rem 0;color:#9ca3af;font-size:0.85rem;">
            <div style="color:#4ade80;font-weight:700;margin-bottom:0.5rem;">✓ Image ready</div>
            <div>{uploaded_file.name}</div>
            <div>{uploaded_file.size // 1024} KB</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Step 2: Settings ───────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div class="card-label">Step 2</div>
  <div class="step-row"><span class="step-badge">2</span> Configure your ad</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox(
        "Ad Tone",
        ["Luxury & Premium", "Fun & Energetic", "Trustworthy & Professional", "Emotional & Storytelling"],
    )
with col2:
    duration = st.slider("Video Duration (seconds)", min_value=9, max_value=30, value=15, step=3)

extra_info = st.text_area(
    "Any extra context? (product name, target audience, key message…)",
    placeholder="e.g. 'Handmade leather wallet for young professionals. Emphasize durability and style.'",
    height=80,
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Step 3: Generate ───────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div class="card-label">Step 3</div>
  <div class="step-row"><span class="step-badge">3</span> Generate your ad video</div>
</div>
""", unsafe_allow_html=True)

generate_btn = st.button("⚡  Generate Advertisement Video", use_container_width=True)

if generate_btn:
    if not api_key:
        st.error("Please enter your Gemini API key above.")
    elif not uploaded_file:
        st.error("Please upload a product image first.")
    else:
        # Phase 1 — AI analysis
        with st.status("🤖 Gemini is analyzing your product…", expanded=True) as status:
            st.write("Sending image to Gemini 2.0 Flash Vision…")
            image_bytes = uploaded_file.read()

            pil_img = Image.open(io.BytesIO(image_bytes))
            buf = io.BytesIO()
            pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
            jpeg_bytes = buf.getvalue()

            try:
                script_data = analyze_product_and_generate_script(
                    jpeg_bytes, tone, duration, extra_info, api_key
                )
                st.write(f"✅ Product detected: **{script_data['product_name']}**")
                st.write(f"✅ Generated {len(script_data['frames'])} ad frames")
                st.write(f"✅ Tagline: *\"{script_data['tagline']}\"*")
                status.update(label="✅ Gemini analysis complete!", state="complete")
            except Exception as e:
                status.update(label="❌ AI analysis failed", state="error")
                st.error(f"Gemini API error: {e}")
                st.stop()

        # Script preview
        with st.expander("📄 View Generated Ad Script", expanded=False):
            st.markdown(f"""
            <div class="script-box"><strong>PRODUCT:</strong> {script_data['product_name']}
<strong>TAGLINE:</strong> {script_data['tagline']}

<strong>VOICEOVER SCRIPT:</strong>
{script_data['full_script']}

<strong>FRAMES:</strong>
{chr(10).join(f"  [{i+1}] {f['headline']} — {f['subtext']}" for i, f in enumerate(script_data['frames']))}
            </div>
            """, unsafe_allow_html=True)

        # Phase 2 — Render frames
        with st.status("🎨 Rendering advertisement frames…", expanded=True) as status:
            frames_rendered = []
            palette = script_data.get("color_palette", {
                "primary": "#16a34a", "secondary": "#2563eb",
                "accent": "#4ade80", "text": "#ffffff"
            })
            product_name = script_data["product_name"]
            tagline      = script_data["tagline"]
            ad_frames    = script_data["frames"]
            total        = len(ad_frames)

            product_pil       = Image.open(io.BytesIO(image_bytes))
            progress          = st.progress(0, text="Rendering frames…")
            frame_preview_slot = st.empty()

            for i, frame_data in enumerate(ad_frames):
                st.write(f"Rendering frame {i+1}: *{frame_data['title']}*")
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
                progress.progress((i+1)/total, text=f"Frame {i+1}/{total}")
                preview_buf = io.BytesIO()
                frame_img.save(preview_buf, format="JPEG", quality=85)
                frame_preview_slot.image(
                    preview_buf.getvalue(),
                    caption=f"Frame {i+1}: {frame_data['title']}",
                    use_container_width=True,
                )

            status.update(label=f"✅ {total} frames rendered!", state="complete")

        # Phase 3 — Encode video
        with st.status("🎬 Encoding video…", expanded=True) as status:
            st.write("Stitching frames with ffmpeg…")
            out_path   = f"/tmp/adgen_{int(time.time())}.mp4"
            frame_dur  = max(2, duration // total)

            try:
                create_video_from_frames(frames_rendered, out_path, fps=24, frame_duration=frame_dur)
                status.update(label="✅ Video encoded!", state="complete")
            except Exception as e:
                status.update(label="❌ Encoding failed", state="error")
                st.error(f"ffmpeg error: {e}\n\nInstall with: sudo apt-get install ffmpeg")
                st.stop()

        # Result
        st.success("🎉 Your advertisement video is ready!")

        col_dl, col_meta = st.columns([2, 1])
        with col_dl:
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                label="⬇️  Download Advertisement Video (.mp4)",
                data=video_bytes,
                file_name=f"ad_{product_name.replace(' ', '_').lower()}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        with col_meta:
            st.markdown(f"""
            <div style="padding:1rem;background:rgba(255,255,255,0.04);border-radius:12px;border:1px solid rgba(255,255,255,0.08);">
                <div style="color:#4ade80;font-size:0.7rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.75rem;">Ad Details</div>
                <div style="color:#e5e7eb;font-size:0.82rem;line-height:2;">
                    🏷️ <strong>{product_name}</strong><br>
                    ✨ <em>"{tagline}"</em><br>
                    🎞️ {total} frames<br>
                    ⏱️ ~{total * frame_dur}s<br>
                    🤖 Gemini 2.0 Flash<br>
                    🎨 {tone}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Frame gallery
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#6b7280;font-size:0.8rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1rem;">FRAME GALLERY</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(total, 3))
        for i, (frame_img, frame_data) in enumerate(zip(frames_rendered, ad_frames)):
            buf = io.BytesIO()
            frame_img.save(buf, format="JPEG", quality=80)
            with cols[i % min(total, 3)]:
                st.image(buf.getvalue(), caption=frame_data["title"], use_container_width=True)

# Footer
st.markdown("""
<div class="info-text" style="margin-top:3rem;">
    Powered by Gemini 2.0 Flash Vision · AdGen AI · Built with Streamlit
</div>
""", unsafe_allow_html=True)