import asyncio
import glob
import logging
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.types import Message
from aiohttp import web
from PIL import Image
from transformers import pipeline

# Set BOT_TOKEN in Render Environment. Do not hardcode tokens in source code.
TOKEN = os.getenv("BOT_TOKEN", "").strip()
THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.70"))
DB_PATH = os.getenv("DB_PATH", "cache.db")
NOTIFY = os.getenv("NOTIFY", "0") == "1"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("nsfw-bot")

if not TOKEN:
    raise RuntimeError("BOT_TOKEN is missing. Add it in Render → Environment.")

def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

FFMPEG = get_ffmpeg()
classifier = None
sem = asyncio.Semaphore(1)

# SQLite cache is used for stickers, whose file_unique_id stays stable.
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS cache (uid TEXT PRIMARY KEY, score REAL)")
db.commit()

def cache_get(uid: str):
    row = db.execute("SELECT score FROM cache WHERE uid=?", (uid,)).fetchone()
    return row[0] if row else None

def cache_put(uid: str, score: float):
    db.execute("INSERT OR REPLACE INTO cache (uid, score) VALUES (?, ?)", (uid, score))
    db.commit()

def score_image(img: Image.Image) -> float:
    global classifier
    if classifier is None:
        raise RuntimeError("NSFW classifier has not loaded")
    img = img.convert("RGB")
    results = classifier(img)
    for item in results:
        if str(item.get("label", "")).strip().lower() in {"nsfw", "porn", "explicit"}:
            return float(item["score"])
    # If the model uses a different label name, log it and fail safe (do not delete).
    log.warning("Model returned labels not recognized as NSFW: %s", results)
    return 0.0

def score_static(path: str) -> float:
    with Image.open(path) as im:
        im.load()
        return score_image(im.copy())

def score_video(path: str) -> float:
    """Sample up to 5 frames from a video/sticker and return the highest NSFW score."""
    with tempfile.TemporaryDirectory() as out_dir:
        pattern = os.path.join(out_dir, "frame-%02d.jpg")
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
            "-vf", "fps=1,scale=640:-1:force_original_aspect_ratio=decrease",
            "-frames:v", "5", "-q:v", "4", pattern,
        ]
        subprocess.run(cmd, check=True, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        paths = sorted(glob.glob(os.path.join(out_dir, "frame-*.jpg")))
        scores = []
        for frame_path in paths:
            with Image.open(frame_path) as im:
                im.load()
                scores.append(score_image(im.copy()))
        return max(scores, default=0.0)

def score_tgs(path: str) -> float:
    from rlottie_python import LottieAnimation
    anim = LottieAnimation.from_tgs(path)
    total = anim.lottie_animation_get_totalframe()
    frame_ids = sorted({0, total // 2, max(total - 1, 0)})
    scores = []
    for frame_id in frame_ids:
        frame = anim.render_pillow_frame(frame_num=frame_id)
        scores.append(score_image(frame))
    return max(scores, default=0.0)

async def classify_file(path: str, kind: str) -> float:
    async with sem:
        if kind == "video":
            return await asyncio.to_thread(score_video, path)
        if kind == "tgs":
            return await asyncio.to_thread(score_tgs, path)
        return await asyncio.to_thread(score_static, path)

async def delete_if_nsfw(message: Message, file_id: str, uid: str, kind: str, suffix: str):
    try:
        score = cache_get(uid) if uid else None
        if score is None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "media" + suffix)
                await bot.download(file_id, destination=path)
                score = await classify_file(path, kind)
            if uid:
                cache_put(uid, score)

        log.info(
            "Checked media chat_id=%s message_id=%s kind=%s score=%.4f threshold=%.2f",
            message.chat.id, message.message_id, kind, score, THRESHOLD,
        )
        if score < THRESHOLD:
            return

        await message.delete()
        log.info("Deleted NSFW media chat_id=%s message_id=%s score=%.4f",
                 message.chat.id, message.message_id, score)
        if NOTIFY:
            try:
                await bot.send_message(message.chat.id, "Сообщение удалено: обнаружен контент 18+.")
            except Exception:
                log.exception("Deleted message, but could not send notification")
    except Exception:
        log.exception(
            "Failed to check/delete media chat_id=%s message_id=%s. "
            "Check bot admin permissions and media format.",
            message.chat.id, message.message_id,
        )

bot = Bot(TOKEN)
dp = Dispatcher()

def is_group(message: Message) -> bool:
    return message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)

@dp.message(F.sticker, F.chat.type.in_({"group", "supergroup"}))
async def on_sticker(message: Message):
    st = message.sticker
    if st.is_animated:
        kind, suffix = "tgs", ".tgs"
    elif st.is_video:
        kind, suffix = "video", ".webm"
    else:
        kind, suffix = "static", ".webp"
    await delete_if_nsfw(message, st.file_id, st.file_unique_id, kind, suffix)

@dp.message(F.photo, F.chat.type.in_({"group", "supergroup"}))
async def on_photo(message: Message):
    photo = message.photo[-1]
    await delete_if_nsfw(message, photo.file_id, photo.file_unique_id, "static", ".jpg")

@dp.message(F.document, F.chat.type.in_({"group", "supergroup"}))
async def on_document(message: Message):
    doc = message.document
    mime = (doc.mime_type or "").lower()
    name = (doc.file_name or "").lower()
    suffix = Path(name).suffix.lower()
    log.info("Received document chat_id=%s message_id=%s mime=%s filename=%s",
             message.chat.id, message.message_id, mime, name or "<no filename>")

    # GIF sent as a document must be sampled as an animation, not as a still image.
    if suffix == ".gif" or mime == "image/gif":
        await delete_if_nsfw(message, doc.file_id, doc.file_unique_id, "video", ".gif")
    elif mime.startswith("image/") or suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        await delete_if_nsfw(message, doc.file_id, doc.file_unique_id, "static",
                             suffix or ".img")
    elif mime.startswith("video/") or suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        await delete_if_nsfw(message, doc.file_id, doc.file_unique_id, "video",
                             suffix or ".mp4")

@dp.message(F.video, F.chat.type.in_({"group", "supergroup"}))
async def on_video(message: Message):
    video = message.video
    await delete_if_nsfw(message, video.file_id, video.file_unique_id, "video", ".mp4")

@dp.message(F.animation, F.chat.type.in_({"group", "supergroup"}))
async def on_animation(message: Message):
    animation = message.animation
    log.info("Received animation chat_id=%s message_id=%s file_name=%s mime=%s",
             message.chat.id, message.message_id, animation.file_name,
             animation.mime_type)
    await delete_if_nsfw(message, animation.file_id, animation.file_unique_id, "video", ".mp4")

async def health(request):
    return web.Response(text="ok")

async def main():
    app = web.Application()
    app.router.add_get("/", health)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "10000"))
    await web.TCPSite(runner, "0.0.0.0", port).start()
    log.info("Health server listening on port %s", port)

    global classifier
    classifier = await asyncio.to_thread(
        pipeline,
        "image-classification",
        model="Falconsai/nsfw_image_detection",
    )
    log.info("NSFW model loaded; starting Telegram polling")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await runner.cleanup()
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
