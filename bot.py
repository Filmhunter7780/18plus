import asyncio
import glob
import logging
import os
import sqlite3
import subprocess
import tempfile

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from PIL import Image
from transformers import pipeline

# Вставьте сюда НОВЫЙ токен от @BotFather (в кавычках),
# либо задайте переменную окружения BOT_TOKEN.
TOKEN = os.getenv("BOT_TOKEN") or "8620454579:AAHIGmMW5nT1fCP33vmTRlKCiH9bLxyBmgg"

THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.7"))  # 0..1, ниже = строже
DB_PATH = os.getenv("DB_PATH", "cache.db")
NOTIFY = os.getenv("NOTIFY", "1") == "1"  # писать в чат, что стикер удалён

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nsfw-bot")


def get_ffmpeg() -> str:
    """Берём ffmpeg из imageio-ffmpeg, если установлен, иначе системный."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG = get_ffmpeg()

# --- модель ---------------------------------------------------------------
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
# ограничиваем параллельные проверки, чтобы не положить CPU
sem = asyncio.Semaphore(2)


def score_image(img: Image.Image) -> float:
    """Вероятность класса nsfw для одной картинки."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    for r in classifier(img):
        if r["label"].lower() == "nsfw":
            return float(r["score"])
    return 0.0


# --- кэш ------------------------------------------------------------------
db = sqlite3.connect(DB_PATH)
db.execute("CREATE TABLE IF NOT EXISTS cache (uid TEXT PRIMARY KEY, score REAL)")
db.commit()


def cache_get(uid: str):
    row = db.execute("SELECT score FROM cache WHERE uid=?", (uid,)).fetchone()
    return row[0] if row else None


def cache_put(uid: str, score: float):
    db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (uid, score))
    db.commit()


# --- получение кадров из стикера -------------------------------------------
def frames_static(path: str):
    yield Image.open(path)


def frames_webm(path: str):
    """До 3 кадров через ffmpeg (с поддержкой прозрачности VP9)."""
    out_dir = tempfile.mkdtemp()
    subprocess.run(
        [FFMPEG, "-v", "error", "-c:v", "libvpx-vp9", "-i", path,
         "-vf", "fps=2", "-frames:v", "3", f"{out_dir}/f%d.png"],
        check=True, timeout=30,
    )
    for p in sorted(glob.glob(f"{out_dir}/*.png")):
        yield Image.open(p)


def frames_tgs(path: str):
    """3 кадра из Lottie-анимации (.tgs) через rlottie-python."""
    from rlottie_python import LottieAnimation

    anim = LottieAnimation.from_tgs(path)
    total = anim.lottie_animation_get_totalframe()
    for i in {0, total // 2, max(total - 1, 0)}:
        yield anim.render_pillow_frame(frame_num=i)


def compute_score(path: str, kind: str) -> float:
    gen = {"static": frames_static, "video": frames_webm, "tgs": frames_tgs}[kind]
    return max((score_image(f) for f in gen(path)), default=0.0)


# --- бот ------------------------------------------------------------------
bot = Bot(TOKEN)
dp = Dispatcher()


@dp.message(F.sticker, F.chat.type.in_({"group", "supergroup"}))
async def on_sticker(message: Message):
    st = message.sticker
    uid = st.file_unique_id

    score = cache_get(uid)
    if score is None:
        kind = "tgs" if st.is_animated else "video" if st.is_video else "static"
        suffix = {"tgs": ".tgs", "video": ".webm", "static": ".webp"}[kind]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sticker" + suffix)
            await bot.download(st.file_id, destination=path)
            try:
                async with sem:
                    score = await asyncio.to_thread(compute_score, path, kind)
            except Exception:
                log.exception("Не удалось проверить стикер %s", uid)
                return  # при ошибке не удаляем и не кэшируем
        cache_put(uid, score)
        log.info("sticker %s set=%s score=%.3f", uid, st.set_name, score)

    if score >= THRESHOLD:
        try:
            await message.delete()
        except Exception:
            log.exception("Не удалось удалить сообщение (нет прав?)")
            return
        if NOTIFY:
            who = message.from_user.mention_html() if message.from_user else "Участник"
            await bot.send_message(
                message.chat.id, f"{who}, стикер удалён (18+).", parse_mode="HTML"
            )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())