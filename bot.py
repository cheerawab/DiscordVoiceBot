"""
Discord Voice Bot - 使用 Faster-Whisper 進行語音辨識 (STT)

使用 discord.py + discord-ext-voice-recv 接收語音，
搭配 Faster-Whisper (CTranslate2) 辨識後直接發送訊息。

運作模式：
  - 收到語音封包 → 累積到使用者的 buffer
  - 偵測到靜音（一句話結束）→ 辨識整段音訊 → 發送新訊息
  - 每一句話都是獨立的一則訊息

指令:
  !join  - 讓機器人加入你所在的語音頻道，開始語音辨識
  !leave - 讓機器人離開語音頻道
"""

import os
import asyncio
import time
from collections import defaultdict

import discord
from discord.ext import commands
from discord.ext.voice_recv import VoiceRecvClient, BasicSink, VoiceData
from faster_whisper import WhisperModel
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from web import set_bot_ref, add_transcription, start_web_server

# ─── 設定 ───────────────────────────────────────────────
TOKEN = os.getenv("DISCORD_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")   # tiny / base / small / medium / large-v3 / turbo
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")      # float16 / int8 / int8_float16 / float32
DEVICE = os.getenv("WHISPER_DEVICE", "auto")           # auto / cpu / cuda
TEXT_CHANNEL_ID = os.getenv("TEXT_CHANNEL_ID")        # 可選：指定輸出辨識結果的文字頻道 ID
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "zh")        # 預設辨識語言
PROMPT_FILE = os.getenv("WHISPER_PROMPT_FILE", "prompt.txt")  # 提示詞檔案路徑
if not os.path.isabs(PROMPT_FILE):
    PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), PROMPT_FILE)

def load_prompt() -> str | None:
    """每次辨識前從檔案動態載入 prompt，檔案不存在或為空則回傳 None"""
    try:
        text = open(PROMPT_FILE, "r", encoding="utf-8").read().strip()
        return text or None
    except FileNotFoundError:
        return None

# 辨識設定
SILENCE_TIMEOUT = 1.5       # 靜音超過此秒數視為一句話結束
SILENCE_THRESHOLD = 0.01    # RMS 能量門檻，低於此值視為靜音（0.0~1.0）
MIN_AUDIO_DURATION = 0.5    # 最短音訊長度（秒），過短則忽略
SAMPLE_RATE = 48000          # Discord 語音取樣率
CHANNELS = 2                 # Discord 語音聲道數（立體聲）
SAMPLE_WIDTH = 2             # 16-bit PCM = 2 bytes
WHISPER_SR = 16000           # Whisper 需要的取樣率
MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", "120"))  # 模型載入逾時（秒），僅用於已快取的情況

# ─── 模型名稱對應 HF repo ─────────────────────────────────
# faster-whisper 內部的名稱映射（參考 faster_whisper/utils.py）
_MODEL_REPO_MAP = {
    "tiny":       "Systran/faster-whisper-tiny",
    "tiny.en":    "Systran/faster-whisper-tiny.en",
    "base":       "Systran/faster-whisper-base",
    "base.en":    "Systran/faster-whisper-base.en",
    "small":      "Systran/faster-whisper-small",
    "small.en":   "Systran/faster-whisper-small.en",
    "medium":     "Systran/faster-whisper-medium",
    "medium.en":  "Systran/faster-whisper-medium.en",
    "large-v1":   "Systran/faster-whisper-large-v1",
    "large-v2":   "Systran/faster-whisper-large-v2",
    "large-v3":   "Systran/faster-whisper-large-v3",
    "large":      "Systran/faster-whisper-large-v3",
    "turbo":      "Systran/faster-whisper-large-v3-turbo",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

def _get_repo_id(model_name: str) -> str:
    """取得模型對應的 HF repo ID"""
    if "/" in model_name:
        return model_name
    return _MODEL_REPO_MAP.get(model_name, f"Systran/faster-whisper-{model_name}")

def _check_model_cached(model_name: str) -> bool:
    """檢查模型是否已下載至本地快取"""
    from huggingface_hub import try_to_load_from_cache
    repo_id = _get_repo_id(model_name)
    result = try_to_load_from_cache(repo_id, "model.bin")
    return result is not None and isinstance(result, str)

def _download_model(model_name: str):
    """預先下載模型檔案（有進度條），確保快取後再做載入"""
    from huggingface_hub import snapshot_download
    repo_id = _get_repo_id(model_name)
    print(f"📥 開始下載模型 '{repo_id}'...")
    print(f"   （大型模型約 3GB，依網路速度可能需要 5-30 分鐘）")
    snapshot_download(
        repo_id,
        allow_patterns=["*.bin", "*.json", "*.txt", "*.md"],
    )
    print(f"✅ 模型 '{repo_id}' 下載完成！")

# ─── 載入 Faster-Whisper 模型（含預下載與降級）──────────────
def _load_whisper_model():
    global _device, _compute
    _device = DEVICE
    _compute = COMPUTE_TYPE

    # 1) 檢查快取，必要時先下載
    try:
        cached = _check_model_cached(WHISPER_MODEL)
        if not cached:
            print(f"⚠️  模型 '{WHISPER_MODEL}' 尚未快取")
            _download_model(WHISPER_MODEL)
        else:
            print(f"✅ 模型 '{WHISPER_MODEL}' 已在本地快取中")
    except Exception as e:
        print(f"⚠️  模型快取檢查/下載階段發生錯誤：{e}")
        print("ℹ️  仍嘗試繼續載入（WhisperModel 會自行處理下載）...")

    # 2) 載入模型（此時檔案應已在本地，設合理逾時）
    print(f"正在載入 Faster-Whisper 模型: {WHISPER_MODEL} (device={_device}, compute_type={_compute}) ...")

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            lambda: WhisperModel(WHISPER_MODEL, device=_device, compute_type=_compute)
        )
        try:
            model = future.result(timeout=MODEL_LOAD_TIMEOUT)
            print(f"✅ Faster-Whisper 模型載入完成！(device={_device}, compute_type={_compute})")
            return model
        except concurrent.futures.TimeoutError:
            print(f"❌ 模型載入逾時（超過 {MODEL_LOAD_TIMEOUT} 秒）！")
            if _device != "cpu":
                print("⚠️  嘗試降級為 CPU (int8) 模式...")
                _device = "cpu"
                _compute = "int8"
            else:
                raise RuntimeError(
                    f"模型 '{WHISPER_MODEL}' 載入逾時。\n"
                    f"可能原因：系統記憶體不足（large 模型需約 6GB RAM）"
                )
        except Exception as e:
            err_msg = str(e)
            cuda_errors = ("libcublas", "libcudnn", "libcublasLt", "CUDA",
                           "out of memory", "OOM", "cudaMalloc",
                           "cudaErrorMemoryAllocation", "CUDNN")
            if any(kw.lower() in err_msg.lower() for kw in cuda_errors):
                print(f"⚠️  CUDA 載入失敗：{err_msg}")
                print("⚠️  自動降級為 CPU (int8) 模式...")
                _device = "cpu"
                _compute = "int8"
            else:
                raise

        # 降級重試
        print(f"正在以 CPU (int8) 模式重新載入...")
        future = executor.submit(
            lambda: WhisperModel(WHISPER_MODEL, device=_device, compute_type=_compute)
        )
        try:
            model = future.result(timeout=MODEL_LOAD_TIMEOUT)
            print(f"✅ Faster-Whisper 模型載入完成！(device={_device}, compute_type={_compute})")
            return model
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                f"模型 '{WHISPER_MODEL}' 即使用 CPU 模式也載入逾時。\n"
                f"可能原因：系統記憶體（RAM）不足（large 模型需約 6GB）\n"
                f"建議：嘗試較小的模型如 'medium' 或 'small'"
            )

whisper_model = _load_whisper_model()

# ─── Warmup：預熱模型避免首次推論延遲 ────────────────────
print("正在預熱模型...")
_warmup_audio = np.zeros(WHISPER_SR, dtype=np.float32)  # 1 秒靜音
list(whisper_model.transcribe(_warmup_audio, language=LANGUAGE, beam_size=1))
del _warmup_audio
print("模型預熱完成！")

# ─── 降低 voice_recv 的日誌噪音 ─────────────────────────
import logging
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.CRITICAL)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.CRITICAL)
logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.CRITICAL)
logging.getLogger("discord.ext.voice_recv.router").setLevel(logging.CRITICAL)

# ─── Bot 設定 ────────────────────────────────────────────
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)


# ─── 使用者語音狀態 ──────────────────────────────────────

class UserVoiceState:
    """追蹤每位使用者的語音緩衝區"""

    def __init__(self):
        self.buffer = bytearray()
        self.last_voice_time: float = 0.0   # 最後一次偵測到有聲音的時間
        self.has_voice: bool = False         # 是否曾偵測到有聲音
        self.processing: bool = False

    def add_data(self, data: bytes):
        # 用 RMS 能量判斷是否有聲音
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) > 0:
            rms = np.sqrt(np.mean(samples ** 2))
            is_voice = rms > SILENCE_THRESHOLD
        else:
            is_voice = False

        if self.has_voice:
            # 已在錄音中：所有封包都存（保持連續性）
            self.buffer.extend(data)
            if is_voice:
                self.last_voice_time = time.time()
        elif is_voice:
            # 首次偵測到聲音：開始錄音
            self.buffer.extend(data)
            self.last_voice_time = time.time()
            self.has_voice = True
        # 否則：尚未開始說話，丟棄靜音封包

    def get_duration(self) -> float:
        return len(self.buffer) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)

    def consume(self) -> bytes:
        data = bytes(self.buffer)
        self.buffer.clear()
        return data

    def is_silent_for(self, timeout: float) -> bool:
        """音訊能量低於門檻超過 timeout 秒"""
        if self.last_voice_time == 0:
            return False
        return (time.time() - self.last_voice_time) > timeout

    def reset(self):
        self.buffer.clear()
        self.has_voice = False
        self.processing = False


# ─── PCM → Whisper 轉換工具 ─────────────────────────────

def pcm_to_whisper_array(pcm_data: bytes) -> np.ndarray | None:
    """將 PCM 音訊轉換為 16kHz float32 mono numpy array"""
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) == 0:
        return None

    if CHANNELS == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    # 降取樣 48kHz → 16kHz
    ratio = SAMPLE_RATE // WHISPER_SR
    samples = samples[::ratio]

    if len(samples) < int(WHISPER_SR * MIN_AUDIO_DURATION):
        return None

    return samples


def run_whisper(audio: np.ndarray) -> str:
    """同步執行 Faster-Whisper 推論，回傳辨識文字"""
    prompt = load_prompt()
    t0 = time.time()
    segments, _info = whisper_model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=1,              # 貪婪解碼，速度最快
        vad_filter=False,         # 已用 RMS 做斷句，不需要 Whisper VAD
        initial_prompt=prompt,
    )
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.time() - t0
    audio_len = len(audio) / WHISPER_SR
    print(f"  [Whisper] 辨識完成：{elapsed:.2f}s（音訊 {audio_len:.1f}s，RTF={elapsed/audio_len:.2f}）")
    return text


# ─── 語音管理器 ──────────────────────────────────────────

class VoiceManager:
    """
    語音辨識管理器。

    運作流程：
    1. 收到語音封包 → 累積到使用者的 buffer
    2. 偵測到靜音（一句話結束）→ 辨識整段音訊
    3. 發送新訊息顯示辨識結果
    """

    def __init__(self, text_channel: discord.TextChannel, bot_instance: commands.Bot):
        self.text_channel = text_channel
        self.bot_instance = bot_instance
        self.user_states: dict[int, UserVoiceState] = defaultdict(UserVoiceState)
        self._running = True
        self._monitor_task: asyncio.Task | None = None

    def start(self):
        self._monitor_task = self.bot_instance.loop.create_task(self._monitor_loop())

    def stop(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        self.user_states.clear()

    def on_voice_data(self, member: discord.Member | discord.User | None, voice_data: VoiceData):
        if member is None:
            return
        pcm = voice_data.pcm
        if pcm:
            self.user_states[member.id].add_data(pcm)

    async def _monitor_loop(self):
        """主迴圈：偵測靜音結束後送辨識"""
        while self._running:
            await asyncio.sleep(0.2)

            for uid, state in list(self.user_states.items()):
                if not state.has_voice:
                    continue

                if state.is_silent_for(SILENCE_TIMEOUT) and not state.processing:
                    if state.get_duration() >= MIN_AUDIO_DURATION:
                        state.processing = True
                        pcm_data = state.consume()
                        asyncio.create_task(self._transcribe_and_send(uid, pcm_data, state))
                    else:
                        state.reset()

    async def _transcribe_and_send(self, user_id: int, pcm_data: bytes, state: UserVoiceState):
        """辨識音訊並發送新訊息"""
        try:
            audio = pcm_to_whisper_array(pcm_data)
            if audio is None:
                return

            text = await asyncio.get_running_loop().run_in_executor(None, run_whisper, audio)
            if not text:
                return

            member = self.text_channel.guild.get_member(user_id)
            name = member.display_name if member else f"User#{user_id}"

            # CLI 即時顯示
            duration = len(pcm_data) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
            print(f"[{time.strftime('%H:%M:%S')}] 🎙️ {name}（{duration:.1f}s）：{text}")

            await self.text_channel.send(f"🎙️ **{name}**：{text}")

            # 推送到 Web Dashboard
            guild_name = self.text_channel.guild.name if self.text_channel.guild else ""
            add_transcription(name, text, duration, guild_name)

        except Exception as e:
            print(f"[辨識錯誤] user_id={user_id}: {e}")
        finally:
            state.processing = False
            # 只有在辨識期間沒有新語音資料時才完全重置
            if len(state.buffer) == 0:
                state.has_voice = False
                state.last_voice_time = 0.0


# 儲存每個 guild 的 VoiceManager 實例
voice_managers: dict[int, VoiceManager] = {}


# ─── Bot 事件 ────────────────────────────────────────────

@bot.event
async def on_ready():
    user = bot.user
    if user is None:
        print("Bot 已上線，但尚未取得 user 資訊。")
        return

    print(f"Bot 已上線：{user} (ID: {user.id})")
    print("------")

    # 啟動 Web Dashboard
    set_bot_ref(bot, voice_managers)
    await start_web_server()


# ─── 指令 ────────────────────────────────────────────────

@bot.command(name="join", help="讓機器人加入你所在的語音頻道")
async def join(ctx: commands.Context):
    if ctx.guild is None:
        await ctx.send("❌ 此指令只能在伺服器中使用。")
        return

    if not isinstance(ctx.author, discord.Member) or not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("❌ 你必須先加入一個語音頻道！")
        return

    guild = ctx.guild
    voice_channel = ctx.author.voice.channel

    # 如果 Bot 已在某語音頻道，先斷開
    if ctx.voice_client:
        old_mgr = voice_managers.pop(guild.id, None)
        if old_mgr:
            old_mgr.stop()
        voice_client = ctx.voice_client
        try:
            if isinstance(voice_client, VoiceRecvClient):
                voice_client.stop_listening()
        except Exception:
            pass
        await voice_client.disconnect(force=True)
        await asyncio.sleep(1)

    # 連接語音頻道，最多重試 3 次
    vc = None
    for attempt in range(3):
        try:
            vc = await voice_channel.connect(cls=VoiceRecvClient, timeout=30.0)
            break
        except TimeoutError:
            print(f"[語音連線] 第 {attempt + 1} 次嘗試超時...")
            if ctx.voice_client:
                voice_client = ctx.voice_client
                try:
                    if isinstance(voice_client, VoiceRecvClient):
                        voice_client.stop_listening()
                    await voice_client.disconnect(force=True)
                except Exception:
                    pass
                await asyncio.sleep(2)

    if vc is None:
        await ctx.send("❌ 無法連接到語音頻道，請稍後再試。")
        return

    # 決定辨識結果輸出的文字頻道
    text_channel: discord.TextChannel | None = ctx.channel if isinstance(ctx.channel, discord.TextChannel) else None
    if TEXT_CHANNEL_ID:
        ch = bot.get_channel(int(TEXT_CHANNEL_ID))
        if isinstance(ch, discord.TextChannel):
            text_channel = ch
        elif ch is not None:
            await ctx.send("❌ TEXT_CHANNEL_ID 必須是文字頻道。")
            return

    if text_channel is None:
        await ctx.send("❌ 請在文字頻道中使用此指令，或設定有效的 TEXT_CHANNEL_ID。")
        return

    # 建立 VoiceManager 並開始監聽
    mgr = VoiceManager(text_channel, bot)
    voice_managers[guild.id] = mgr

    sink = BasicSink(mgr.on_voice_data)
    vc.listen(sink)
    mgr.start()

    await ctx.send(
        f"✅ 已加入語音頻道：**{voice_channel.name}**\n"
        f"🎧 開始監聽語音，辨識結果將顯示在此頻道。\n"
        f"📝 模型：`{WHISPER_MODEL}` ｜語言：`{LANGUAGE}`"
    )


@bot.command(name="leave", help="讓機器人離開語音頻道")
async def leave(ctx: commands.Context):
    if ctx.guild is None:
        await ctx.send("❌ 此指令只能在伺服器中使用。")
        return

    if not ctx.voice_client:
        await ctx.send("❌ 我目前不在任何語音頻道中。")
        return

    guild = ctx.guild
    mgr = voice_managers.pop(guild.id, None)
    if mgr:
        mgr.stop()

    voice_client = ctx.voice_client
    try:
        if isinstance(voice_client, VoiceRecvClient):
            voice_client.stop_listening()
    except Exception:
        pass
    await voice_client.disconnect(force=True)
    await ctx.send("👋 已離開語音頻道。")


# ─── 啟動 ────────────────────────────────────────────────

if __name__ == "__main__":
    if not TOKEN:
        print("錯誤：請在 .env 檔案中設定 DISCORD_TOKEN")
        exit(1)
    bot.run(TOKEN)
