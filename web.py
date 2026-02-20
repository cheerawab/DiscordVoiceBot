"""
Web Dashboard - 語音機器人控制面板

提供即時的語音辨識串流檢視、Prompt 設定、機器人狀態監控。
使用 aiohttp 與 Discord bot 共用同一個 asyncio 事件迴圈。
"""

import os
import asyncio
import json
import time
from pathlib import Path
from aiohttp import web

# ─── 共用狀態 ─────────────────────────────────────────────

# 最近的辨識記錄（最多保留 200 筆）
transcription_log: list[dict] = []
MAX_LOG_SIZE = 200

# SSE 訂閱者
_sse_queues: list[asyncio.Queue] = []

PROMPT_FILE = os.getenv("WHISPER_PROMPT_FILE", "prompt.txt")
# 轉為絕對路徑（以本檔案所在目錄為基準）
if not os.path.isabs(PROMPT_FILE):
    PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), PROMPT_FILE)
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")

# bot 實例（由 bot.py 注入）
_bot_ref = None
_voice_managers_ref = None


def set_bot_ref(bot, voice_managers):
    """由 bot.py 呼叫，注入 bot 與 voice_managers 參照"""
    global _bot_ref, _voice_managers_ref
    _bot_ref = bot
    _voice_managers_ref = voice_managers


def add_transcription(user_name: str, text: str, duration: float, guild_name: str = ""):
    """新增一筆辨識記錄，同時推送 SSE"""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_name,
        "text": text,
        "duration": round(duration, 1),
        "guild": guild_name,
    }
    transcription_log.append(entry)
    if len(transcription_log) > MAX_LOG_SIZE:
        transcription_log[:] = transcription_log[-MAX_LOG_SIZE:]

    # 推送到所有 SSE 訂閱者
    data = json.dumps(entry, ensure_ascii=False)
    for q in _sse_queues:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


# ─── API 路由 ─────────────────────────────────────────────

async def handle_index(request):
    """回傳 Dashboard HTML"""
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    html = html_path.read_text(encoding="utf-8")
    return web.Response(text=html, content_type="text/html")


async def handle_get_prompt(request):
    """取得目前 prompt"""
    try:
        text = open(PROMPT_FILE, "r", encoding="utf-8").read()
    except FileNotFoundError:
        text = ""
    return web.json_response({"prompt": text})


async def handle_set_prompt(request):
    """更新 prompt"""
    data = await request.json()
    new_prompt = data.get("prompt", "")
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(new_prompt)
    return web.json_response({"ok": True, "prompt": new_prompt})


async def handle_status(request):
    """取得機器人目前狀態"""
    status = {
        "bot_name": str(_bot_ref.user) if _bot_ref and _bot_ref.user else "未連線",
        "bot_id": str(_bot_ref.user.id) if _bot_ref and _bot_ref.user else "",
        "guilds": [],
    }

    if _bot_ref:
        for guild in _bot_ref.guilds:
            guild_info = {
                "name": guild.name,
                "id": str(guild.id),
                "voice_connected": False,
                "voice_channel": "",
                "active_users": 0,
            }
            # 檢查該 guild 的語音連線
            vc = guild.voice_client
            if vc and vc.is_connected():
                guild_info["voice_connected"] = True
                guild_info["voice_channel"] = vc.channel.name if vc.channel else ""

            # 活躍使用者數
            if _voice_managers_ref and guild.id in _voice_managers_ref:
                mgr = _voice_managers_ref[guild.id]
                guild_info["active_users"] = sum(
                    1 for s in mgr.user_states.values() if s.has_voice
                )

            status["guilds"].append(guild_info)

    return web.json_response(status)


async def handle_log(request):
    """取得歷史辨識記錄"""
    limit = int(request.query.get("limit", "50"))
    entries = transcription_log[-limit:]
    return web.json_response(entries)


async def handle_sse(request):
    """SSE 端點 - 即時串流辨識結果"""
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    queue = asyncio.Queue(maxsize=100)
    _sse_queues.append(queue)

    try:
        # 發送心跳以保持連線
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=15)
                await response.write(f"data: {data}\n\n".encode("utf-8"))
            except asyncio.TimeoutError:
                # 心跳
                await response.write(b": heartbeat\n\n")
            except ConnectionResetError:
                break
    finally:
        _sse_queues.remove(queue)

    return response


# ─── 建立與啟動 Web App ──────────────────────────────────

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/prompt", handle_get_prompt)
    app.router.add_post("/api/prompt", handle_set_prompt)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/log", handle_log)
    app.router.add_get("/api/stream", handle_sse)
    return app


async def start_web_server():
    """啟動 web server（在既有事件迴圈中）"""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, WEB_HOST, WEB_PORT)
    await site.start()
    print(f"🌐 Web Dashboard 已啟動：http://{WEB_HOST}:{WEB_PORT}")
    return runner
