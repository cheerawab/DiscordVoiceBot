# 🎙️ Voicebot - Discord 語音辨識機器人

使用 [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)（CTranslate2 後端）對 Discord 語音頻道中的成員進行語音辨識 (STT)，並將辨識結果輸出到文字頻道。

## 功能

- `!join` — 讓機器人加入你所在的語音頻道，開始語音辨識
- `!leave` — 讓機器人離開語音頻道

## 前置需求

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/)（音訊處理所需）
- Discord Bot Token（需啟用 **Message Content Intent** 與 **Voice** 權限）

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## 安裝

```bash
# 1. 安裝 Python 套件
pip install -r requirements.txt

# 2. 複製環境變數範本並填入你的 Bot Token
cp .env.example .env
# 編輯 .env，填入 DISCORD_TOKEN
```

## 設定說明

在 `.env` 中可設定以下參數：

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `DISCORD_TOKEN` | Discord Bot Token（**必填**） | — |
| `WHISPER_MODEL` | Whisper 模型大小 (`tiny` / `base` / `small` / `medium` / `large-v3` / `turbo`) | `base` |
| `WHISPER_LANGUAGE` | 辨識語言代碼（`zh` / `en` / `ja` 等） | `zh` |
| `COMPUTE_TYPE` | 量化類型（`int8` / `float16` / `float32`） | `int8` |
| `WHISPER_DEVICE` | 推論裝置（`auto` / `cpu` / `cuda`） | `auto` |
| `TEXT_CHANNEL_ID` | 指定輸出結果的文字頻道 ID（留空則用下指令的頻道） | — |

## 啟動

```bash
python3 bot.py
```

## Discord Bot 設定提醒

1. 前往 [Discord Developer Portal](https://discord.com/developers/applications) 建立應用程式
2. 在 **Bot** 頁面啟用 **Message Content Intent**
3. 在 **OAuth2 → URL Generator** 中勾選：
   - Scopes: `bot`
   - Bot Permissions: `Send Messages`, `Connect`, `Speak`, `Use Voice Activity`
4. 使用產生的連結邀請 Bot 到你的伺服器

## 運作原理

1. 使用者輸入 `!join`，Bot 加入語音頻道
2. Bot 透過 `discord-ext-voice-recv` 接收每位成員的 PCM 語音資料
3. 當偵測到使用者停止說話（靜音 1.5 秒），將累積的音訊送入 Faster-Whisper 辨識
4. 辨識結果以「🎙️ **使用者名稱**：辨識文字」格式發送到文字頻道
