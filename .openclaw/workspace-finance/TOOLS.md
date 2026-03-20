# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## 📈 金融数据查询路由

### ETF基金查询（51/15/16/50/58开头的代码）
**必须使用** `akshare-wrapper` 技能的脚本：
```bash
exec python ~/.openclaw/workspace/skills/akshare-wrapper/main.py "513180 今日表现"
```
❗ **不要** 在内联代码中调用 `fund_etf_hist_em()` 或 `fund_etf_spot_em()` 或 `stock_zh_a_spot_em()` —— 东方财富接口反爬会导致超时断连。

### A股个股/大盘/板块查询
**推荐使用** `akshare-wrapper` 技能的脚本：
```bash
exec python ~/.openclaw/workspace/skills/akshare-wrapper/main.py "A股大盘"
exec python ~/.openclaw/workspace/skills/akshare-wrapper/main.py "600519 最新行情"
```

### 美股/加密货币
使用 `stock-analysis` 技能（Yahoo Finance）。

---

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.
