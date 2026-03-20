# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## 📈 金融数据查询路由

**唯一金融技能: `finance-monster`** — 所有金融相关问题统一使用。

### 中国A股/ETF/板块
```bash
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_market.py "大盘行情"
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_market.py "600519"
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_market.py "贵州茅台"
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_market.py "ETF 513180"
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_market.py "行业板块"
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_market.py "今日涨停"
```

### 中国期货/期权
```bash
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_derivatives.py futures-board --symbol PTA
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_derivatives.py futures-indicators --contract IF2603 --period 5
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_derivatives.py options-greeks --underlying 510050
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/china_derivatives.py options-rr25 --underlying 510050
```

### 美股/全球/加密货币
```bash
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/global_market.py AAPL --fast
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/dividends.py AAPL
```

### 投资组合/自选股/热点/传闻
```bash
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/portfolio.py list
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/watchlist.py list
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/hot_scanner.py
exec python ~/.openclaw/workspace/skills/finance-monster/scripts/rumor_scanner.py
```

❗ **严禁使用东方财富 `*_em` 批量接口！** 已全面反爬封禁。

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
