# OpenClaw 跨电脑迁移指南

> 将已配置好的 OpenClaw 从一台 Windows 电脑迁移到另一台。

## 迁移内容概览

| 内容 | 是否可直接复制 | 说明 |
|---|---|---|
| `~/.openclaw/agents/*/agent/models.json` | ✅ 直接可用 | 模型配置，无路径依赖 |
| `~/.openclaw/agents/*/agent/auth-profiles.json` | ✅ 直接可用 | API Key |
| `~/.openclaw/workspace*/` 下人格文件 | ✅ 直接可用 | SOUL.md、IDENTITY.md 等纯文本 |
| `~/.openclaw/skills/` | ✅ 直接可用 | 已安装的 skill |
| `~/.openclaw/openclaw.json` | ⚠️ 需改用户名 | 包含硬编码绝对路径 |
| 用户级环境变量 | ❌ 需重新设置 | 不在文件系统中 |
| Gateway 计划任务 | ❌ 需重新注册 | Windows 计划任务不随文件迁移 |
| Node.js / OpenClaw / ClawHub | ❌ 需重新安装 | 运行时依赖 |

---

## 一、旧电脑：打包配置

```powershell
Compress-Archive -Path "$env:USERPROFILE\.openclaw" -DestinationPath "$env:USERPROFILE\Desktop\openclaw-backup.zip"
```

将生成的 `openclaw-backup.zip` 传到新电脑。

---

## 二、新电脑：安装运行时

```powershell
# 安装 Node.js（v22+）
winget install OpenJS.NodeJS.LTS

# 重新打开终端后安装 OpenClaw 和 ClawHub CLI
npm install -g openclaw clawhub
```

---

## 三、新电脑：解压配置

```powershell
# 将 openclaw-backup.zip 放到新电脑任意位置，然后解压到用户目录
Expand-Archive -Path "openclaw-backup.zip" -DestinationPath "$env:USERPROFILE"
```

解压后目录结构为 `C:\Users\<新用户名>\.openclaw\`。

---

## 四、新电脑：修改硬编码路径

`openclaw.json` 中有硬编码的旧电脑用户名路径，需要替换：

```powershell
# 将 yanmeizhao 替换为新电脑的用户名
$config = Get-Content "$env:USERPROFILE\.openclaw\openclaw.json" -Raw
$config = $config -replace 'yanmeizhao', $env:USERNAME
Set-Content "$env:USERPROFILE\.openclaw\openclaw.json" $config -Encoding UTF8
```

> **需要替换的路径示例：**
> - `C:\Users\yanmeizhao\.openclaw\workspace` → `C:\Users\新用户名\.openclaw\workspace`
> - `C:\Users\yanmeizhao\.openclaw\workspace-research` → `C:\Users\新用户名\.openclaw\workspace-research`
> - `C:\Users\yanmeizhao\.openclaw\agents\research\agent` → `C:\Users\新用户名\.openclaw\agents\research\agent`

---

## 五、新电脑：设置用户级环境变量

OpenClaw Gateway 以 Windows 计划任务运行，读不到终端临时变量，必须设为**用户级**永久环境变量：

```powershell
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_VERSION", "2025-04-01-preview", "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_BASE_URL", "https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai", "User")
```

> 如果 Azure 资源不同（换了订阅/区域），请替换为新的 endpoint URL。

---

## 六、新电脑：启动 Gateway

```powershell
openclaw gateway start
```

验证：

```powershell
# 查看 agent 列表
openclaw agents list

# 查看 skill 状态
openclaw skills list

# 打开 Dashboard 测试
openclaw dashboard
```

---

## 完整一键脚本

将以下内容保存为 `migrate-openclaw.ps1`，在新电脑上运行：

```powershell
# ===== 按实际修改 =====
$OLD_USERNAME = "yanmeizhao"                    # 旧电脑用户名
$BACKUP_ZIP   = "$env:USERPROFILE\Desktop\openclaw-backup.zip"  # zip 文件路径
$AZURE_API_VERSION = "2025-04-01-preview"
$AZURE_BASE_URL    = "https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai"
# ======================

# 1. 解压
Write-Host "[1/4] 解压配置..."
Expand-Archive -Path $BACKUP_ZIP -DestinationPath $env:USERPROFILE -Force

# 2. 替换用户名
Write-Host "[2/4] 修改路径中的用户名: $OLD_USERNAME -> $env:USERNAME"
$config = Get-Content "$env:USERPROFILE\.openclaw\openclaw.json" -Raw
$config = $config -replace [regex]::Escape($OLD_USERNAME), $env:USERNAME
Set-Content "$env:USERPROFILE\.openclaw\openclaw.json" $config -Encoding UTF8

# 3. 设置环境变量
Write-Host "[3/4] 设置用户级环境变量..."
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_VERSION", $AZURE_API_VERSION, "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_BASE_URL", $AZURE_BASE_URL, "User")

# 4. 启动 gateway
Write-Host "[4/4] 启动 Gateway..."
openclaw gateway start

Write-Host ""
Write-Host "迁移完成！运行以下命令验证："
Write-Host "  openclaw agents list"
Write-Host "  openclaw dashboard"
```

---

## 注意事项

1. **API Key 安全**：`auth-profiles.json` 包含明文 API Key，传输 zip 文件时注意安全（不要上传到公开仓库）。
2. **OpenAI OAuth Token**：如果 main agent 配有 OpenAI OAuth（`openai-codex:default`），token 有过期时间，迁移后可能需要重新 `openclaw configure` 登录。
3. **Skill 更新**：迁移后建议运行 `clawhub update --all` 更新所有 skill。
4. **Session 历史**：`~/.openclaw/agents/*/sessions/` 中的历史会话也会一并迁移，如不需要可删除以减小 zip 体积。
