# OpenClaw 跨电脑迁移指南

> 将已配置好的 OpenClaw 从一台 Windows 电脑迁移到另一台。
> 本机用户名: `yanmeizhao` | 本文档基于 2026-03-20 最新配置更新。

---

## 【核心速查】需要在新电脑上修改的所有参数

> 从 GitHub 仓库 `git clone` 后，以下参数需要根据新电脑环境修改。

### 参数清单总表

| # | 文件 | 参数/字段 | 当前值（本机） | 新电脑修改说明 |
|---|------|----------|-------------|-------------|
| 1 | `.openclaw/openclaw.json` | `agents.defaults.workspace` | `C:\Users\yanmeizhao\.openclaw\workspace` | 替换为 `C:\Users\<新用户名>\.openclaw\workspace` |
| 2 | `.openclaw/openclaw.json` | `agents.list[1].workspace` (research) | `C:\Users\yanmeizhao\.openclaw\workspace-research` | 替换为 `C:\Users\<新用户名>\.openclaw\workspace-research` |
| 3 | `.openclaw/openclaw.json` | `agents.list[1].agentDir` (research) | `C:\Users\yanmeizhao\.openclaw\agents\research\agent` | 替换为 `C:\Users\<新用户名>\.openclaw\agents\research\agent` |
| 4 | `.openclaw/openclaw.json` | `agents.list[2].workspace` (finance) | `C:\Users\yanmeizhao\.openclaw\workspace-finance` | 替换为 `C:\Users\<新用户名>\.openclaw\workspace-finance` |
| 5 | `.openclaw/openclaw.json` | `agents.list[2].agentDir` (finance) | `C:\Users\yanmeizhao\.openclaw\agents\finance\agent` | 替换为 `C:\Users\<新用户名>\.openclaw\agents\finance\agent` |
| 6 | `.openclaw/openclaw.json` | `gateway.auth.token` | `dccb8f41c7d2d0233dfc9755532834f922488e7c496b0afc` | 可保留或重新生成（仅 localhost 访问，安全性不高） |
| 7 | `.openclaw/openclaw.json` | `skills.entries.nano-banana-pro.apiKey` | `AIzaSyDqK2NGd9FiwY5Ixq-xzL2sJsZhRyPW4CI` | 如使用 Google 服务则填你自己的 API Key，否则忽略 |
| 8 | `.openclaw/agents/main/agent/auth-profiles.json` | `profiles.azure-openai-responses:default.token` | `<YOUR_AZURE_OPENAI_API_KEY>` | **必填** — Azure AI Services 的 API Key |
| 9 | `.openclaw/agents/main/agent/auth-profiles.json` | `profiles.openai-codex:default.access` | `<YOUR_OPENAI_ACCESS_TOKEN>` | OpenAI OAuth — 运行 `openclaw configure` 重新登录即可自动获取 |
| 10 | `.openclaw/agents/main/agent/auth-profiles.json` | `profiles.openai-codex:default.refresh` | `<YOUR_OPENAI_REFRESH_TOKEN>` | 同上，自动获取 |
| 11 | `.openclaw/agents/main/agent/auth-profiles.json` | `profiles.openai-codex:default.accountId` | `<YOUR_OPENAI_ACCOUNT_ID>` | 同上，自动获取 |
| 12 | `.openclaw/agents/research/agent/auth-profiles.json` | 同 #8~#11 | 同上 | **与 main 相同的值**，需同步修改 |
| 13 | `.openclaw/agents/finance/agent/auth-profiles.json` | 同 #8~#11 | 同上 | **与 main 相同的值**，需同步修改 |
| 14 | `.openclaw/agents/*/agent/models.json` | `providers.azure-openai-responses.baseUrl` | `https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai` | 如使用不同 Azure 资源则修改为新的 endpoint |
| 15 | `.openclaw/agents/*/agent/models.json` | `providers.azure-openai-responses.apiVersion` | `2025-04-01-preview` | 一般保持不变 |
| 16 | `.openclaw/agents/*/agent/models.json` | `providers.ollama.baseUrl` | `http://127.0.0.1:11434` | 如新电脑也装了 Ollama 则保持；否则可忽略 |
| 17 | `.openclaw/identity/device.json` | `deviceId`, `publicKeyPem`, `privateKeyPem` | 本机设备身份密钥 | **不复制**，新电脑执行 `openclaw configure` 会自动生成 |
| 18 | `.openclaw/identity/device-auth.json` | `tokens.operator.token` | operator token | **不复制**，新电脑自动生成 |
| 19 | 用户级环境变量 | `AZURE_OPENAI_API_VERSION` | `2025-04-01-preview` | 必须在新电脑设置（见下方命令） |
| 20 | 用户级环境变量 | `AZURE_OPENAI_BASE_URL` | `https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai` | 必须在新电脑设置（见下方命令） |
| 21 | Windows 计划任务 | `OpenClaw Gateway` | 端口 18789 | 需在新电脑重新注册（见下方命令） |
| 22 | Python 依赖 | `akshare`, `pandas`, `numpy` | 已安装 | 新电脑需 `pip install akshare pandas numpy` |

---

## 迁移内容概览

| 内容 | 是否可直接复制 | 说明 |
|---|---|---|
| `agents/*/agent/models.json` | ✅ 直接可用 | 模型配置（如用同一 Azure 资源） |
| `agents/*/agent/auth-profiles.json` | ⚠️ 需填入真实密钥 | GitHub 上已替换为占位符 |
| `workspace*/` 人格文件 | ✅ 直接可用 | SOUL.md、IDENTITY.md 等纯文本 |
| `workspace/skills/` | ✅ 直接可用 | 已安装的 skill 及修改后的脚本 |
| `openclaw.json` | ⚠️ 需改用户名 | 包含 5 处硬编码绝对路径 |
| `identity/` | ❌ 不复制 | 设备密钥，新电脑自动生成 |
| 用户级环境变量 | ❌ 需重新设置 | 不在文件系统中 |
| Gateway 计划任务 | ❌ 需重新注册 | Windows 计划任务不随文件迁移 |
| Node.js / OpenClaw / Python | ❌ 需重新安装 | 运行时依赖 |

---

## 详细步骤

### 第一步：新电脑安装运行时

```powershell
# 安装 Node.js（v22+）
winget install OpenJS.NodeJS.LTS

# 重新打开终端后安装 OpenClaw 和 ClawHub CLI
npm install -g openclaw clawhub

# 安装 Python 依赖（akshare 金融数据技能需要）
pip install akshare pandas numpy
```

### 第二步：从 GitHub 拉取配置

```powershell
cd $env:USERPROFILE
git clone https://github.com/endy-see/CodeRepo.git _temp_repo
cd _temp_repo
git checkout myClaw

# 复制 .openclaw 到用户目录（跳过 identity 目录）
robocopy ".openclaw" "$env:USERPROFILE\.openclaw" /MIR /XD ".git" "identity"
```

### 第三步：替换用户名路径（5处）

```powershell
$config = Get-Content "$env:USERPROFILE\.openclaw\openclaw.json" -Raw
$config = $config -replace [regex]::Escape("yanmeizhao"), $env:USERNAME
Set-Content "$env:USERPROFILE\.openclaw\openclaw.json" $config -Encoding UTF8
```

> 验证替换结果：
> ```powershell
> Select-String "yanmeizhao" "$env:USERPROFILE\.openclaw\openclaw.json"
> # 应无输出，表示替换完毕
> ```

### 第四步：填入 Azure API Key（最关键）

从 Azure 门户获取 Key：
1. 登录 https://portal.azure.com
2. 找到资源 `ym-claude-opus-dev-0316-resource` → 密钥和终结点
3. 复制 Key 1

然后替换 3 个 agent 的 auth-profiles.json：

```powershell
$AZURE_KEY = "在此粘贴你的Azure API Key"

# main agent
$f = "$env:USERPROFILE\.openclaw\agents\main\agent\auth-profiles.json"
(Get-Content $f -Raw) -replace '<YOUR_AZURE_OPENAI_API_KEY>', $AZURE_KEY | Set-Content $f -Encoding UTF8

# research agent
$f = "$env:USERPROFILE\.openclaw\agents\research\agent\auth-profiles.json"
(Get-Content $f -Raw) -replace '<YOUR_AZURE_OPENAI_API_KEY>', $AZURE_KEY | Set-Content $f -Encoding UTF8

# finance agent
$f = "$env:USERPROFILE\.openclaw\agents\finance\agent\auth-profiles.json"
(Get-Content $f -Raw) -replace '<YOUR_AZURE_OPENAI_API_KEY>', $AZURE_KEY | Set-Content $f -Encoding UTF8
```

### 第五步：设置用户级环境变量

```powershell
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_VERSION", "2025-04-01-preview", "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_BASE_URL", "https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai", "User")
```

> ⚠️ 设置后需**重新打开终端**才生效。

### 第六步：重新登录 OpenAI（可选）

如果需要使用 OpenAI Codex provider：

```powershell
openclaw configure
```

按提示完成 OAuth 登录，token 会自动写入 3 个 agent 的 auth-profiles.json。

### 第七步：启动 Gateway

```powershell
openclaw gateway start
```

或手动注册计划任务（开机自启）：

```powershell
$action = New-ScheduledTaskAction -Execute "openclaw" -Argument "serve --port 18789"
$trigger = New-ScheduledTaskTrigger -AtLogon
Register-ScheduledTask -TaskName "OpenClaw Gateway" -Action $action -Trigger $trigger -RunLevel Limited
schtasks /Run /TN "OpenClaw Gateway"
```

### 第八步：验证

```powershell
# 查看 agent 列表（应显示 main, research, finance）
openclaw agents list

# 查看 skill 状态（应全部 ✓ ready）
openclaw skills list

# 打开 Dashboard 测试
openclaw dashboard
```

---

## 完整一键脚本

将以下内容保存为 `migrate-openclaw.ps1`，在新电脑上运行：

```powershell
# ===== 按实际修改以下参数 =====
$OLD_USERNAME = "yanmeizhao"
$AZURE_KEY    = "在此粘贴你的Azure API Key"
$AZURE_API_VERSION = "2025-04-01-preview"
$AZURE_BASE_URL    = "https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai"
$GIT_REPO     = "https://github.com/endy-see/CodeRepo.git"
$GIT_BRANCH   = "myClaw"
# ==============================

$ErrorActionPreference = "Stop"

# 1. 拉取配置
Write-Host "[1/6] 从 GitHub 拉取配置..."
$tempDir = Join-Path $env:TEMP "openclaw-migrate-$(Get-Date -Format 'yyyyMMddHHmmss')"
git clone --branch $GIT_BRANCH --single-branch $GIT_REPO $tempDir

# 2. 复制到用户目录（跳过 identity 和 .git）
Write-Host "[2/6] 复制配置到 $env:USERPROFILE\.openclaw ..."
robocopy "$tempDir\.openclaw" "$env:USERPROFILE\.openclaw" /MIR /XD ".git" "identity" /NFL /NDL /NJH /NJS /NC /NS

# 3. 替换用户名
Write-Host "[3/6] 替换路径中的用户名: $OLD_USERNAME -> $env:USERNAME"
$config = Get-Content "$env:USERPROFILE\.openclaw\openclaw.json" -Raw
$config = $config -replace [regex]::Escape($OLD_USERNAME), $env:USERNAME
Set-Content "$env:USERPROFILE\.openclaw\openclaw.json" $config -Encoding UTF8

# 4. 填入 Azure Key
Write-Host "[4/6] 写入 Azure API Key..."
foreach ($agent in @("main", "research", "finance")) {
    $f = "$env:USERPROFILE\.openclaw\agents\$agent\agent\auth-profiles.json"
    if (Test-Path $f) {
        (Get-Content $f -Raw) -replace '<YOUR_AZURE_OPENAI_API_KEY>', $AZURE_KEY | Set-Content $f -Encoding UTF8
    }
}

# 5. 设置环境变量
Write-Host "[5/6] 设置用户级环境变量..."
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_VERSION", $AZURE_API_VERSION, "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_BASE_URL", $AZURE_BASE_URL, "User")

# 6. 启动 gateway
Write-Host "[6/6] 启动 Gateway..."
openclaw gateway start

# 清理临时目录
Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "===== 迁移完成 ====="
Write-Host "运行以下命令验证："
Write-Host "  openclaw agents list     # 应显示: main, research, finance"
Write-Host "  openclaw skills list     # 应全部 ready"
Write-Host "  openclaw dashboard       # 打开浏览器测试"
Write-Host ""
Write-Host "如需 OpenAI Codex，请运行: openclaw configure"
```

---

## 不同 Azure 资源的情况

如果新电脑使用**不同的 Azure AI Services 资源**（不同订阅/区域），除了修改上述 Azure Key 外，还需修改：

```powershell
$NEW_ENDPOINT = "https://你的资源名.cognitiveservices.azure.com/openai"

# 修改 3 个 agent 的 models.json
foreach ($agent in @("main", "research", "finance")) {
    $f = "$env:USERPROFILE\.openclaw\agents\$agent\agent\models.json"
    if (Test-Path $f) {
        $content = Get-Content $f -Raw
        $content = $content -replace [regex]::Escape("https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai"), $NEW_ENDPOINT
        Set-Content $f $content -Encoding UTF8
    }
}

# 同步更新环境变量
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_BASE_URL", $NEW_ENDPOINT, "User")
```

---

## 注意事项

1. **API Key 安全**：`auth-profiles.json` 包含明文 API Key，GitHub 仓库中已替换为占位符（`<YOUR_*>`），迁移后必须手动填入真实值。
2. **OpenAI OAuth Token** 有过期时间，迁移后需运行 `openclaw configure` 重新登录。
3. **设备身份**（`identity/` 目录）不可复制，每台电脑有唯一设备 ID 和密钥对。
4. **Skill 更新**：迁移后建议运行 `clawhub update --all` 更新所有 skill。
5. **金融数据技能**：`akshare-wrapper` 的 `main.py` 已修改为使用新浪/同花顺数据源避免东方财富反爬，迁移时 skill 代码会一并复制，无需额外操作。
6. **Session 历史**：`agents/*/sessions/` 中的历史会话也会一并迁移，如不需要可手工删除。
