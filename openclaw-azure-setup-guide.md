# OpenClaw + Azure AI Services 模型配置指南

> 本文档记录如何将部署在 Azure AI Services 上的模型配置到 OpenClaw，以及解决过程中遇到的各种坑。

## 前置条件

- Windows 系统，已安装 Node.js (v22+)
- 已通过 npm 全局安装 OpenClaw：`npm install -g openclaw`
- 已在 Azure 上部署模型（本例为 `gpt-5.1-codex-mini`）
- 已获得 Azure API Key

## 你的 Azure 模型信息（按实际替换）

| 项目 | 值 |
|---|---|
| Azure 资源名 | `ym-claude-opus-dev-0316-resource` |
| 端点域名 | `ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com` |
| API 路径 | `/openai/responses?api-version=2025-04-01-preview` |
| 模型 ID | `gpt-5.1-codex-mini` |
| API Key | `你的Azure API Key` |

---

## 第一步：首次运行 OpenClaw（初始化配置）

```powershell
openclaw
```

按照引导向导完成初始设置。这会生成以下配置文件：

- `~/.openclaw/openclaw.json` — 主配置
- `~/.openclaw/agents/main/agent/models.json` — 模型提供者配置
- `~/.openclaw/agents/main/agent/auth-profiles.json` — 认证凭据

## 第二步：配置模型提供者

编辑 `~/.openclaw/agents/main/agent/models.json`，添加 `azure-openai-responses` 提供者：

```json
{
  "providers": {
    "azure-openai-responses": {
      "baseUrl": "https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai",
      "api": "azure-openai-responses",
      "apiVersion": "2025-04-01-preview",
      "models": [
        {
          "id": "gpt-5.1-codex-mini",
          "name": "gpt-5.1-codex-mini",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 200000,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

**关键说明：**
- `baseUrl` 只写到 `/openai`，不要加 `/v1` 也不要加 `/responses`
- `api` 必须是 `azure-openai-responses`（使用 Azure 专用 SDK，会自动添加 `?api-version=` 参数）
- `apiVersion` 必须和你 Azure 端点实际支持的版本一致（如 `2025-04-01-preview`）

## 第三步：配置 API Key 认证

编辑 `~/.openclaw/agents/main/agent/auth-profiles.json`，添加 Azure 认证配置：

```json
{
  "version": 1,
  "profiles": {
    "azure-openai-responses:default": {
      "type": "token",
      "provider": "azure-openai-responses",
      "token": "你的Azure API Key"
    }
  }
}
```

## 第四步：设置默认模型

编辑 `~/.openclaw/openclaw.json`，在 `agents.defaults` 中设置默认模型：

```json5
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "azure-openai-responses/gpt-5.1-codex-mini"
      },
      "models": {
        "azure-openai-responses/gpt-5.1-codex-mini": {}
      }
    }
  }
}
```

## 第五步：设置用户级环境变量（关键！）

OpenClaw gateway 以 **Windows 计划任务** 运行，终端里的 `$Env:` 变量对它不可见。必须设置**用户级**永久环境变量：

```powershell
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_VERSION", "2025-04-01-preview", "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_BASE_URL", "https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com/openai", "User")
```

> 也可以在 "系统属性 → 环境变量" 中手动添加。

## 第六步：禁用不需要的渠道（可选）

如果没有配置 Telegram bot，在 `~/.openclaw/openclaw.json` 中禁用，避免刷屏报错：

```json5
{
  "channels": {
    "telegram": {
      "enabled": false  // 改为 false
    }
  }
}
```

## 第七步：启动 Gateway

```powershell
openclaw gateway start
```

然后打开 Dashboard：

```powershell
openclaw dashboard
```

在 Chat 界面发消息测试。

---

## 常见问题排查

### 问题 1：`Azure OpenAI base URL is required`

**原因：** SDK 读不到 Azure 端点 URL。

**解决：** 设置用户级环境变量（第五步），或检查 models.json 中 `baseUrl` 配置。

### 问题 2：`HTTP 404: Resource not found`

可能原因有多个，按顺序排查：

#### 2a. API Version 不对

OpenClaw 的 Azure SDK 默认 API version 是 `v1`，但 Azure AI Services 端点需要 `2025-04-01-preview`。

**解决：** 在 models.json 中指定 `"apiVersion": "2025-04-01-preview"`，并设置环境变量 `AZURE_OPENAI_API_VERSION`。

#### 2b. 旧进程未被杀掉

`openclaw gateway stop` + `start` 只是操作 Windows 计划任务，旧 node 进程可能仍在运行并占用端口，使用的还是旧配置。

**解决：**

```powershell
# 停止计划任务
openclaw gateway stop

# 等 3 秒
Start-Sleep -Seconds 3

# 杀掉所有残留的 openclaw node 进程
Get-Process -Name "node" | Where-Object { $_.CommandLine -match 'openclaw' } | Stop-Process -Force

# 重新启动
openclaw gateway start
```

#### 2c. 端点类型不匹配

Azure 有两种端点格式：
- **Azure OpenAI**（传统）：`https://{resource}.openai.azure.com` — URL 路径包含 `/deployments/{name}/`
- **Azure AI Services**（统一）：`https://{resource}.cognitiveservices.azure.com` — URL 路径直接是 `/openai/responses`

本例使用的是 Azure AI Services 统一端点。幸运的是 `AzureOpenAI` SDK 的 `/responses` 调用**不会**自动拼接 `/deployments/` 路径，所以这两种端点都兼容。

### 问题 3：Dashboard 打不开（`ERR_CONNECTION_REFUSED`）

**原因：** Gateway 没有运行。

**解决：** 先 `openclaw gateway start`，再 `openclaw dashboard`。

### 问题 4：Telegram 不停报 404 错误

**原因：** Telegram bot token 配置无效（比如填了 `"no"`）。

**解决：** 在 `openclaw.json` 中设置 `channels.telegram.enabled = false`。

---

## 快速验证 Azure API 是否可用

在配置 OpenClaw 之前，先用 PowerShell 验证 Azure API 是否能通：

```powershell
$apiKey = "你的Azure API Key"
$url = "https://你的资源名.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"
$headers = @{ "api-key" = $apiKey; "Content-Type" = "application/json" }
$body = '{"model":"gpt-5.1-codex-mini","input":"hello","max_output_tokens":100}'

Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $body
```

如果返回正常的 JSON 响应，说明 Azure 端没问题，可以继续配置 OpenClaw。

---

## 配置文件总览

```
~/.openclaw/
├── openclaw.json                          # 主配置（默认模型、渠道、gateway设置）
├── agents/
│   └── main/
│       └── agent/
│           ├── models.json                # 模型提供者配置（baseUrl、api、模型列表）
│           └── auth-profiles.json         # 认证凭据（API Key）
```

## 一键配置脚本（在新电脑上使用）

将以下内容保存为 `setup-openclaw-azure.ps1`，按需修改变量后运行：

```powershell
# ===== 按实际修改以下变量 =====
$AZURE_RESOURCE = "ym-claude-opus-dev-0316-resource"
$AZURE_DOMAIN = "$AZURE_RESOURCE.cognitiveservices.azure.com"
$AZURE_API_KEY = "你的Azure API Key"
$AZURE_API_VERSION = "2025-04-01-preview"
$MODEL_ID = "gpt-5.1-codex-mini"
# ==============================

# 1. 设置用户级环境变量
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_VERSION", $AZURE_API_VERSION, "User")
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_BASE_URL", "https://$AZURE_DOMAIN/openai", "User")

# 2. 确保目录存在
$agentDir = "$env:USERPROFILE\.openclaw\agents\main\agent"
New-Item -ItemType Directory -Path $agentDir -Force | Out-Null

# 3. 写入 models.json
@"
{
  "providers": {
    "azure-openai-responses": {
      "baseUrl": "https://$AZURE_DOMAIN/openai",
      "api": "azure-openai-responses",
      "apiVersion": "$AZURE_API_VERSION",
      "models": [
        {
          "id": "$MODEL_ID",
          "name": "$MODEL_ID",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 200000,
          "maxTokens": 16384
        }
      ]
    }
  }
}
"@ | Set-Content "$agentDir\models.json" -Encoding UTF8

# 4. 写入 auth-profiles.json
@"
{
  "version": 1,
  "profiles": {
    "azure-openai-responses:default": {
      "type": "token",
      "provider": "azure-openai-responses",
      "token": "$AZURE_API_KEY"
    }
  }
}
"@ | Set-Content "$agentDir\auth-profiles.json" -Encoding UTF8

Write-Host "配置完成！请运行以下命令："
Write-Host "  openclaw          # 首次运行，完成初始化向导"
Write-Host "  openclaw gateway start"
Write-Host "  openclaw dashboard"
```

> **注意：** 首次在新电脑上使用时，需要先运行一次 `openclaw` 完成初始化向导，生成 `openclaw.json` 后再运行脚本覆盖 models.json 和 auth-profiles.json。

---

## Skill 与 Agent 安装指南

### Skill 概述

Skill 是一个包含 `SKILL.md` 的目录，用来教 agent 如何使用工具或执行特定工作流。OpenClaw 从三个位置加载 skill（优先级从高到低）：

| 位置 | 路径 | 说明 |
|---|---|---|
| Workspace skills | `<workspace>/skills/` | 当前 agent 专用，优先级最高 |
| Managed/local skills | `~/.openclaw/skills/` | 所有 agent 共享 |
| Bundled skills | OpenClaw 安装目录内 | 随 OpenClaw 自带，优先级最低 |

### 安装 Skill 的两种方式

#### 方式一：通过 ClawHub CLI 安装（推荐）

ClawHub 是 OpenClaw 的公共 skill 仓库，浏览地址：https://clawhub.ai/skills

```powershell
# 1. 安装 clawhub CLI（首次需要）
npm install -g clawhub

# 2. 安装 skill
clawhub install <skill-slug>

# 3. 更新所有已安装的 skill
clawhub update --all

# 4. 同步（扫描 + 发布更新）
clawhub sync --all
```

默认安装到当前工作目录的 `./skills/`，或 OpenClaw 配置的 workspace 目录下。

#### 方式二：手动安装（Git Clone）

直接将 skill 仓库克隆到 `~/.openclaw/skills/` 目录：

```powershell
git clone https://github.com/<作者>/<skill名>.git ~/.openclaw/skills/<skill名>
```

### 安装后的通用步骤

1. **验证 skill 是否被识别：**
   ```powershell
   openclaw skills list          # 列出所有 skill
   openclaw skills check         # 检查 skill 是否就绪
   openclaw skills info <name>   # 查看某个 skill 详情
   ```

2. **重启 gateway 使新 skill 生效：**
   ```powershell
   openclaw gateway stop
   Start-Sleep -Seconds 3
   Get-Process -Name "node" | Where-Object { $_.CommandLine -match 'openclaw' } | Stop-Process -Force -ErrorAction SilentlyContinue
   Start-Sleep -Seconds 2
   openclaw gateway start
   ```

3. **在 dashboard 中开一个新 session**（旧 session 不会加载新 skill）。

### Skill 配置（可选）

在 `~/.openclaw/openclaw.json` 中可以启用/禁用 skill 或传入环境变量：

```json5
{
  "skills": {
    "entries": {
      "skill-name": {
        "enabled": true,
        "apiKey": "如果需要的话",
        "env": {
          "SOME_ENV_VAR": "value"
        }
      }
    }
  }
}
```

---

## 实例：安装 self-improving-agent Skill

> 来源：https://clawhub.ai/pskoett/self-improving-agent
>
> 功能：自动捕获 agent 的错误、用户纠正和学习记录，实现持续改进。

### 步骤 1：安装 Skill

```powershell
# 方式一：通过 ClawHub
clawhub install self-improving-agent

# 方式二：手动克隆
git clone https://github.com/peterskoett/self-improving-agent.git ~/.openclaw/skills/self-improving-agent
```

### 步骤 2：创建 learnings 目录和文件

在 OpenClaw workspace 下创建 `.learnings/` 目录和三个日志文件：

```powershell
# 创建目录
New-Item -ItemType Directory -Path "$env:USERPROFILE\.openclaw\workspace\.learnings" -Force

# 创建日志文件
Set-Content "$env:USERPROFILE\.openclaw\workspace\.learnings\LEARNINGS.md" "# Learnings`n" -Encoding UTF8
Set-Content "$env:USERPROFILE\.openclaw\workspace\.learnings\ERRORS.md" "# Errors`n" -Encoding UTF8
Set-Content "$env:USERPROFILE\.openclaw\workspace\.learnings\FEATURE_REQUESTS.md" "# Feature Requests`n" -Encoding UTF8
```

三个文件的用途：

| 文件 | 记录内容 |
|---|---|
| `LEARNINGS.md` | 纠正、知识盲区、最佳实践 |
| `ERRORS.md` | 命令失败、异常、错误信息 |
| `FEATURE_REQUESTS.md` | 用户请求的新功能 |

### 步骤 3：验证安装

```powershell
openclaw skills list | Select-String "self"
# 应看到 "✓ ready" 状态
```

### 步骤 4：重启 Gateway

```powershell
openclaw gateway stop
Start-Sleep -Seconds 3
Get-Process -Name "node" | Where-Object { $_.CommandLine -match 'openclaw' } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
openclaw gateway start
```

### 步骤 5：使用

打开 dashboard 开启一个**新 session**，agent 会自动：
- 检测命令/操作失败 → 记录到 `ERRORS.md`
- 检测用户纠正 → 记录到 `LEARNINGS.md`
- 检测功能需求 → 记录到 `FEATURE_REQUESTS.md`

当某个 learning 被验证为普遍适用时，会被提升（promote）到以下文件：

| 目标文件 | 适用内容 |
|---|---|
| `AGENTS.md` | 工作流改进、委派模式 |
| `SOUL.md` | 行为准则、沟通风格 |
| `TOOLS.md` | 工具使用技巧、集成注意事项 |

### 可选：启用 Hook（自动提醒）

```powershell
# 复制 hook 到 OpenClaw hooks 目录
Copy-Item -Recurse "$env:USERPROFILE\.openclaw\skills\self-improving-agent\hooks\openclaw" "$env:USERPROFILE\.openclaw\hooks\self-improvement"

# 启用 hook
openclaw hooks enable self-improvement
```

### Workspace 目录结构（安装后）

```
~/.openclaw/workspace/
├── AGENTS.md              # 多 agent 工作流、委派模式
├── SOUL.md                # 行为准则、个性、原则
├── TOOLS.md               # 工具能力、集成注意事项
├── MEMORY.md              # 长期记忆
├── memory/                # 每日记忆文件
│   └── YYYY-MM-DD.md
└── .learnings/            # self-improving-agent 的日志文件
    ├── LEARNINGS.md
    ├── ERRORS.md
    └── FEATURE_REQUESTS.md
```

---

## Agent 管理指南

### Agent 与 Skill 的区别

| 概念 | 说明 | 管理方式 |
|---|---|---|
| **Agent** | 独立的 AI 人格/大脑，拥有自己的工作区、会话、认证凭据、模型配置和路由规则 | `openclaw agents` CLI |
| **Skill** | 能力包（一个含 `SKILL.md` 的文件夹），教 agent 如何使用某个工具或执行某种工作流 | `clawhub` CLI + `openclaw skills` CLI |

> **重要：** ClawHub 只有 Skill 商店，没有 Agent 商店。Agent 通过 `openclaw agents add` 在本地创建，然后给它安装 Skill、配置人格。

### Agent CLI 命令速查

```powershell
# 列出所有 agent
openclaw agents list
openclaw agents list --json      # JSON 格式输出

# 创建新 agent（交互式）
openclaw agents add <name>

# 创建新 agent（非交互式，需指定 workspace）
openclaw agents add <name> --non-interactive --workspace <dir>

# 指定模型创建
openclaw agents add <name> --workspace <dir> --model <provider/model-id>

# 设置 agent 身份（名字/emoji/头像）
openclaw agents set-identity <name>

# 路由绑定（将某个 channel 路由到指定 agent）
openclaw agents bind <name> --bind <channel[:accountId]>
openclaw agents unbind <name>
openclaw agents bindings

# 删除 agent
openclaw agents delete <name>
```

### Agent 目录结构

```
~/.openclaw/
├── openclaw.json                          # 主配置
├── agents/
│   ├── main/                              # 默认 agent
│   │   ├── agent/
│   │   │   ├── auth-profiles.json         # 认证凭据
│   │   │   └── models.json               # 模型配置
│   │   └── sessions/                      # 会话记录
│   ├── research/                          # 自建 agent 示例
│   │   ├── agent/
│   │   │   ├── auth-profiles.json
│   │   │   └── models.json
│   │   └── sessions/
│   └── <agentId>/                         # 更多 agent...
├── workspace/                             # main agent 的工作区
├── workspace-research/                    # research agent 的工作区
│   ├── AGENTS.md                          # 操作指令 + 记忆
│   ├── SOUL.md                            # 人格、边界、风格
│   ├── TOOLS.md                           # 工具使用笔记
│   ├── USER.md                            # 用户档案
│   ├── IDENTITY.md                        # 名称/emoji/头像
│   └── skills/                            # 该 agent 专属 skill
└── skills/                                # 全局共享 skill
```

---

## 实例：创建 Scientific Research Agent

> 目标：创建一个专注于科学研究的 agent，具备文献调研、假说生成、实验设计等能力。

### 步骤 1：创建 Agent

```powershell
openclaw agents add research --non-interactive --workspace "$env:USERPROFILE\.openclaw\workspace-research"
```

输出确认：
```
Agent: research
Workspace: ~\.openclaw\workspace-research
Agent dir: ~\.openclaw\agents\research\agent
```

### 步骤 2：复制模型和认证配置

新 agent 默认没有 `models.json` 和 `auth-profiles.json`，需要从 main agent 复制（或自行创建）：

```powershell
# 创建 agent 配置目录
New-Item -ItemType Directory -Path "$env:USERPROFILE\.openclaw\agents\research\agent" -Force | Out-Null

# 复制 main agent 的模型和认证配置
Copy-Item "$env:USERPROFILE\.openclaw\agents\main\agent\models.json" `
          "$env:USERPROFILE\.openclaw\agents\research\agent\models.json"
Copy-Item "$env:USERPROFILE\.openclaw\agents\main\agent\auth-profiles.json" `
          "$env:USERPROFILE\.openclaw\agents\research\agent\auth-profiles.json"
```

> 如果想让 research agent 使用不同的模型，可以单独编辑它的 `models.json`。

### 步骤 3：安装科研 Skill（Scientify）

```powershell
# 安装 Scientify（科研能力包）
clawhub install install-scientify --force --no-input
```

> Scientify 提供的能力：research-pipeline（研究管线）、literature-survey（文献调研）、idea-generation（假说生成）、arxiv 工具等。

验证安装：
```powershell
clawhub list
# 应看到 install-scientify 及版本号
```

### 步骤 4：配置科研人格

编辑 `~/.openclaw/workspace-research/SOUL.md`：

```markdown
# SOUL.md - Scientific Research Agent

_You are a dedicated scientific research assistant._

## Core Identity

**You are a rigorous, methodical research partner.** Your purpose is to help with
literature review, hypothesis generation, experimental design, data analysis, and
scientific writing.

**Be precise and evidence-based.** Always cite sources when possible. Distinguish
between established facts, emerging evidence, and speculation.

**Think critically.** Question assumptions, identify gaps in reasoning, consider
alternative explanations.

**Respect intellectual honesty.** If you don't know something, say so. Never
fabricate citations or data.

## Research Capabilities

- Literature survey and systematic review
- Hypothesis generation and refinement
- Experimental design critique
- Statistical analysis guidance
- Scientific writing and paper drafting
- Research pipeline management (via Scientify skill)
- ArXiv paper search and analysis

## Vibe

Scholarly but approachable. Precise but not pedantic.
```

编辑 `~/.openclaw/workspace-research/IDENTITY.md`：

```markdown
# IDENTITY.md - Scientific Research Agent

- **Name:** Researcher
- **Creature:** AI Research Partner
- **Vibe:** Scholarly, precise, curious, thorough
- **Emoji:** 🔬
```

### 步骤 5：重启 Gateway

```powershell
openclaw gateway stop
Start-Sleep -Seconds 3
Get-Process -Name "node" | Where-Object { $_.CommandLine -match 'openclaw' } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
openclaw gateway start
```

### 步骤 6：验证

```powershell
openclaw agents list
```

应看到：
```
- main (default)
  Model: azure-openai-responses/gpt-5.1-codex-mini
  Routing: default (no explicit rules)
- research
  Identity: 🔬 Researcher (IDENTITY.md)
  Model: azure-openai-responses/gpt-5.1-codex-mini
```

### 步骤 7：使用 Research Agent

**方式一：通过 Dashboard**

打开 Dashboard，切换到 `research` agent 进行对话。

**方式二：通过 CLI**

```powershell
openclaw agent --agent research --message "请帮我调研 transformer 在蛋白质结构预测中的最新应用"
```

**方式三：绑定到特定 Channel**

```powershell
# 将 Discord 的消息路由到 research agent
openclaw agents bind research --bind discord

# 查看所有路由绑定
openclaw agents bindings
```

### Agent 管理常用操作

```powershell
# 查看 agent 详情（JSON 格式）
openclaw agents list --json

# 修改 agent 身份
openclaw agents set-identity research

# 解除路由绑定
openclaw agents unbind research

# 删除 agent（会清理工作区和配置）
openclaw agents delete research
```
