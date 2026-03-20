---
name: finance-official-research
description: Analyze stocks, ETFs, mutual funds, listed companies, financial statements, disclosures, index constituents, and investment questions using official or primary-source data only. Use when the user asks for stock analysis, fund analysis, company fundamentals, valuation inputs, portfolio holdings, earnings interpretation, exchange filings, prospectus/report review, or investment due diligence and requires authoritative, citation-backed answers verified against official disclosure sites before answering.
---

# Finance Official Research

## Overview

Use this skill for investment research that must be grounded in official disclosures, primary filings, fund reports, exchange data, regulator notices, and issuer-published documents. Do not present unverified market lore, forum claims, or memory-based numbers as fact.

## Hard Rules

- Treat this as a **high-stakes factual workflow**.
- Use **official / primary sources first**. If unavailable, say so clearly.
- Do **not** invent prices, holdings, AUM, revenue, valuation multiples, macro figures, dates, or ticker metadata.
- Separate:
  - **Fact**: directly supported by cited source
  - **Inference**: reasoned interpretation from facts
  - **Opinion / scenario**: explicitly framed as uncertain
- For any time-sensitive number, include:
  - metric name
  - value
  - currency / unit
  - as-of date or reporting period
  - source name
- If latest official data cannot be confirmed, say: **“未确认到官方最新数据，以下不作为确定性结论。”**
- Never claim something is “cheap”, “safe”, “undervalued”, or “best” without showing the inputs and their dates.
- Avoid direct personalized investment advice. Provide research and risk framing, not guarantees.

## Source Priority

Use sources in this order.

### China equities / listed companies

1. Exchange filings and issuer announcements:
   - SSE
   - SZSE
   - BSE
   - CNINFO / 巨潮资讯
2. Company investor-relations site / annual report / interim report / quarterly report
3. CSRC notices and official rule interpretations
4. CSI index provider materials when index membership or methodology matters

### China public funds / ETFs

1. Fund manager official site
2. Fund prospectus, KIID/summary, annual/interim/quarterly reports
3. Exchange disclosures for listed ETFs / LOFs
4. AMAC / CSRC / exchange notices

### US / global equities and funds

1. SEC EDGAR
2. Issuer IR site and filed reports
3. Official fund sponsor site and statutory filings
4. Official exchange / index provider pages

### Never use as primary evidence

- social media posts
- broker marketing copy
- AI summaries without source links
- news rewrites when the original filing is available
- stale cached numbers with no reporting date

## Workflow

### 1. Identify the instrument precisely

Resolve ambiguity before analysis:

- full name
- ticker / code
- exchange / market
- fund type (ETF, mutual fund, LOF, index fund, active fund)
- share class / currency if relevant

If ambiguous, ask one narrow clarification question.

### 2. Decide the analysis type

Common modes:

- **Company fundamentals**: revenue, profit, cash flow, debt, margins, segment mix
- **Fund analysis**: objective, benchmark, holdings, industry exposure, fees, turnover, tracking error
- **Valuation support**: inputs only when sourced and dated
- **Event review**: earnings, dividend, buyback, secondary offering, risk warning, regulatory notice
- **Disclosure review**: annual report, prospectus, announcement, fund report

### 3. Gather official evidence first

For each core claim, capture the minimal source set needed.

Examples:

- Revenue / net profit → annual or interim report
- Shares outstanding / buyback / dividend → exchange announcement
- Top holdings / sector allocation / bond duration → latest fund report or fund factsheet
- Benchmark / fee / investment scope → prospectus / fund contract / official factsheet

Prefer fewer, higher-quality sources over many weak ones.

### 4. Check freshness

Before answering, verify:

- reporting period
- publication date
- whether a newer report may exist
- whether the metric is point-in-time or period-based

If using multiple periods, label them clearly.

### 5. Analyze conservatively

Only infer what the data supports. Good examples:

- “净利润增长快于营收，可能说明利润率改善。”
- “基金前十大持仓集中度较高，意味着主题暴露更强。”

Bad examples:

- “这只基金一定会跑赢。”
- “这家公司绝对被低估。”

### 6. Produce a citation-backed answer

Default answer structure:

1. **结论摘要** — 2-5 bullets
2. **官方数据与依据** — metric + value + date + source
3. **分析解读** — what the facts may imply
4. **风险与不确定性** — stale data, disclosure lag, market sensitivity
5. **需要继续核验的点** — if any

## Output Standard

When the user asks an investment question, prefer this compact template:

### 结论
- 1-3 bullets, factual first

### 官方依据
- 指标 / 事实: ...
- 数值 / 内容: ...
- 时点: ...
- 来源: ...

### 分析
- Explain only from cited facts

### 风险提示
- reporting lag / valuation sensitivity / concentration / policy / liquidity / FX / credit / duration as relevant

If multiple official documents are compared, add a short comparison table in plain text bullets rather than a markdown table when chat formatting is uncertain.

## Specific Guidance

### Stock / company analysis

Check at least these before making any strong statement:

- latest annual / interim / quarterly report available
- major recent announcements
- audit opinion if annual report
- debt, cash, cash flow, receivables/inventory where relevant
- share issuance, buyback, pledge, dividend, litigation, delisting or risk-warning items if material

### Fund analysis

Check at least these first:

- fund objective and benchmark
- management fee / custody fee if relevant
- latest disclosed holdings and concentration
- industry / asset allocation
- fund size / shares where disclosed
- tracking method and error notes for passive products
- manager changes or contract amendments if material

### Valuation questions

Only use valuation inputs that are clearly sourced and dated. State assumptions separately from facts.

### “Can I buy / should I sell?” questions

Do not give deterministic advice. Reframe into:

- what official data says now
- what key variables matter next
- what would confirm / weaken the thesis

## Failure Modes To Avoid

- Mixing different reporting periods without labels
- Quoting secondary websites as if they are official
- Treating old fund holdings as current positions
- Confusing ETF secondary-market premium/discount with NAV trend
- Using unaudited / audited figures interchangeably without saying so
- Presenting incomplete evidence as certainty

## References

Read these as needed:

- `references/source-map.md` for official-source checklists by market and instrument
- `references/answer-template.md` for a stricter response format for high-stakes questions
