import os
import requests
# 执行步骤
# 先设置环境变量：$Env:AZURE_API_KEY = "<YOUR_AZURE_API_KEY>"
# 然后运行：python call_model.py

def chat(question: str) -> str:
    api_key = os.environ.get("AZURE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 AZURE_API_KEY")

    url = (
        "https://ym-claude-opus-dev-0316-resource.cognitiveservices.azure.com"
        "/openai/responses?api-version=2025-04-01-preview"
    )

    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": "gpt-5.1-codex-mini",
            "input": question,
            "max_output_tokens": 16384,
        },
        timeout=120,
    )
    if not resp.ok:
        print("API 错误:", resp.status_code, resp.text)
        resp.raise_for_status()

    data = resp.json()
    # 兼容 responses API 和 chat completions API 两种返回格式
    if "output" in data:
        # responses API 格式
        for item in data["output"]:
            if item.get("type") == "message":
                return "".join(
                    part["text"]
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                )
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return str(data)


if __name__ == "__main__":
    question = input("请输入你的问题: ")
    print("\n模型回答:\n")
    print(chat(question))
