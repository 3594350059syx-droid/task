"""
学伴 - 智能学习助手（实验5脚手架）
================================================
适用环境：Thonny + Python 3.8+
技术栈：Flask（一个文件搞定后端 + 前端托管）

【运行步骤】
1. 在 Thonny 顶部菜单：工具 → 管理包 → 搜索 flask → 安装
2. 在 Thonny 顶部菜单：工具 → 管理包 → 搜索 requests → 安装
3. 把下面 API_KEY 的值替换成你自己申请的 DeepSeek API Key
   （申请地址：https://platform.deepseek.com）
4. 直接在 Thonny 中点击"运行"按钮（或按 F5）
5. 看到 "Running on http://127.0.0.1:5000" 后，
   打开浏览器访问 http://127.0.0.1:5000 即可

【常见报错】
- ModuleNotFoundError: 没装包，回到第 1、2 步
- 401 Unauthorized: API Key 填错了
- Connection refused: 检查网络，部分校园网需要代理
"""

from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import os
from dotenv import load_dotenv
# ============ 配置区（学生需要修改这里）============
# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中读取 API Key
API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "GLM-4.7"

# 四种学习模式的系统提示词（学生可自由修改）
SYSTEM_PROMPTS = {
    "explain": """你是一位耐心的大学课程导师，名字叫"学伴"。
请用通俗易懂的语言为学生讲解知识点，多举生活化的例子，
讲解时遵循"是什么→为什么→怎么用"的结构。""",

    "solve": """你是一位严谨的解题导师，名字叫"学伴"。
请使用思维链（Chain of Thought）方法，一步一步推导题目的解答过程。
每一步都要说明"为什么这样做"，最后用一句话总结关键思路。""",

    "plan": """你是一位贴心的学习规划师，名字叫"学伴"。
请根据学生描述的剩余天数和薄弱环节，生成一份具体可执行的复习计划。
计划要包含每日任务、时间分配、复习重点和检验方式。""",

    "review": """你是一位细心的错题分析师，名字叫"学伴"。
请分析学生提供的错题，指出可能的知识漏洞和思维误区，
并给出 2-3 道针对性的练习题目建议。""",
}

# ============ Flask 应用 ============
app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    """访问根路径时返回前端页面"""
    return send_from_directory("static", "index.html")


@app.route("/cards/<path:filename>")
def cards(filename):
    """提供知识卡片图片访问"""
    return send_from_directory("static/cards", filename)


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    核心接口：接收前端消息，调用大模型 API，返回回复
    """
    data = request.get_json()
    mode = data.get("mode", "explain")
    user_messages = data.get("messages", [])

    # 根据模式选择系统提示词
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["explain"])

    # 拼接发送给大模型的完整消息列表
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(user_messages)

    try:
        # 调用 DeepSeek API（其他大模型 API 格式类似，改 URL 和 model 即可）
        response = requests.post(
            API_URL,
            headers={
                "Authorization": "Bearer " + API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": full_messages,
                "temperature": 0.7,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})

    except requests.exceptions.HTTPError as e:
        return jsonify({"error": "API 调用失败：" + str(e)}), 500
    except Exception as e:
        return jsonify({"error": "服务器错误：" + str(e)}), 500


# ============ 启动服务 ============
if __name__ == "__main__":
    print("=" * 50)
    print("学伴服务启动成功！")
    print("请在浏览器打开：http://127.0.0.1:5050")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
