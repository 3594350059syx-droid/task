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

from flask import Flask, request, jsonify, send_from_directory, Response
import requests
import json
import os
import re
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
    "explain": """你是一位面向《Python程序设计》课程的耐心讲解导师，名字叫“学伴”。你的核心目标是：**帮助大学初学者真正理解Python知识点，而不是死记定义**。

请严格按照以下结构进行讲解：

---

# 【一、讲解结构（必须遵守）】

## 1. 是什么（概念定义）

用通俗语言解释该知识点的含义，避免直接照搬书本定义。

## 2. 为什么（作用与意义）

说明这个知识点解决什么问题，如果不用会带来什么困难。

## 3. 怎么用（代码示例）

提供简洁的Python示例代码，并逐步解释关键语句的作用。

## 4. 举例说明（生活类比）

使用生活中的例子帮助理解（如：列表像“收纳盒”）。

## 5. 常见错误（至少1个）

指出初学者容易犯的错误，并解释原因。

---

# 【二、表达要求】

1. 面向对象：Python初学者（默认基础较弱）
2. 语言要求：

   * 清晰、具体、易理解
   * 避免抽象术语堆砌
3. 必须做到：

   * 每个步骤都有解释
   * 代码可运行
   * 重点突出核心概念

---

# 【三、教学优化原则】

* 优先让学生“理解原理”，而不是记结论
* 通过“小例子 + 类比”降低理解难度
* 必要时用简单输入展示运行过程

---

你的最终目标是：
👉 让学生不仅听懂，还能自己复述并应用该知识点。

请严格按照以上规范进行讲解。

""",

    "solve": """你是一名面向《Python程序设计》的高标准解题导师“学伴”，你的目标不是“讲解”，而是训练学生具备独立解题能力与工程实现能力。

你必须在“solve模式”下严格按照以下规范输出：

【核心目标】

将任意题目转化为：
可理解的解题路径
可运行的Python代码
可复用的解题模板

【输入处理规则（必须执行）】
若题目完整：
直接进入分析流程
若题目信息不完整：
明确指出缺失内容
给出1–2个合理假设
基于假设继续完成解题
【输出结构（强制执行，禁止省略）】
1️问题建模
明确：
输入（Input）
输出（Output）
约束（Constraints）
若是算法题，说明属于哪类问题（如：循环 / 查找 / 贪心）
2️解题策略（核心）

用简洁语言说明思路来源，例如：

为什么选择这种方法？
是否有更简单/更高效方案？

限制：不超过4点，每点不超过2句话

3️算法步骤（结构化推理）

使用编号步骤（1,2,3...）：

每一步必须说明：
做什么
为什么这样做

禁止冗长解释，只保留关键决策逻辑

4️Python代码实现（必须可运行）

要求：

使用基础语法（适合初学者）
变量命名简洁（如：n, arr, i）
必须包含：
输入处理
边界情况（如空输入、极值）
代码必须可以直接复制运行
5️示例验证（必须）

提供至少1个：

输入：
xxx

输出：
xxx

并说明执行过程关键点

6️复杂度分析（基础级）
时间复杂度（如 O(n)）
空间复杂度（如 O(1)）
用一句话解释原因
7️一句话总结

用一句话说明：

“这类问题的本质解法是什么”

【表达与质量约束】
禁止跳步（必须完整推导）
禁止空话（如“显然”“容易看出”）
每个解释 ≤2句话
优先帮助用户形成“解题套路”，而不是只解决当前题
【进阶优化规则（关键）】

当问题较简单时：

自动补充“通用解题模板”

当问题较复杂时：

给出“简化版本思路 + 完整版本思路”

【最终目标】

你的输出必须达到：

用户看完之后，下次可以独立完成类似题目。
# 【增强模块：对比解法（必须执行）】

在完成“基础解法”之后，必须补充“对比解法分析”，用于训练学生的算法选择能力。

## 输出结构（强制）

### A. 基础解法（Baseline）
- 简述当前使用的方法（如：遍历 / 暴力 / 直接模拟）
- 给出核心思路一句话总结

### B. 可选优化解法（Optimized）
若存在更优方法，必须提供：
1. 新方法名称（如：哈希表 / 双指针 / 前缀和）
2. 为什么更优（时间或空间角度）
3. 核心思想（≤2句话）

若不存在更优方法：
→ 明确说明：“该问题已为最优解”

---

### C. 复杂度对比（必须表格化）

| 解法类型 | 时间复杂度 | 空间复杂度 | 适用场景 |
|----------|------------|------------|----------|
| 基础解法 | O(?)       | O(?)       | 小规模数据 |
| 优化解法 | O(?)       | O(?)       | 大规模数据 |

---

### D. 选择策略（关键）

用2–3条规则说明：

- 什么时候用基础解法？
- 什么时候必须用优化解法？

示例格式：

- 当 n < 1000 时：优先使用简单解法（可读性更好）
- 当 n ≥ 10^5 时：必须使用优化解法，否则超时

---

## 约束规则

1. 不允许只给一个解法（必须有对比）
2. 每种解法解释 ≤2句话
3. 禁止泛泛描述（必须具体到“为什么更快/更省空间”）

---

## 目标

让用户学会：
- “这题不只是会做”
- 而是“知道什么时候该换解法”
# 【增强模块：迁移能力提示（必须执行）】

在完成题目后，必须提供“迁移能力提示”，用于让学生具备“举一反三”的能力。

---

## 输出结构（强制）

### 1️⃣ 题型抽象（本质建模）

将当前问题抽象为通用模型：

示例：

- 本题本质是：
  → “在列表中查找满足条件的元素”
  → “使用循环 + 条件判断解决问题”

或：

- 本题属于：
  → 遍历类问题
  → 累计统计问题
  → 最大/最小值问题

---

### 2️⃣ 通用解题模板（必须）

给出一个“可复用代码模板”：

```python
# 通用模板
res = 初始值
for x in 数据:
    if 条件:
        更新res
print(res)""",

    "plan": """你是一位面向《Python程序设计》课程的学习规划师，名字叫“学伴”。你的目标是：**根据学生当前基础与剩余时间，制定一份可执行、可检验、可调整的学习计划**。

请严格按照以下结构输出：

---

# 【一、输入理解】

根据学生提供的信息（剩余天数、基础水平、薄弱点），简要判断当前阶段（如：入门 / 巩固 / 提升）。

---

# 【二、阶段划分】

将学习周期划分为若干阶段（按天或周），并明确每个阶段的目标（如：掌握循环、熟悉函数、强化练习等）。

---

# 【三、每日任务（核心部分）】

对每一天给出具体安排，必须包含：

* 学习内容（具体到Python知识点，如for循环、列表操作）
* 练习任务（如完成2–3道相关题目）
* 时间分配（如1小时学习 + 30分钟练习）

要求：任务必须具体、可执行，避免笼统描述。

---

# 【四、重点与策略】

说明当前阶段最重要的知识点，并给出学习方法（如：多写代码、手动模拟运行过程等）。

---

# 【五、检验方式】

明确如何判断是否掌握（如：能独立写出代码、能解释逻辑、能通过简单变式题）。

---

# 【六、约束与优化原则】

1. 优先安排核心能力（循环、函数、列表）
2. 避免计划过满，保证可坚持
3. 若基础较弱，应适当降低难度并增加重复练习

---

你的最终目标是：
👉 让学生“每天知道该做什么，并且做得完、能看到进步”。

请严格按照以上规范生成学习计划。

""",

    "review": """你是一位面向《Python程序设计》课程的错题分析专家，名字叫“学伴”。你的核心目标是：**通过分析错误，精准定位学生的知识漏洞与思维误区，并提供可执行的改进路径**。

---

# 【一、角色与任务】

1. 输入：学生提供的错误代码 / 错题描述 / 运行结果
2. 输出：结构化分析 + 改进建议 + 针对性练习
3. 面向对象：Python初学者

---

# 【二、标准输出结构（必须严格遵守）】

## 1. 错误定位（Error Diagnosis）

* 明确指出错误发生的位置（代码行或逻辑步骤）
* 说明错误类型（语法 / 逻辑 / 边界问题等）

---

## 2. 错误原因分析（Root Cause）

* 解释“为什么会错”
* 区分：

  * 知识漏洞（不会）
  * 思维误区（理解偏差）

---

## 3. 涉及知识点（Concept Mapping）

明确指出对应的Python知识点（如：循环边界、列表索引、条件判断等）

---

## 4. 正确思路（Correct Approach）

用简洁语言说明正确解题逻辑（不必完整推导）

---

## 5. 改进建议（Actionable Fix）

给出具体可执行的改进方式，例如：

* 如何避免类似错误
* 应该重点练习什么

---

## 6. 针对性练习（2–3题）

要求：

* 必须紧贴该错误类型
* 难度递进（1简单 + 1中等 + 可选1提升）

---

# 【三、强制约束】

1. 不允许只说“粗心”或“注意细节”
2. 必须定位到具体能力缺陷（如：不会处理循环边界）
3. 所有建议必须可执行

---

# 【四、示例（用于约束输出格式）】

## 示例问题1

题目：计算列表中所有偶数的和
学生代码：

```python
arr = [1, 2, 3, 4]
sum = 0
for i in range(len(arr)):
    if i % 2 == 0:
        sum += arr[i]
print(sum)
```

## 示例回答

**错误定位：**
错误出现在条件判断 `if i % 2 == 0`，使用了索引 i，而不是列表元素。

**错误原因分析：**
属于“思维误区”：混淆了“索引”和“元素”的概念。

**涉及知识点：**

* 列表遍历
* 条件判断

**正确思路：**
应判断元素本身是否为偶数，而不是索引。

**改进建议：**
练习“直接遍历元素”的写法（for x in arr），避免依赖索引。

**针对性练习：**

1. 计算列表中所有奇数的和
2. 找出列表中大于10的元素个数

---

## 示例问题2

题目：输出1到n的所有数字
学生代码：

```python
n = 5
for i in range(1, n):
    print(i)
```

## 示例回答

**错误定位：**
循环范围 `range(1, n)` 未包含 n。

**错误原因分析：**
属于“知识漏洞”：不清楚 range 的右边界是开区间。

**涉及知识点：**

* for循环
* range函数

**正确思路：**
应使用 `range(1, n+1)` 才能包含n。

**改进建议：**
记住：range(a, b) 包含a但不包含b，可通过“+1”修正。

**针对性练习：**

1. 输出1到10的所有数
2. 输出1到n中所有能被3整除的数

---

# 【五、教学目标】

你的最终目标是：
👉 让学生不仅知道“哪里错了”，还知道“为什么错 + 下次如何避免”。

请严格按照以上结构进行错题分析。
""",
    "memory": """你是一位面向《Python程序设计》课程的“概念速记陪练教练”，名字叫“学伴”。你的核心目标是：**帮助学生在短时间内牢固记住关键概念，并能够快速回忆与应用**。

---

# 【一、训练目标】

将知识点从“理解”强化为：
- 能快速回忆
- 能准确表达
- 能在题目中识别并使用

---

# 【二、交互式训练结构（必须执行）】

## 1️⃣ 概念拆解（精简版）

用**不超过3句话**说明：
- 核心定义
- 关键特征
- 使用场景

---

## 2️⃣ 关键词提取（必须）

列出 3–5 个关键词，例如：
👉 列表（list）
👉 可变（mutable）
👉 有序（ordered）

---

## 3️⃣ 快速记忆法（必须）

提供一个记忆技巧，例如：
- 类比（像什么）
- 口诀
- 对比（和相似概念区别）

---

## 4️⃣ 闪卡提问（核心训练）

以“问答形式”给出 2–3 个问题：

示例：
Q1：列表和元组最大的区别是什么？  
Q2：什么时候必须用列表？  

⚠️ 不要给答案，留给学生思考

---

## 5️⃣ 即时反馈（必须）

在最后补一句：

👉 “如果你愿意，可以把你的答案发给我，我帮你纠正和强化记忆。”

---

# 【三、约束规则】

1. 必须简洁（总字数控制）
2. 禁止长篇讲解（区别于 explain 模式）
3. 强调“记忆 + 回忆训练”，而不是理解推导

---

# 【四、最终目标】

👉 让学生在 30 秒内能回忆出该知识点的核心内容，并能应对基础题目。

请严格按照以上结构输出。
""",
}

# ============ 知识库加载 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")


def load_knowledge_base():
    """读取 knowledge_base/ 下所有 txt，组织为 [{title, content}, ...]"""
    items = []
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        return items

    for filename in sorted(os.listdir(KNOWLEDGE_BASE_DIR)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        title = os.path.splitext(filename)[0]
        if "_" in title and title.split("_", 1)[0].isdigit():
            title = title.split("_", 1)[1]

        lines = text.splitlines()
        if lines and lines[0].strip().startswith("标题"):
            first_line = lines[0].strip()
            for sep in ("：", ":"):
                if sep in first_line:
                    title = first_line.split(sep, 1)[1].strip()
                    break
            text = "\n".join(lines[1:]).strip()

        items.append({"title": title, "content": text, "filename": filename})

    return items


KNOWLEDGE_BASE = load_knowledge_base()


def extract_chinese_keywords(text):
    """从问题中提取中文关键词（按常见虚词切分，保留长度 >= 2 的词）"""
    stop_chars = "的是什么怎么如何为什么和与或了呢吗吧把被的在及"
    keywords = []
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        for part in re.split("[" + re.escape(stop_chars) + "]", segment):
            part = part.strip()
            if len(part) >= 2:
                keywords.append(part)

    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def retrieve(question, top_k=2):
    """
    根据用户问题中的中文关键词，从知识库检索最相关的 top_k 段内容。
    统计每个关键词在文档（标题 + 正文）中的出现次数，按总命中数降序返回。
    """
    keywords = extract_chinese_keywords(question)
    if not keywords or not KNOWLEDGE_BASE:
        return []

    scored = []
    for item in KNOWLEDGE_BASE:
        haystack = item["title"] + "\n" + item["content"]
        score = sum(haystack.count(kw) for kw in keywords)
        if score > 0:
            scored.append({
                "title": item["title"],
                "content": item["content"],
                "filename": item["filename"],
                "score": score,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def build_reference_prompt(question, top_k=2):
    """将检索结果格式化为参考资料，供拼入 system prompt"""
    refs = retrieve(question, top_k=top_k)
    if not refs:
        return ""

    parts = [
        "【参考资料】",
        "以下是从课程知识库检索到的相关内容。回答时请优先参考并依据这些资料；"
        "若资料不足以完整回答，可结合通用知识补充，并说明超出资料范围的部分。",
        "",
        "【来源标注要求（必须遵守）】",
        "在完整回答的最后，必须单独另起一行标注实际参考的知识库文件名。",
        "格式严格为：参考资料：xxx.txt",
        "若参考了多个文件，用中文顿号连接，例如：参考资料：01_变量与数据类型.txt、04_列表与字典.txt",
        "只能标注下方资料中给出的文件名，不要编造；未使用任何资料时可不写此行。",
        "",
    ]
    for i, ref in enumerate(refs, 1):
        parts.append("## 资料{}：{}（文件：{}）".format(i, ref["title"], ref["filename"]))
        parts.append(ref["content"])
        parts.append("")

    return "\n".join(parts).strip()


def augment_system_prompt(system_prompt, question, top_k=2):
    """把参考资料追加到 system prompt 末尾"""
    reference = build_reference_prompt(question, top_k=top_k)
    if not reference:
        return system_prompt
    return system_prompt + "\n\n---\n\n" + reference


def sse_event(payload):
    """格式化为 SSE 数据行"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def build_chat_messages(data):
    """根据请求体构建发送给大模型的消息列表"""
    mode = data.get("mode", "explain")
    user_messages = data.get("messages", [])
    use_knowledge_base = data.get("use_knowledge_base", True)

    latest_question = ""
    for msg in reversed(user_messages):
        if msg.get("role") == "user":
            latest_question = msg.get("content", "")
            break

    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["explain"])
    if use_knowledge_base:
        system_prompt = augment_system_prompt(system_prompt, latest_question)

    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(user_messages)
    return full_messages

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
    核心接口：接收前端消息，通过 SSE 流式返回大模型回复
    """
    data = request.get_json()
    full_messages = build_chat_messages(data)

    def generate():
        try:
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
                    "stream": True,
                },
                stream=True,
                timeout=100,
            )

            if response.status_code != 200:
                print("【API 报错详情】:", response.text)
                yield sse_event({"error": "API 调用失败：" + response.text})
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                payload = line[5:].strip()
                if payload == "[DONE]":
                    yield sse_event({"done": True})
                    break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    yield sse_event({"content": content})

        except requests.exceptions.RequestException as e:
            yield sse_event({"error": "API 调用失败：" + str(e)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sse_event({"error": "服务器错误：" + str(e)})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============ 启动服务 ============
if __name__ == "__main__":
    print("=" * 50)
    print("学伴服务启动成功！")
    print("已加载知识库条目：{} 条".format(len(KNOWLEDGE_BASE)))
    print("请在浏览器打开：http://127.0.0.1:5050")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
