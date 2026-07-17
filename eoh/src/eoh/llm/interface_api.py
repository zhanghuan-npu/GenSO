"""
InterfaceAPI 类 — LLM API 接口封装

本模块定义了 `InterfaceAPI` 类，用于与远程大语言模型 (LLM) API 进行交互，
提供发送提示语 (prompt) 并获取生成文本响应的功能。通过封装 HTTP 请求和
错误重试机制，方便上层算法调用 LLM 而无需关心底层网络细节。

功能说明
--------
1. 初始化 (__init__)：
   - `api_endpoint` : LLM 服务的主机地址或域名。
   - `api_key`      : 用于认证的 API Key。
   - `model_LLM`    : 指定使用的 LLM 模型类型，如 "deepseek-chat"。
   - `debug_mode`   : 是否开启调试模式，打印错误信息。
   - `n_trial`      : 最大重试次数（默认 5 次）。

2. 获取响应 (get_response)：
   - 输入：`prompt_content` (字符串)，即用户希望 LLM 回答的提示语。
   - 处理：
     1. 构建 JSON 请求体，包含模型和对话信息。
     2. 设置请求头，包括 Authorization、Content-Type 等。
     3. 循环发送 POST 请求，最多尝试 `n_trial` 次：
        - 使用 HTTPS 连接向 API 发送请求。
        - 解析返回的 JSON 数据，提取生成文本内容。
        - 如果请求失败，会根据 `debug_mode` 打印错误并重试。
   - 输出：返回 LLM 生成的文本响应字符串，如果超过重试次数仍失败，则返回 None。
"""

import http.client
import json
import socket

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

        res = self.get_response("1+1=?")
        if res == None:
             print(">> Error in LLM API, wrong endpoint, key, model or local deployment!")
             exit()

    def get_response(self, user_prompt, system_prompt=None, temperature=1.0):

        # 建立 messages
        messages = []
        if system_prompt:  # 只有当传入时才加 system 提示
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # 建立json格式数据
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": messages,
                "temperature": temperature,
                "thinking": {"type": "disabled"}
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }
        
        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint, timeout=300)
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break

            except (socket.timeout, TimeoutError):
                # 超时处理：连接建立或等待返回超过300秒
                if self.debug_mode:
                    print(f"LLM no response within 300 s. Retrying ({n_trial}/{self.n_trial})...")
                continue

            except Exception as e:
                # 其他错误（网络、解析等）
                if self.debug_mode:
                    print(f"Error in API ({n_trial}/{self.n_trial}): {e}")
                continue

        return response