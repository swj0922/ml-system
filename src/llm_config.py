from abc import ABC, abstractmethod
from openai import OpenAI


class BaseLLM(ABC):
    """
    大语言模型基类，定义了所有大模型应该实现的接口
    """
    
    def __init__(self, api_key, base_url=None):
        """
        初始化大模型
        
        Args:
            api_key (str): API密钥
            base_url (str, optional): API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
    
    @abstractmethod
    def get_client(self):
        """
        获取模型客户端
        
        Returns:
            object: 模型客户端实例
        """
        pass
    
    @abstractmethod
    def get_response(self, prompt, system_prompt="You are a helpful assistant."):
        """
        获取模型回复
        
        Args:
            prompt (str): 用户输入的提示
            system_prompt (str): 系统提示，默认为"You are a helpful assistant."
        
        Returns:
            str: 模型的回复内容
        """
        pass


class GeminiLLM(BaseLLM):
    """
    Gemini模型实现类
    """
    
    def __init__(self, api_key="AIzaSyD36taFUaT7sv0iKwzLyuFeqZiZPoQtSnA"):
        """
        初始化Gemini模型
        
        Args:
            api_key (str): Gemini API密钥
        """
        super().__init__(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    def get_client(self):
        """
        获取Gemini客户端
        
        Returns:
            OpenAI: 配置好的Gemini客户端实例
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_response(self, prompt, system_prompt="You are a helpful assistant."):
        """
        使用Gemini模型获取回复（流式输出）
        
        Args:
            prompt (str): 用户输入的提示
            system_prompt (str): 系统提示，默认为"You are a helpful assistant."
        
        Returns:
            str: 模型的回复内容
        """
        client = self.get_client()
        
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True  # 启用流式输出
        )

        collected_messages = []
        for chunk in response:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message:
                collected_messages.append(chunk_message)
                # 实时打印流式输出（可选）
                print(chunk_message, end='', flush=True)
        
        # 换行，因为流式输出没有换行
        print()
        
        return "".join(collected_messages)


class QwenLLM(BaseLLM):
    """
    Qwen模型实现类
    """
    
    def __init__(self, api_key="sk-1ef165b563f646a482c2a0b589fa9b09", base_url=None):
        """
        初始化Qwen模型
        
        Args:
            api_key (str): Qwen API密钥
            base_url (str, optional): Qwen API基础URL，默认使用阿里云DashScope
        """
        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        super().__init__(
            api_key=api_key,
            base_url=base_url
        )
    
    def get_client(self):
        """
        获取Qwen客户端
        
        Returns:
            OpenAI: 配置好的Qwen客户端实例
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_response(self, prompt, system_prompt="You are a helpful assistant.", model="qwen-turbo"):
        """
        使用Qwen模型获取回复
        
        Args:
            prompt (str): 用户输入的提示
            system_prompt (str): 系统提示，默认为"You are a helpful assistant."
            model (str): 使用的Qwen模型，默认为"qwen-turbo"
        
        Returns:
            str: 模型的回复内容
        """
        client = self.get_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content


# 工厂函数，用于创建不同类型的LLM实例
def create_llm(model_type, **kwargs):
    """
    创建大模型实例的工厂函数
    
    Args:
        model_type (str): 模型类型，支持"gemini"和"qwen"
        **kwargs: 传递给模型构造函数的参数
    
    Returns:
        BaseLLM: 大模型实例
    
    Raises:
        ValueError: 当模型类型不支持时抛出异常
    """
    if model_type.lower() == "gemini":
        return GeminiLLM(**kwargs)
    elif model_type.lower() == "qwen":
        return QwenLLM(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 示例用法
if __name__ == "__main__":
    # 使用Gemini模型
    gemini_llm = create_llm("gemini")
    response = gemini_llm.get_response("Explain to me how AI works")
    print("Gemini回复:", response)
    
    # 使用Qwen模型
    qwen_llm = create_llm("qwen")
    response = qwen_llm.get_response("请解释一下人工智能是如何工作的")
    print("Qwen回复:", response)