from abc import ABC, abstractmethod
from openai import OpenAI
from google import genai
from google.genai import types


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
    
    @abstractmethod
    def get_streaming_response(self, prompt, system_prompt="You are a helpful assistant."):
        """
        获取模型流式回复
        
        Args:
            prompt (str): 用户输入的提示
            system_prompt (str): 系统提示，默认为"You are a helpful assistant."
        
        Yields:
            str: 模型的流式回复内容片段
        """
        pass


class GeminiLLM(BaseLLM):
    """
    Gemini模型实现类
    """
    def __init__(self, api_key):
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
            genai.Client: 配置好的Gemini客户端实例
        """
        return genai.Client(api_key=self.api_key)
    
    def get_response(self, prompt, system_prompt="You are a helpful assistant."):
        """
        使用Gemini模型获取回复（非流式）
        
        Args:
            prompt (str): 用户输入的提示
            system_prompt (str): 系统提示，默认为"You are a helpful assistant."
        
        Returns:
            str: 模型的回复内容
        """
        client = self.get_client()
        
        # 将系统提示和用户提示合并
        combined_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[combined_prompt],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
        )
        
        return response.text
    
    def get_streaming_response(self, prompt, system_prompt="You are a helpful assistant."):
        """
        使用Gemini模型获取流式回复
        
        Args:
            prompt (str): 用户输入的提示
            system_prompt (str): 系统提示，默认为"You are a helpful assistant."
        
        Yields:
            str: 模型的流式回复内容片段
        """
        client = self.get_client()
        
        try:
            # 将系统提示和用户提示合并
            combined_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=[combined_prompt],
                config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
            )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Gemini调用失败: {e}")
            print(f"错误详情: {error_details}")
            yield f"Gemini流式输出失败: {str(e)}"


class QwenLLM(BaseLLM):
    """
    Qwen模型实现类
    """
    
    def __init__(self, api_key, base_url=None):
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
    
    def get_streaming_response(self, prompt, system_prompt="You are a helpful assistant.", model="qwen-turbo"):
        """
        使用Qwen模型获取流式回复
        
        Args:
            prompt (str): 用户输入的提示
            system_prompt (str): 系统提示，默认为"You are a helpful assistant."
            model (str): 使用的Qwen模型，默认为"qwen-turbo"
        
        Yields:
            str: 模型的流式回复内容片段
        """
        client = self.get_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True  # 启用流式输出
        )
        
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                chunk_message = chunk.choices[0].delta.content
                if chunk_message:
                    yield chunk_message


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
    response = gemini_llm.get_streaming_response("Explain to me how AI works")
    print("Gemini回复:", response)
    '''
    # 使用Qwen模型
    qwen_llm = create_llm("qwen")
    response = qwen_llm.get_response("请解释一下人工智能是如何工作的")
    print("Qwen回复:", response)
    '''