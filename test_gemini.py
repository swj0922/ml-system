#!/usr/bin/env python3
"""
测试Gemini流式输出的脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llm_config import create_llm

def test_gemini_streaming():
    """测试Gemini流式输出"""
    print("=== 测试Gemini流式输出 ===")
    
    try:
        # 创建Gemini LLM实例
        gemini_llm = create_llm("gemini")
        print("✓ Gemini LLM实例创建成功")
        
        # 测试流式输出
        prompt = "请简单介绍一下机器学习，用中文回答，大约500字。"
        print(f"发送提示: {prompt}")
        print("流式输出结果:")
        print("-" * 50)
        
        chunk_count = 0
        total_content = ""
        
        for chunk in gemini_llm.get_streaming_response(prompt):
            chunk_count += 1
            total_content += chunk
            print(f"[Chunk {chunk_count}]: {repr(chunk)}")
            
        print("-" * 50)
        print(f"总共收到 {chunk_count} 个chunks")
        print(f"完整内容: {total_content}")
        
        if chunk_count == 0:
            print("❌ 没有收到任何流式输出!")
        else:
            print("✓ 流式输出正常")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_qwen_streaming():
    """测试Qwen流式输出作为对比"""
    print("\n=== 测试Qwen流式输出 (对比) ===")
    
    try:
        # 创建Qwen LLM实例
        qwen_llm = create_llm("qwen")
        print("✓ Qwen LLM实例创建成功")
        
        # 测试流式输出
        prompt = "请简单介绍一下机器学习，用中文回答，大约50字。"
        print(f"发送提示: {prompt}")
        print("流式输出结果:")
        print("-" * 50)
        
        chunk_count = 0
        total_content = ""
        
        for chunk in qwen_llm.get_streaming_response(prompt):
            chunk_count += 1
            total_content += chunk
            print(f"[Chunk {chunk_count}]: {repr(chunk)}")
            
        print("-" * 50)
        print(f"总共收到 {chunk_count} 个chunks")
        print(f"完整内容: {total_content}")
        
        if chunk_count == 0:
            print("❌ 没有收到任何流式输出!")
        else:
            print("✓ 流式输出正常")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini_streaming()
    #test_qwen_streaming()