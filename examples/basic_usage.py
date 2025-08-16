#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteMemory 基础使用示例
展示核心功能：添加、搜索、更新、删除记忆
"""

import logging
from lite_memory import LiteMemory
from lite_memory.config import MemoryConfig, EmbeddingConfig, VectorStoreConfig

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def basic_example():
    """基础功能演示"""
    print("=== LiteMemory 基础功能演示 ===\n")
    
    # 初始化记忆系统
    print("1. 初始化记忆系统...")
    memory = LiteMemory()
    
    # 测试连接
    if not memory.test_connection():
        print("❌ 连接失败！请确保Ollama服务正在运行。")
        print("提示：运行 'ollama serve' 启动服务")
        return
    
    print("✅ 连接成功！\n")
    
    # 添加记忆
    print("2. 添加记忆...")
    memories_to_add = [
        "我喜欢喝咖啡，特别是拿铁",
        "我住在北京，工作是软件工程师",
        "我的爱好是阅读科幻小说和看电影",
        "我最喜欢的编程语言是Python",
        "我正在学习机器学习和人工智能"
    ]
    
    memory_ids = []
    for text in memories_to_add:
        memory_id = memory.add(text)
        memory_ids.append(memory_id)
        print(f"✅ 添加记忆: {text[:30]}... (ID: {memory_id[:8]}...)")
    
    print(f"\n总共添加了 {len(memory_ids)} 条记忆")
    print(f"当前记忆总数: {memory.count()}\n")
    
    # 搜索记忆
    print("3. 搜索记忆...")
    queries = [
        "咖啡相关的内容",
        "我的工作是什么",
        "编程相关的内容",
        "我住在哪里"
    ]
    
    for query in queries:
        print(f"\n🔍 搜索: '{query}'")
        results = memory.search(query, limit=2)
        
        if results:
            for i, result in enumerate(results, 1):
                score = result.metadata.get('similarity_score', 0)
                print(f"  {i}. {result.text} (相似度: {score:.3f})")
        else:
            print("  没有找到相关记忆")
    
    # 获取指定记忆
    print(f"\n4. 获取指定记忆...")
    if memory_ids:
        memory_item = memory.get(memory_ids[0])
        if memory_item:
            print(f"✅ 记忆详情:")
            print(f"   ID: {memory_item.id}")
            print(f"   内容: {memory_item.text}")
            print(f"   创建时间: {memory_item.created_at}")
    
    # 更新记忆
    print("\n5. 更新记忆...")
    if memory_ids:
        updated_text = "我喜欢喝咖啡，特别是卡布奇诺和摩卡"
        success = memory.update(memory_ids[0], text=updated_text)
        if success:
            print(f"✅ 更新成功: {updated_text}")
            updated_memory = memory.get(memory_ids[0])
            print(f"   更新时间: {updated_memory.updated_at}")
    
    # 查找相似记忆
    print("\n6. 查找相似记忆...")
    if memory_ids:
        similar = memory.similar_memories(memory_ids[0], limit=2)
        print(f"与记忆 {memory_ids[0][:8]}... 相似的记忆:")
        for i, mem in enumerate(similar, 1):
            score = mem.metadata.get('similarity_score', 0)
            print(f"  {i}. {mem.text} (相似度: {score:.3f})")
    
    # 列出所有记忆
    print("\n7. 列出所有记忆...")
    all_memories = memory.list_all(limit=10)
    print(f"共有 {len(all_memories)} 条记忆:")
    for i, mem in enumerate(all_memories, 1):
        print(f"  {i}. {mem.text[:50]}{'...' if len(mem.text) > 50 else ''}")
    
    # 系统信息
    print("\n8. 系统信息...")
    info = memory.get_info()
    print(f"✅ 系统信息:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n=== 演示完成 ===")


def custom_config_example():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===\n")
    
    # 自定义配置
    embedding_config = EmbeddingConfig(
        model="nomic-embed-text",
        embedding_dims=768,
        ollama_base_url="http://localhost:11434"
    )
    
    vector_config = VectorStoreConfig(
        collection_name="custom_memories",
        path="./custom_memory_data",
        distance_strategy="cosine",
        normalize_L2=True
    )
    
    memory_config = MemoryConfig(
        embedding_config=embedding_config,
        vector_store_config=vector_config,
        max_memory_size=5000
    )
    
    # 使用自定义配置
    print("1. 使用自定义配置初始化...")
    custom_memory = LiteMemory(config=memory_config)
    
    # 添加一些记忆
    print("2. 添加自定义记忆...")
    custom_texts = [
        "这是自定义配置的记忆系统",
        "使用cosine距离计算相似度",
        "最大存储5000条记忆"
    ]
    
    for text in custom_texts:
        memory_id = custom_memory.add(text, metadata={"source": "custom_config"})
        print(f"✅ 添加: {text}")
    
    # 搜索
    print("\n3. 搜索自定义记忆...")
    results = custom_memory.search("配置", filters={"source": "custom_config"})
    for result in results:
        print(f"🔍 找到: {result.text}")
    
    print("✅ 自定义配置演示完成")


def batch_operations_example():
    """批量操作示例"""
    print("\n=== 批量操作示例 ===\n")
    
    memory = LiteMemory()
    
    # 批量添加
    print("1. 批量添加记忆...")
    batch_texts = [
        "Python是一种高级编程语言",
        "机器学习是人工智能的分支",
        "FAISS是Facebook开发的向量检索库",
        "Ollama支持本地运行大语言模型",
        "向量数据库用于存储高维向量"
    ]
    
    batch_metadata = [
        {"category": "编程", "language": "Python"},
        {"category": "AI", "field": "机器学习"},
        {"category": "工具", "company": "Facebook"},
        {"category": "工具", "type": "LLM"},
        {"category": "数据库", "type": "向量"}
    ]
    
    memory_ids = memory.add_batch(batch_texts, metadata_list=batch_metadata)
    print(f"✅ 批量添加了 {len(memory_ids)} 条记忆")
    
    # 按分类搜索
    print("\n2. 按分类搜索...")
    categories = ["编程", "AI", "工具"]
    
    for category in categories:
        results = memory.search("相关内容", filters={"category": category})
        print(f"\n📂 分类 '{category}' 的记忆:")
        for result in results:
            print(f"  - {result.text}")
    
    print("✅ 批量操作演示完成")


if __name__ == "__main__":
    try:
        # 基础功能演示
        basic_example()
        
        # 自定义配置演示
        custom_config_example()
        
        # 批量操作演示
        batch_operations_example()
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("请检查:")
        print("1. Ollama服务是否正在运行 (ollama serve)")
        print("2. 是否已下载嵌入模型 (ollama pull nomic-embed-text)")
        print("3. 网络连接是否正常")