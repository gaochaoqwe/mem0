# LiteMemory - 轻量级AI记忆模块

> 🚀 专为Python 3.8和Windows 7设计的离线AI记忆系统

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Windows 7](https://img.shields.io/badge/Windows-7+-lightgrey.svg)](https://www.microsoft.com/en-us/windows)

## 🎯 项目特色

LiteMemory 是一个轻量级的AI记忆系统，专门为需要在较老系统上运行的用户设计。它提供了核心的记忆存储和检索功能，无需复杂的依赖，支持完全离线部署。

### ✨ 核心优势

- **🔹 Python 3.8 兼容**：完美支持Python 3.8，兼容Windows 7等老系统
- **🔹 离线部署**：基于FAISS和Ollama，支持完全离线运行
- **🔹 轻量级设计**：精简的代码结构，最小化依赖
- **🔹 高性能**：使用FAISS进行快速向量搜索
- **🔹 易于使用**：简洁的API设计，开箱即用

### 🛠️ 技术栈

- **向量存储**：FAISS（Facebook AI Similarity Search）
- **嵌入模型**：Ollama（支持本地部署）
- **数据格式**：Pickle + JSON
- **兼容性**：Python 3.8+，Windows 7+

## 📦 安装指南

### 系统要求

- Python 3.8 或更高版本
- Windows 7+ / Linux / macOS
- 至少 500MB 可用磁盘空间

### 快速安装

```bash
# 基础安装
pip install lite-memory

# 或者从源码安装
git clone https://github.com/lite-memory/lite-memory.git
cd lite-memory
pip install -e .
```

### Ollama 安装

LiteMemory 需要 Ollama 作为嵌入模型服务：

```bash
# Windows
# 下载并安装 Ollama for Windows
# https://ollama.ai/download/windows

# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# 下载嵌入模型
ollama pull nomic-embed-text
```

## 🚀 快速开始

### 基础使用

```python
from lite_memory import LiteMemory

# 初始化记忆系统
memory = LiteMemory()

# 添加记忆
memory_id = memory.add("我喜欢喝咖啡")
print(f"添加记忆成功，ID: {memory_id}")

# 搜索相关记忆
results = memory.search("咖啡相关的内容")
for result in results:
    print(f"记忆: {result.text}, 相似度: {result.metadata['similarity_score']}")

# 获取指定记忆
memory_item = memory.get(memory_id)
print(f"记忆内容: {memory_item.text}")
```

### 批量操作

```python
# 批量添加记忆
texts = [
    "我住在北京",
    "我的工作是程序员", 
    "我喜欢阅读科幻小说"
]
memory_ids = memory.add_batch(texts)
print(f"批量添加了 {len(memory_ids)} 条记忆")

# 列出所有记忆
all_memories = memory.list_all(limit=10)
for mem in all_memories:
    print(f"ID: {mem.id}, 内容: {mem.text}")
```

### 高级配置

```python
from lite_memory import LiteMemory
from lite_memory.config import MemoryConfig, EmbeddingConfig, VectorStoreConfig

# 自定义配置
embedding_config = EmbeddingConfig(
    model="nomic-embed-text",
    embedding_dims=768,
    ollama_base_url="http://localhost:11434"
)

vector_config = VectorStoreConfig(
    collection_name="my_memories",
    path="./my_memory_data",
    distance_strategy="cosine"
)

memory_config = MemoryConfig(
    embedding_config=embedding_config,
    vector_store_config=vector_config,
    max_memory_size=10000
)

# 使用自定义配置初始化
memory = LiteMemory(config=memory_config)
```

## 📋 API 参考

### LiteMemory 类

#### 核心方法

- `add(text, metadata=None, memory_id=None)` - 添加记忆
- `search(query, limit=5, filters=None)` - 搜索记忆
- `get(memory_id)` - 获取指定记忆
- `update(memory_id, text=None, metadata=None)` - 更新记忆
- `delete(memory_id)` - 删除记忆

#### 批量操作

- `add_batch(texts, metadata_list=None)` - 批量添加记忆
- `list_all(filters=None, limit=100)` - 列出所有记忆
- `clear_all()` - 清空所有记忆

#### 实用功能

- `similar_memories(memory_id, limit=5)` - 查找相似记忆
- `count()` - 获取记忆总数
- `get_info()` - 获取系统信息
- `test_connection()` - 测试连接状态

## 💡 使用示例

### 个人知识库

```python
from lite_memory import LiteMemory

# 创建个人知识库
kb = LiteMemory()

# 添加知识条目
kb.add("Python是一种高级编程语言", {"category": "编程", "topic": "Python"})
kb.add("机器学习是人工智能的一个分支", {"category": "AI", "topic": "机器学习"})
kb.add("FAISS是Facebook开发的向量检索库", {"category": "工具", "topic": "向量数据库"})

# 智能搜索
results = kb.search("Python编程相关")
for result in results:
    category = result.metadata.get("category", "未分类")
    print(f"[{category}] {result.text}")
```

### 对话记忆系统

```python
from lite_memory import LiteMemory
from datetime import datetime

class ChatMemory:
    def __init__(self):
        self.memory = LiteMemory()
    
    def remember_conversation(self, user_input, ai_response, user_id="default"):
        """记住对话内容"""
        timestamp = datetime.now().isoformat()
        
        # 记住用户输入
        self.memory.add(
            text=user_input,
            metadata={
                "type": "user_input",
                "user_id": user_id,
                "timestamp": timestamp,
                "response": ai_response
            }
        )
    
    def recall_context(self, current_input, user_id="default", limit=3):
        """回忆相关对话上下文"""
        filters = {"user_id": user_id, "type": "user_input"}
        related_memories = self.memory.search(
            query=current_input,
            limit=limit,
            filters=filters
        )
        
        context = []
        for mem in related_memories:
            context.append({
                "user": mem.text,
                "assistant": mem.metadata.get("response", ""),
                "time": mem.metadata.get("timestamp", "")
            })
        
        return context

# 使用示例
chat_memory = ChatMemory()

# 记录对话
chat_memory.remember_conversation(
    user_input="我想学习Python",
    ai_response="Python是一个很好的编程语言选择..."
)

# 回忆相关上下文
context = chat_memory.recall_context("Python相关的问题")
for item in context:
    print(f"用户: {item['user']}")
    print(f"助手: {item['assistant']}")
    print(f"时间: {item['time']}")
    print("---")
```

## 🔧 配置选项

### 嵌入模型配置

```python
from lite_memory.config import EmbeddingConfig

config = EmbeddingConfig(
    model="nomic-embed-text",          # Ollama模型名称
    embedding_dims=768,                # 嵌入向量维度
    ollama_base_url="http://localhost:11434"  # Ollama服务地址
)
```

### 向量存储配置

```python
from lite_memory.config import VectorStoreConfig

config = VectorStoreConfig(
    collection_name="my_collection",   # 集合名称
    path="./data",                     # 存储路径
    distance_strategy="cosine",        # 距离策略: cosine, euclidean, inner_product
    normalize_L2=True,                 # 是否L2归一化
    embedding_dims=768                 # 向量维度
)
```

## 🐛 故障排除

### 常见问题

1. **Ollama 连接失败**
   ```bash
   # 检查 Ollama 是否运行
   ollama list
   
   # 启动 Ollama 服务
   ollama serve
   ```

2. **FAISS 安装失败**
   ```bash
   # Windows 7 用户推荐使用 CPU 版本
   pip install faiss-cpu
   
   # 避免安装 GPU 版本（除非有 CUDA 支持）
   pip uninstall faiss-gpu
   ```

3. **内存不足**
   ```python
   # 限制最大记忆数量
   config = MemoryConfig(max_memory_size=1000)
   memory = LiteMemory(config=config)
   ```

### 性能优化

- 定期调用 `memory.save()` 手动保存数据
- 使用过滤条件减少搜索范围
- 合理设置 `embedding_dims` 平衡精度和性能
- 批量操作替代单条操作

## 📈 性能基准

在典型的Windows 7系统上（4GB RAM, Intel Core i5）：

- **添加记忆**: ~100ms/条
- **搜索记忆**: ~50ms（1000条记忆库）
- **批量添加**: ~10ms/条（批量100条）
- **内存占用**: ~50MB（10000条记忆）

## 🤝 贡献指南

欢迎贡献代码、报告bug或提出功能建议！

```bash
# 克隆项目
git clone https://github.com/lite-memory/lite-memory.git
cd lite-memory

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black lite_memory/
isort lite_memory/
```

## 📄 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [FAISS](https://github.com/facebookresearch/faiss) - 高性能向量搜索库
- [Ollama](https://ollama.ai/) - 本地大语言模型部署工具
- [Mem0](https://github.com/mem0ai/mem0) - 原项目灵感来源

---

💡 **提示**: 如果您在 Windows 7 上遇到兼容性问题，请参考我们的 [Windows 7 兼容性指南](docs/windows7-guide.md)。