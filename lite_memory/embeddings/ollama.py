"""
Ollama嵌入模型实现 - 支持离线部署
重构自原项目，适配Python 3.8
"""

import logging
import sys
import subprocess
from typing import List, Optional

from lite_memory.embeddings.base import EmbeddingBase
from lite_memory.config import EmbeddingConfig

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入ollama，如果失败则提示安装
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests库未安装，将尝试安装")

try:
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama库未安装")


class OllamaEmbedding(EmbeddingBase):
    """Ollama嵌入模型实现"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化Ollama嵌入模型
        
        Args:
            config: 嵌入配置
        """
        super().__init__(config)
        
        # 检查并安装依赖
        self._ensure_dependencies()
        
        # 设置默认模型和维度
        if not self.config.model:
            self.config.model = "nomic-embed-text"
        if not self.config.embedding_dims:
            self.config.embedding_dims = 768
        
        # 初始化客户端
        self.client = None
        self._init_client()
        
        # 确保模型存在
        self._ensure_model_exists()
    
    def _ensure_dependencies(self):
        """确保依赖库已安装"""
        if not REQUESTS_AVAILABLE:
            self._install_package("requests")
        
        if not OLLAMA_AVAILABLE:
            self._install_package("ollama")
    
    def _install_package(self, package_name: str):
        """安装Python包"""
        try:
            print(f"正在安装 {package_name} 库...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} 安装成功")
            
            # 重新导入
            if package_name == "ollama":
                global Client, OLLAMA_AVAILABLE
                from ollama import Client
                OLLAMA_AVAILABLE = True
            elif package_name == "requests":
                global requests, REQUESTS_AVAILABLE
                import requests
                REQUESTS_AVAILABLE = True
                
        except subprocess.CalledProcessError as e:
            logger.error(f"安装 {package_name} 失败: {e}")
            raise RuntimeError(f"无法安装 {package_name}，请手动安装: pip install {package_name}")
    
    def _init_client(self):
        """初始化Ollama客户端"""
        try:
            self.client = Client(host=self.config.ollama_base_url)
            # 测试连接
            self.client.list()
            logger.info(f"已连接到Ollama服务: {self.config.ollama_base_url}")
        except Exception as e:
            logger.error(f"无法连接到Ollama服务: {e}")
            raise ConnectionError(
                f"无法连接到Ollama服务 ({self.config.ollama_base_url})。"
                f"请确保Ollama服务正在运行。"
            )
    
    def _ensure_model_exists(self):
        """确保指定的模型在本地存在，如果不存在则下载"""
        try:
            local_models = self.client.list()
            model_names = [model.get("name", "").split(":")[0] for model in local_models.get("models", [])]
            
            if self.config.model not in model_names:
                logger.info(f"模型 {self.config.model} 不存在，正在下载...")
                self.client.pull(self.config.model)
                logger.info(f"模型 {self.config.model} 下载完成")
            else:
                logger.info(f"模型 {self.config.model} 已存在")
                
        except Exception as e:
            logger.error(f"检查/下载模型失败: {e}")
            raise RuntimeError(f"无法获取模型 {self.config.model}: {e}")
    
    def embed(self, text: str) -> List[float]:
        """
        使用Ollama生成文本嵌入
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            List[float]: 嵌入向量
        """
        if not text or not text.strip():
            # 返回零向量
            return [0.0] * self.config.embedding_dims
        
        try:
            response = self.client.embeddings(
                model=self.config.model,
                prompt=text.strip()
            )
            
            embedding = response.get("embedding", [])
            if not embedding:
                logger.warning(f"获取到空的嵌入向量，文本: {text[:50]}...")
                return [0.0] * self.config.embedding_dims
            
            # 确保向量维度正确
            if len(embedding) != self.config.embedding_dims:
                logger.warning(
                    f"嵌入向量维度不匹配: 期望 {self.config.embedding_dims}, "
                    f"实际 {len(embedding)}"
                )
                # 截断或填充到正确维度
                if len(embedding) > self.config.embedding_dims:
                    embedding = embedding[:self.config.embedding_dims]
                else:
                    embedding.extend([0.0] * (self.config.embedding_dims - len(embedding)))
            
            return embedding
            
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            # 返回零向量作为fallback
            return [0.0] * self.config.embedding_dims
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        results = []
        for text in texts:
            try:
                embedding = self.embed(text)
                results.append(embedding)
            except Exception as e:
                logger.error(f"批量嵌入失败，文本: {text[:50]}..., 错误: {e}")
                # 添加零向量
                results.append([0.0] * self.config.embedding_dims)
        
        return results
    
    def test_connection(self) -> bool:
        """
        测试与Ollama服务的连接
        
        Returns:
            bool: 连接是否正常
        """
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """
        列出可用的模型
        
        Returns:
            List[str]: 模型名称列表
        """
        try:
            models = self.client.list()
            return [model.get("name", "") for model in models.get("models", [])]
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []