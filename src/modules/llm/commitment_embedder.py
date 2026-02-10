import torch
from typing import List, Optional, Any
from loguru import logger
from .llm_wrapper import APIHandler, LLMConfig # Assuming APIHandler and LLMConfig are in llm_wrapper.py

class CommitmentEmbedder:
    """
    Encapsulates the logic for fetching commitment text embeddings using an API handler.
    """
    def __init__(self, args: Any, llm_config: LLMConfig):
        """
        Initializes the CommitmentEmbedder.

        Args:
            args: Configuration arguments, expected to contain 'commitment_embedding_model_name' 
                  and potentially other embedding related settings.
            llm_config: LLMConfig object for initializing the APIHandler.
        """
        self.args = args
        self.llm_config = llm_config
        
        self.embedding_model_name = getattr(args, "commitment_embedding_model_name", "BAAI/bge-large-en-v1.5")
        logger.info(f"CommitmentEmbedder initialized with model: {self.embedding_model_name}")

        self.api_handler = APIHandler(llm_config)
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def embed_commitments(self, commitment_texts: List[str]) -> Optional[torch.Tensor]:
        """
        Fetches embeddings for a list of commitment texts.
        
        如果API嵌入不可用，则使用简单的基于文本长度的本地嵌入。

        Args:
            commitment_texts: A list of strings, where each string is a commitment text.

        Returns:
            A torch.Tensor containing the embeddings (batch_size, embedding_dim),
            or None if fetching embeddings fails.
        """
        if not commitment_texts:
            logger.warning("embed_commitments called with an empty list of texts.")
            return None

        try:
            embeddings_list_of_lists = self.api_handler.generate_embeddings(
                input_texts=commitment_texts,
                model=self.embedding_model_name
            )

            if embeddings_list_of_lists is not None:
                embeddings_tensor = torch.tensor(embeddings_list_of_lists, dtype=torch.float32).to(self.device)
                if hasattr(self.args, 'debug') and self.args.debug:
                     logger.debug(f"Successfully embedded {len(commitment_texts)} commitments via API. Shape: {embeddings_tensor.shape}")
                return embeddings_tensor
            else:
                logger.warning("API embedding failed, using local text-based embedding fallback")
                return self._generate_local_embeddings(commitment_texts)
                
        except Exception as e:
            logger.warning(f"Error during API embedding, falling back to local embedding: {e}")
            return self._generate_local_embeddings(commitment_texts)
    
    def _generate_local_embeddings(self, commitment_texts: List[str]) -> torch.Tensor:
        """
        生成基于文本特征的简单本地嵌入。
        
        Args:
            commitment_texts: 承诺文本列表
            
        Returns:
            torch.Tensor: 本地生成的嵌入向量
        """
        embedding_dim = getattr(self.args, 'commitment_embedding_dim', 1024)
        batch_size = len(commitment_texts)
        
        embeddings = []
        for text in commitment_texts:
            text_length = len(text)
            char_counts = {}
            for char in text.lower():
                char_counts[char] = char_counts.get(char, 0) + 1
            
            base_features = [
                text_length / 100.0,  # 归一化文本长度
                len(set(text.lower())) / 26.0,  # 归一化唯一字符数
                text.count(' ') / max(1, text_length),  # 空格密度
                text.count('.') / max(1, text_length),  # 句号密度
            ]
            
            embedding = []
            torch.manual_seed(hash(text) % 2**31)  # 基于文本内容的确定性种子
            
            for i in range(embedding_dim):
                if i < len(base_features):
                    embedding.append(base_features[i])
                else:
                    embedding.append(torch.randn(1).item() * 0.1)
            
            embeddings.append(embedding)
        
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        if hasattr(self.args, 'debug') and self.args.debug:
            logger.debug(f"Generated local embeddings for {batch_size} commitments. Shape: {embeddings_tensor.shape}")
        
        return embeddings_tensor 