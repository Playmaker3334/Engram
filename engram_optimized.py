from typing import List
from dataclasses import dataclass, field
from collections import OrderedDict
import math
import time
import logging

from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    cache_size: int = 10000
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

class CompressedTokenizer:
    def __init__(self, tokenizer_name_or_path):
        logger.info("Initializing CompressedTokenizer with tokenizer=%s", tokenizer_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        logger.info("Building vocabulary compression lookup table")
        self.lookup_table, self.num_new_token = self._build_lookup_table()
        compression_ratio = (1 - self.num_new_token / len(self.tokenizer)) * 100
        logger.info("Vocabulary compressed: %d -> %d tokens (%.2f%% reduction)", 
                   len(self.tokenizer), self.num_new_token, compression_ratio)
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask].astype(np.int64)
        # Clip to valid range to prevent index out of bounds
        valid_ids = np.clip(valid_ids, 0, len(self.lookup_table) - 1)
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)

class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        logger.debug("Initializing ShortConv: hidden_size=%d, kernel_size=%d, dilation=%d, hc_mult=%d",
                    hidden_size, kernel_size, dilation, hc_mult)
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, G, C = x.shape
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y

def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class FastNgramHashMapping(nn.Module):
    """GPU-accelerated N-gram hashing"""
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        super().__init__()
        logger.info("Initializing FastNgramHashMapping for layers=%s", layer_ids)
        
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
        
        logger.info("Computing layer-specific multipliers with seed=%d", seed)
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers
            logger.debug("Layer %d multipliers: %s", layer_id, multipliers[:3])

        logger.info("Calculating vocabulary sizes across layers")
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()
        
        logger.info("Converting multipliers and prime mods to GPU tensors")
        for layer_id in self.layer_ids:
            mult_tensor = torch.tensor(
                self.layer_multipliers[layer_id], 
                dtype=torch.long
            )
            self.register_buffer(f"multipliers_layer_{layer_id}", mult_tensor)
            
            all_primes = []
            for ngram_heads in self.vocab_size_across_layers[layer_id]:
                all_primes.extend(ngram_heads)
            prime_tensor = torch.tensor(all_primes, dtype=torch.long)
            self.register_buffer(f"prime_mods_layer_{layer_id}", prime_tensor)
            
            logger.info("Layer %d: registered %d prime modulos on GPU", 
                       layer_id, len(all_primes))

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes_gpu(
        self,
        input_ids: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """GPU-accelerated hashing using torch operations"""
        B, T = input_ids.shape
        device = input_ids.device
        
        logger.debug("Computing GPU hash for layer=%d, batch_size=%d, seq_len=%d", 
                    layer_id, B, T)
        
        start_time = time.time()
        
        multipliers = getattr(self, f"multipliers_layer_{layer_id}")
        prime_mods = getattr(self, f"prime_mods_layer_{layer_id}")
        
        pad_tensor = torch.full((B, 1), self.pad_id, dtype=torch.long, device=device)
        
        shifts = []
        for k in range(self.max_ngram_size):
            if k == 0:
                shifts.append(input_ids)
            else:
                padding = pad_tensor.expand(B, k)
                shifted = torch.cat([padding, input_ids[:, :-k]], dim=1)
                shifts.append(shifted)
        
        all_hashes = []
        hash_idx = 0
        
        for n in range(2, self.max_ngram_size + 1):
            tokens = shifts[:n]
            
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            num_heads = self.n_head_per_ngram
            
            for j in range(num_heads):
                mod_val = prime_mods[hash_idx]
                head_hash = mix % mod_val
                all_hashes.append(head_hash)
                hash_idx += 1
        
        result = torch.stack(all_hashes, dim=2)
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug("GPU hash computation completed in %.2fms", elapsed_ms)
        
        return result

    def hash(self, input_ids):
        """Main hashing interface - converts to compressed IDs then hashes on GPU"""
        logger.debug("Starting hash pipeline for input_ids shape=%s", input_ids.shape)
        
        device = input_ids.device
        
        logger.debug("Compressing tokenizer vocabulary on CPU")
        input_ids_cpu = input_ids.cpu().numpy()
        compressed_ids = self.compressed_tokenizer(input_ids_cpu)
        
        compressed_tensor = torch.from_numpy(compressed_ids).to(device)
        
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes_gpu(
                compressed_tensor, 
                layer_id=layer_id
            )
        
        logger.debug("Hash pipeline complete for %d layers", len(self.layer_ids))
        return hash_ids_for_all_layers

class LRUCache:
    """Simple LRU cache for embedding lookups"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        logger.info("Initialized LRU cache with capacity=%d", capacity)
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                evicted = self.cache.popitem(last=False)
                logger.debug("Cache full, evicted key with hash=%s", hash(evicted[0]))
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": hit_rate
        }
    
    def reset_stats(self):
        self.hits = 0
        self.misses = 0

class CachedMultiHeadEmbedding(nn.Module):
    """MultiHeadEmbedding with LRU cache"""
    def __init__(self, list_of_N: List[int], D: int, cache_size: int = 10000):
        super().__init__()
        logger.info("Initializing CachedMultiHeadEmbedding with cache_size=%d", cache_size)
        
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.cache = LRUCache(cache_size)
        
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        logger.info("Total embedding table size: %d entries x %d dims", total_N, D)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T, num_heads = input_ids.shape
        
        cache_key = tuple(input_ids.flatten().tolist())
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Cache hit for batch_size=%d, seq_len=%d", B, T)
            return cached_result
        
        logger.debug("Cache miss, computing embeddings")
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        
        self.cache.put(cache_key, output)
        
        return output
    
    def log_cache_stats(self):
        stats = self.cache.get_stats()
        logger.info("Cache stats - Hits: %d, Misses: %d, Hit rate: %.2f%%",
                   stats["hits"], stats["misses"], stats["hit_rate_pct"])
        return stats

class Engram(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        logger.info("Initializing Engram module for layer_id=%d", layer_id)
        
        self.layer_id = layer_id
        self.hash_mapping = FastNgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        
        self.multi_head_embedding = CachedMultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
            cache_size = engram_cfg.cache_size,
        )
        
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        
        logger.info("Engram module initialization complete for layer %d", layer_id)
    
    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """
        logger.debug("Engram forward pass - input_ids shape=%s, hidden_states shape=%s",
                    input_ids.shape, hidden_states.shape)
        
        start_time = time.time()
        
        hash_result = self.hash_mapping.hash(input_ids)
        hash_input_ids = hash_result[self.layer_id].to(input_ids.device)
        
        hash_time = (time.time() - start_time) * 1000
        logger.debug("Hash computation took %.2fms", hash_time)
        
        embed_start = time.time()
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        embed_time = (time.time() - embed_start) * 1000
        logger.debug("Embedding lookup took %.2fms", embed_time)
        
        gate_start = time.time()
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates, dim=2)
        gate_time = (time.time() - gate_start) * 1000
        logger.debug("Gating computation took %.2fms", gate_time)
        
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)
        
        total_time = (time.time() - start_time) * 1000
        logger.info("Engram forward pass complete in %.2fms (hash=%.2f, embed=%.2f, gate=%.2f)",
                   total_time, hash_time, embed_time, gate_time)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        logger.debug("Initializing TransformerBlock for layer %d", layer_id)
        self.attn = lambda x: x
        self.moe = lambda x: x
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)
            logger.info("TransformerBlock %d includes Engram module", layer_id)
    
    def forward(self, input_ids, hidden_states):
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states

if __name__ == '__main__':
    logger.info("Starting Engram optimized demo")
    
    LLM = [
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]
    
    logger.info("Model architecture initialized with %d layers", backbone_config.num_layers)

    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    
    logger.info("Input text tokenized: %s", text)
    logger.info("Input shape: batch_size=%d, seq_len=%d", input_ids.shape[0], input_ids.shape[1])

    B, L = input_ids.shape
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU device: %s", torch.cuda.get_device_name(0))
        input_ids = input_ids.to(device)
        for layer in LLM:
            layer.to(device)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    logger.info("Starting forward pass through %d layers", len(LLM))
    forward_start = time.time()

    for idx, layer in enumerate(LLM):
        layer_start = time.time()
        
        if idx == 0:
            hidden_states = LLM[0](input_ids)
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)
            logger.debug("Layer 0 (embedding): output shape=%s", hidden_states.shape)
        elif idx == len(LLM) - 1:
            hidden_states = hidden_states[:,:,0,:] 
            output = layer(hidden_states)
            logger.debug("Layer %d (output): output shape=%s", idx, output.shape)
        else:
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)
            logger.debug("Layer %d: output shape=%s", idx, hidden_states.shape)
        
        layer_time = (time.time() - layer_start) * 1000
        logger.debug("Layer %d forward pass took %.2fms", idx, layer_time)

    forward_time = (time.time() - forward_start) * 1000
    logger.info("Total forward pass completed in %.2fms", forward_time)
    
    logger.info("Final output shape: %s", output.shape)
    logger.info("Input IDs shape: %s", input_ids.shape)
    
    for layer in LLM:
        if isinstance(layer, TransformerBlock) and layer.engram is not None:
            stats = layer.engram.multi_head_embedding.log_cache_stats()
    
    logger.info("Forward pass complete successfully")