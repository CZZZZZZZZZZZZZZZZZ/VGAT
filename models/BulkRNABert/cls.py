import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import os
import torch
import numpy as np

from bulk_rna_bert.pretrained import get_pretrained_model
from bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
print(jax.local_device_count())

device = jax.devices("gpu")[1]

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory="VGAT-main/models/BulkRNABert/checkpoints",
)
forward_fn = hk.transform(forward_fn)

# 将参数移动到指定设备
parameters = jax.device_put(parameters, device)

# Get bulk RNASeq data and tokenize it
rna_seq_df = pd.read_csv("./tcga_blca/blca_preprocessed.csv")
rna_name = rna_seq_df["identifier"]
rna_seq_array = preprocess_rna_seq_for_bulkrnabert(rna_seq_df, config)

batch_size = 1
num_samples = len(rna_seq_array)

# 外部指定的保存目录
save_directory = ""
os.makedirs(save_directory, exist_ok=True)

# 使用指定设备的上下文管理器
with jax.default_device(device):
    for i in range(0, num_samples, batch_size):
        batch_tokens_ids = tokenizer.batch_tokenize(rna_seq_array[i:i + batch_size])
        batch_tokens = jnp.asarray(batch_tokens_ids, dtype=jnp.int32)
        
        # 将数据移动到指定设备
        batch_tokens = jax.device_put(batch_tokens, device)
        
        # 进行推理
        random_key = jax.random.PRNGKey(0)
        outs = forward_fn.apply(parameters, random_key, batch_tokens)
        
        current_name = rna_name[i:i + batch_size].iloc[0]
        # current_name = current_name.rsplit('-', 1)[0]
        
        # Get mean embeddings from layer 4
        mean_embedding = outs["embeddings_4"].mean(axis=1)
        
        # 将结果移回主机内存
        mean_embedding = jax.device_get(mean_embedding)
        
        save_path = os.path.join(save_directory, f"{current_name}.pt")
        
        # 保存到指定地址
        torch.save(mean_embedding, save_path)
        
        # 释放显存
        del batch_tokens_ids
        del batch_tokens
        del outs
        del mean_embedding
        
        # 手动调用垃圾回收
        import gc
        gc.collect()
