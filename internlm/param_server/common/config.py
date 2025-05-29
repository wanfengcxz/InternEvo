import os
import random
import torch

random.seed(0)

ip_str = os.getenv("PS_SERVERS")
if ip_str is None:
    print(f"error, environment variable PS_SERVERS not set!")
    sys.exit(1)
ip_list = ps_servers.split(',')
ps_servers = {i: ip for i, ip in enumerate(ip_list)}

MASTER_ADDR = os.getenv("PS_SERVER_MASTER_ADDR")
if MASTER_ADDR is None:
    print(f"error, environment variable MASTER_ADDR not set!")
    sys.exit(1)

NUM_PS = len(ps_servers)

# MASTER_ADDR = "10.10.41.41"
MASTER_PORT = 55500
ZMQ_PORT = 55503
GRPC_PORT = 55504
GRPC_TIMEOUT = 10.0  # second

UPDATE_METHOD = "partial_sync"  # sync or partial_sync or async_buffer

GRACE_TIME_IN_SECOND = 60
PARTIAL_SYNC_RATE_THRESHOLD = 0.8
PARTIAL_SYNC_WAIT_TIME = 300  # second
SOCKET_TIME_IN_SECOND = 300

PS_OFFLINE_THRESHOLD = 80  # second
GROUPS_OFFLINE_THRESHOLD = 80  # second
# PS_OFFLINE_THRESHOLD = 5 * 60 # second
# GROUPS_OFFLINE_THRESHOLD = 5 * 60 # second
ALIVE_CHECK_INTERVAL = 25  # second
DUMP_STATE_INTERVAL = 60  # second
HEARTBEAT_INTERVAL = 25  # second


CONSUME_TOKEN_WHEN_PS_SAVE_CKPT = 1024 * 1024 * 1024 * 1024  # 1B

master_server = f"{MASTER_ADDR}:{MASTER_PORT}"
grpc_servers = {ps_id: f"{server}:{GRPC_PORT}" for ps_id, server in ps_servers.items()}
zmq_servers = {ps_id: f"tcp://{server}:{ZMQ_PORT}" for ps_id, server in ps_servers.items()}


model_type = os.getenv("MODEL_TYPE")
if model_type is None:
    print(f"error, environment variable MODEL_TYPE not set!")
    sys.exit(1)
    
MODEL_TYPE_LIST = ["INTERNLM_2_7B", "QWEN_2_7B", "LLAMA_2_7B"]
assert model_type in MODEL_TYPE_LIST "error, model don't support!"
MODEL_PARAM_DICT = dict()

MODEL_PARAM_DICT["LLAMA_2_7B"] = dict()
MODEL_PARAM_DICT["LLAMA_2_7B"]["NUM_LAYERS"] = 32
MODEL_PARAM_DICT["LLAMA_2_7B"]["MLP_RATIO"] = 2.6875
MODEL_PARAM_DICT["LLAMA_2_7B"]["HIDDEN_SIZE"] = 4096
MODEL_PARAM_DICT["LLAMA_2_7B"]["NUM_ATTENTION_HEAD"] = 32
MODEL_PARAM_DICT["LLAMA_2_7B"]["NUM_KV_ATTENTION_HEAD"] = 32
MODEL_PARAM_DICT["LLAMA_2_7B"]["VOCAB_SIZE"] = 32000
MODEL_PARAM_DICT["LLAMA_2_7B"]["HEAD_DIM"] = 128

MODEL_PARAM_DICT["QWEN_2_7B"] = dict()
MODEL_PARAM_DICT["QWEN_2_7B"]["NUM_LAYERS"] = 28
MODEL_PARAM_DICT["QWEN_2_7B"]["MLP_RATIO"] = 5.25
MODEL_PARAM_DICT["QWEN_2_7B"]["HIDDEN_SIZE"] = 3584
MODEL_PARAM_DICT["QWEN_2_7B"]["NUM_ATTENTION_HEAD"] = 28
MODEL_PARAM_DICT["QWEN_2_7B"]["NUM_KV_ATTENTION_HEAD"] = 4
MODEL_PARAM_DICT["QWEN_2_7B"]["VOCAB_SIZE"] = 152064
MODEL_PARAM_DICT["QWEN_2_7B"]["HEAD_DIM"] = 128

MODEL_PARAM_DICT["INTERNLM_2_7B"] = dict()
MODEL_PARAM_DICT["INTERNLM_2_7B"]["NUM_LAYERS"] = 32
MODEL_PARAM_DICT["INTERNLM_2_7B"]["MLP_RATIO"] = 3.5
MODEL_PARAM_DICT["INTERNLM_2_7B"]["HIDDEN_SIZE"] = 4096
MODEL_PARAM_DICT["INTERNLM_2_7B"]["NUM_ATTENTION_HEAD"] = 32
MODEL_PARAM_DICT["INTERNLM_2_7B"]["NUM_KV_ATTENTION_HEAD"] = 8
MODEL_PARAM_DICT["INTERNLM_2_7B"]["VOCAB_SIZE"] = 92544
MODEL_PARAM_DICT["INTERNLM_2_7B"]["HEAD_DIM"] = 128

NUM_LAYERS = MODEL_PARAM_DICT[model_type]["NUM_LAYERS"]
MLP_RATIO = MODEL_PARAM_DICT[model_type]["MLP_RATIO"]
HIDDEN_SIZE = MODEL_PARAM_DICT[model_type]["HIDDEN_SIZE"]
NUM_ATTENTION_HEAD = MODEL_PARAM_DICT[model_type]["NUM_ATTENTION_HEAD"]
NUM_KV_ATTENTION_HEAD = MODEL_PARAM_DICT[model_type]["NUM_KV_ATTENTION_HEAD"]
VOCAB_SIZE = MODEL_PARAM_DICT[model_type]["VOCAB_SIZE"]
HEAD_DIM = MODEL_PARAM_DICT[model_type]["HEAD_DIM"]

model = dict(
    dtype=torch.bfloat16,
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    vocab_size=VOCAB_SIZE,
    num_attention_heads=NUM_ATTENTION_HEAD,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    mlp_ratio=MLP_RATIO,
    head_dim=HEAD_DIM,
)


def get_chunks(num_layers, num_chunks):
    layer_idxs = list(range(num_layers))
    chunk_size = num_layers // num_chunks
    remaining_size = num_layers % num_chunks

    chunks = []
    start_idx = 0
    for _ in range(num_chunks - remaining_size):
        chunks.append(layer_idxs[start_idx : start_idx + chunk_size])
        start_idx += chunk_size

    if remaining_size == 0:
        return chunks

    chunk_size += 1
    for _ in range(num_chunks - remaining_size, num_chunks):
        chunks.append(layer_idxs[start_idx : start_idx + chunk_size])
        start_idx += chunk_size

    if num_chunks > 2:
        chunks[1][-1], chunks[-1][-1] = chunks[-1][-1], chunks[1][-1]

    return chunks


def get_param_shapes(config):
    """
    Given a configuration dictionary for a LLaMA2 model, this function computes and outputs
    the shapes of common parameters such as w1, w2, w3, wq, wk, wv, etc.

    Args:
        config (dict): Dictionary containing the model configuration.

    Returns:
        dict: A dictionary mapping parameter names to their shapes.
    """
    # Extract config parameters
    hidden_size = config.get("hidden_size")
    num_attention_heads = config.get("num_attention_heads")
    num_kv_attention_heads = config.get("num_kv_attention_heads")
    mlp_ratio = config.get("mlp_ratio")
    vocab_size = config.get("vocab_size")

    mlp_hidden_features = int(hidden_size * mlp_ratio)
    if model_type.startswith("QWEN_2"):
        multiple_of = 256
        mlp_hidden_features = (mlp_hidden_features + multiple_of - 1) // multiple_of * multiple_of
    # Compute per-head size
    attn_head_dim = config.get("head_dim")
    q_dim = attn_head_dim * num_attention_heads
    kv_dim = attn_head_dim * num_kv_attention_heads

    # Define parameter shapes
    param_shapes = {
        "attention_norm.weight": (hidden_size,),  # Layer 23 attention norm weight
        "ffn_norm.weight": (hidden_size,),  # Layer 23 feedforward norm weight
        "attention.wqkv.weight": (int(q_dim + 2 * kv_dim), hidden_size),  # Combined QKV weight for attention
        "attention.wq.weight": (q_dim, hidden_size),  # Query weight for attention
        "attention.wk.weight": (kv_dim, hidden_size),  # Key weight for attention
        "attention.wv.weight": (kv_dim, hidden_size),  # Value weight for attention
        "attention.wo.weight": (hidden_size, q_dim),  # Output projection weight for attention
        "feed_forward.w1.weight": (mlp_hidden_features, hidden_size),  # MLP first layer weight
        "feed_forward.w3.weight": (mlp_hidden_features, hidden_size),  # MLP third layer weight
        "feed_forward.w2.weight": (hidden_size, mlp_hidden_features),  # MLP second layer weight
        "feed_forward.fused_w1_w3.weight": (mlp_hidden_features * 2, hidden_size),  # Fused first and third layer weight
        "norm.weight": (hidden_size,),  # Final normalization layer weight
        "tok_embeddings.weight": (vocab_size, hidden_size),  # Token embedding matrix
        "output.weight": (vocab_size, hidden_size),  # Output projection weight
        "embed_tokens.weight": (vocab_size, hidden_size), # Token embedding matrix of qwen2
        "attention.wq.bias": (q_dim,),
        "attention.wk.bias": (kv_dim,),
        "attention.wv.bias": (kv_dim,),
        "attention.wo.bias": (q_dim,),
        "attention.k_norm.weight": (attn_head_dim,),
        "attention.q_norm.weight": (attn_head_dim,),
        "feed_forward.moe_layer.gate.wg.weight": (attn_head_dim, hidden_size),
        "w1.weight": (mlp_ratio * hidden_size, hidden_size),
        "w2.weight": (hidden_size, mlp_ratio * hidden_size),
        "w3.weight": (mlp_ratio * hidden_size, hidden_size),
    }

    return param_shapes

CHECKPOINT_FOLDER = os.getenv("CHECKPOINT_FOLDER")
if CHECKPOINT_FOLDER is None:
    print(f"error, environment variable CHECKPOINT_FOLDER not set!")
    sys.exit(1)

ckpt_dict = {
    "INTERNLM_2_7B": "/data/deeplink_yidian/init_weight/ckpt_internlm2_7B_convert/model_tp0_pp0.pt",
    "QWEN_2_7B": "/data/deeplink_yidian/init_weight/ckpt_qwen2_7B_convert/model_tp0_pp0.pt", 
    "LLAMA_2_7B": "/data/deeplink_yidian/init_weight/ckpt_llama2_7B_convert/model_tp0_pp0.pt",
}

ckpt = dict(
    auto_resume=False,
    load_ckpt_path=ckpt_dict[model_type],
    save_ckpt_path=CHECKPOINT_FOLDER,
)
layer_chunks = get_chunks(NUM_LAYERS, NUM_PS)
param_shapes = get_param_shapes(model)
optimizer = dict(
    name="nesterov",
    lr=0.9,
    momentum=0.8,
)

grad = dict(
    use_ewma_outlier=False,
    ewma=dict(
        alpha=0.02,
        warmup_steps=5,
        base_threshold=3,
    ),
    use_weighted_avg=True,
    clip_pseudo_grad=1.0,
)