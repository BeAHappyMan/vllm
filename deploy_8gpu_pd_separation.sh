#!/bin/bash
# 8-GPU PD Separation Deployment Script with EP Communication
# P: 4 GPUs (0,1,2,3) for Prefill
# D: 4 GPUs (4,5,6,7) for Decode
# KV Cache: Layer-by-layer transmission

set -xe

echo "🚀 Starting 8-GPU PD Separation Deployment with EP Communication 🚀"
echo "📋 Configuration:"
echo "   - Prefill GPUs: 0,1,2,3 (Ports: 8100-8103)"
echo "   - Decode GPUs: 4,5,6,7 (Ports: 8200-8203)"
echo "   - Proxy Server: Port 8000"
echo "   - KV Cache: Layer-by-layer transmission via P2P NCCL"

# Configuration
MODEL_NAME=${MODEL_NAME:-"/app/model"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.8}
KV_BUFFER_SIZE=${KV_BUFFER_SIZE:-1.0}  # Layer-by-layer transmission
HOST_IP=${VLLM_HOST_IP:-$(hostname -I | awk '{print $1}')}

# Process tracking
PIDS=()

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    echo "✅ Cleanup complete"
}

trap cleanup EXIT INT TERM

# Wait for server function
wait_for_server() {
    local port=$1
    local timeout=${2:-120}
    echo "⏳ Waiting for server on port $port..."
    timeout $timeout bash -c "
        until curl -s localhost:${port}/health > /dev/null 2>&1 || curl -s localhost:${port}/v1/models > /dev/null 2>&1; do
            sleep 1
        done" && echo "✅ Server on port $port is ready" || {
        echo "❌ Server on port $port failed to start within ${timeout}s"
        return 1
    }
}

echo "🔧 Installing dependencies..."
pip install quart aiohttp uvicorn fastapi >/dev/null 2>&1 || true

echo ""
echo "🚀 Starting Prefill Nodes (4 GPUs)..."

# Prefill Node 0 (GPU 0, Rank 0)
echo "Starting Prefill Node 0 on GPU 0, Port 8100..."
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8100 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_rank": 0,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 5000,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8100,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/prefill_0.log 2>&1 &
PIDS+=($!)

# Prefill Node 1 (GPU 1, Rank 1)
echo "Starting Prefill Node 1 on GPU 1, Port 8101..."
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8101 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_rank": 1,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 5001,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8101,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/prefill_1.log 2>&1 &
PIDS+=($!)

# Prefill Node 2 (GPU 2, Rank 2)
echo "Starting Prefill Node 2 on GPU 2, Port 8102..."
CUDA_VISIBLE_DEVICES=2 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8102 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_rank": 2,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 5002,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8102,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/prefill_2.log 2>&1 &
PIDS+=($!)

# Prefill Node 3 (GPU 3, Rank 3)
echo "Starting Prefill Node 3 on GPU 3, Port 8103..."
CUDA_VISIBLE_DEVICES=3 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8103 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_rank": 3,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 5003,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8103,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/prefill_3.log 2>&1 &
PIDS+=($!)

echo ""
echo "🔄 Starting Decode Nodes (4 GPUs)..."

# Decode Node 0 (GPU 4, Rank 4)
echo "Starting Decode Node 0 on GPU 4, Port 8200..."
CUDA_VISIBLE_DEVICES=4 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8200 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_rank": 4,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 6000,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8200,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/decode_0.log 2>&1 &
PIDS+=($!)

# Decode Node 1 (GPU 5, Rank 5)
echo "Starting Decode Node 1 on GPU 5, Port 8201..."
CUDA_VISIBLE_DEVICES=5 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8201 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_rank": 5,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 6001,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8201,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/decode_1.log 2>&1 &
PIDS+=($!)

# Decode Node 2 (GPU 6, Rank 6)
echo "Starting Decode Node 2 on GPU 6, Port 8202..."
CUDA_VISIBLE_DEVICES=6 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8202 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_rank": 6,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 6002,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8202,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/decode_2.log 2>&1 &
PIDS+=($!)

# Decode Node 3 (GPU 7, Rank 7)
echo "Starting Decode Node 3 on GPU 7, Port 8203..."
CUDA_VISIBLE_DEVICES=7 \
UCX_NET_DEVICES=all \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
vllm serve "$MODEL_NAME" \
    --port 8203 \
    --host 0.0.0.0 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_rank": 7,
        "kv_parallel_size": 8,
        "kv_buffer_size": '$KV_BUFFER_SIZE',
        "kv_ip": "'$HOST_IP'",
        "kv_port": 6003,
        "kv_connector_extra_config": {
            "nccl_num_channels": "8",
            "send_type": "PUT_ASYNC",
            "mem_pool_size_gb": 16,
            "http_port": 8203,
            "proxy_ip": "'$HOST_IP'",
            "proxy_port": "8000"
        }
    }' \
    --max-num-seqs 64 \
    >/tmp/decode_3.log 2>&1 &
PIDS+=($!)

echo ""
echo "⏳ Waiting for all nodes to start..."

# Wait for all prefill nodes
for port in 8100 8101 8102 8103; do
    wait_for_server $port
done

# Wait for all decode nodes  
for port in 8200 8201 8202 8203; do
    wait_for_server $port
done

echo ""
echo "🌐 Starting Proxy Server on port 8000..."

# Start proxy server
python3 pd_proxy_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --prefill-servers "localhost:8100,localhost:8101,localhost:8102,localhost:8103" \
    --decode-servers "localhost:8200,localhost:8201,localhost:8202,localhost:8203" \
    >/tmp/proxy.log 2>&1 &
PIDS+=($!)

wait_for_server 8000

echo ""
echo "🎉 All services started successfully!"
echo ""
echo "📊 Service Status:"
echo "   Prefill Nodes: localhost:8100-8103 (GPUs 0-3)"
echo "   Decode Nodes:  localhost:8200-8203 (GPUs 4-7)"
echo "   Proxy Server:  localhost:8000"
echo ""
echo "📝 Log files:"
echo "   Prefill logs: /tmp/prefill_[0-3].log"
echo "   Decode logs:  /tmp/decode_[0-3].log"
echo "   Proxy log:    /tmp/proxy.log"
echo ""
echo "🚀 Ready to serve requests!"
echo ""

# Keep the script running
echo "Press Ctrl+C to stop all services..."
wait
