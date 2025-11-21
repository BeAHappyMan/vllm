#!/bin/bash
# Quick Start Script for 8-GPU PD Separation Deployment

set -e

echo "🚀 vLLM 8-GPU PD Separation Quick Start"
echo "========================================"

# Check if all required files exist
required_files=("deploy_8gpu_pd_separation.sh" "pd_proxy_server.py" "send_requests.py")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Error: Required file $file not found!"
        exit 1
    fi
done

# Make scripts executable
chmod +x deploy_8gpu_pd_separation.sh
chmod +x pd_proxy_server.py
chmod +x send_requests.py

echo "✅ All required files found and made executable"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $gpu_count GPUs"
    
    if [[ $gpu_count -lt 8 ]]; then
        echo "⚠️  Warning: Found only $gpu_count GPUs, but 8 are required for optimal performance"
        echo "   The script will still work but may need adjustment"
    else
        echo "✅ Sufficient GPUs available"
    fi
    
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | head -8
else
    echo "❌ nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo ""
echo "🔧 Installation Requirements:"
echo "   - Python packages: vllm, quart, aiohttp, uvicorn, fastapi"
echo "   - CUDA and NCCL libraries"
echo "   - Model: meta-llama/Meta-Llama-3.1-8B-Instruct (will be downloaded automatically)"

echo ""
read -p "Do you want to install Python dependencies? (y/N): " install_deps
if [[ $install_deps =~ ^[Yy]$ ]]; then
    echo "📦 Installing Python dependencies..."
    pip install vllm quart aiohttp uvicorn fastapi
    echo "✅ Dependencies installed"
fi

echo ""
echo "🚀 Deployment Options:"
echo "   1. Full deployment (all 8 GPUs)"
echo "   2. Test deployment (2 GPUs only)"
echo "   3. Show commands only (no execution)"
echo ""
read -p "Select option (1-3): " option

case $option in
    1)
        echo "🚀 Starting full 8-GPU deployment..."
        echo "This will start:"
        echo "   - 4 Prefill nodes (GPUs 0-3, ports 8100-8103)"
        echo "   - 4 Decode nodes (GPUs 4-7, ports 8200-8203)"
        echo "   - 1 Proxy server (port 8000)"
        echo ""
        echo "⚠️  This will take several minutes to start all services..."
        read -p "Continue? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            ./deploy_8gpu_pd_separation.sh
        fi
        ;;
    2)
        echo "🧪 Starting test deployment (2 GPUs)..."
        echo "This will start a minimal setup for testing:"
        echo "   - 1 Prefill node (GPU 0, port 8100)"
        echo "   - 1 Decode node (GPU 1, port 8200)"
        echo "   - 1 Proxy server (port 8000)"
        
        # Set environment variables for test deployment
        export MODEL_NAME="/app/model/"
        export MAX_MODEL_LEN=2048
        export GPU_MEMORY_UTIL=0.8
        export KV_BUFFER_SIZE=1.0
        export HOST_IP=$(hostname -I | awk '{print $1}')
        
        echo "Starting test deployment..."
        
        # Start prefill node
        echo "Starting Prefill node on GPU 0..."
        CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
            --port 8100 --host 0.0.0.0 --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEMORY_UTIL --trust-remote-code --enforce-eager \
            --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":1.0}' \
            >/tmp/test_prefill.log 2>&1 &
        
        # Start decode node
        echo "Starting Decode node on GPU 1..."
        CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_NAME" \
            --port 8200 --host 0.0.0.0 --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEMORY_UTIL --trust-remote-code --enforce-eager \
            --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":1.0}' \
            >/tmp/test_decode.log 2>&1 &
        
        # Wait for servers to start
        echo "Waiting for servers to start..."
        sleep 30
        
        # Start proxy
        echo "Starting Proxy server..."
        python3 pd_proxy_server.py \
            --host 0.0.0.0 --port 8000 \
            --prefill-servers "localhost:8100" \
            --decode-servers "localhost:8200" \
            >/tmp/test_proxy.log 2>&1 &
        
        sleep 10
        echo "✅ Test deployment started!"
        echo "   Prefill: localhost:8100"
        echo "   Decode: localhost:8200"
        echo "   Proxy: localhost:8000"
        echo ""
        echo "Test with: python3 send_requests.py --single"
        ;;
    3)
        echo "📋 Deployment Commands:"
        echo ""
        echo "1. Full 8-GPU deployment:"
        echo "   ./deploy_8gpu_pd_separation.sh"
        echo ""
        echo "2. Send test requests:"
        echo "   python3 send_requests.py --single"
        echo "   python3 send_requests.py --num-requests 10 --concurrency 4"
        echo ""
        echo "3. Monitor logs:"
        echo "   tail -f /tmp/prefill_*.log"
        echo "   tail -f /tmp/decode_*.log"
        echo "   tail -f /tmp/proxy.log"
        echo ""
        echo "4. Stop all services:"
        echo "   pkill -f vllm"
        echo "   pkill -f pd_proxy_server.py"
        ;;
    *)
        echo "Invalid option selected"
        exit 1
        ;;
esac

echo ""
echo "📚 Usage Examples:"
echo ""
echo "# Send a single test request"
echo "python3 send_requests.py --single"
echo ""
echo "# Send multiple requests"
echo "python3 send_requests.py --num-requests 20 --concurrency 8"
echo ""
echo "# Test streaming responses"
echo "python3 send_requests.py --single --stream"
echo ""
echo "# Test chat completions"
echo "python3 send_requests.py --single --request-type chat"
echo ""
echo "# Monitor system resources"
echo "watch -n 1 nvidia-smi"
echo ""
echo "# Check service health"
echo "curl http://localhost:8000/health"
echo ""
echo "🔧 Troubleshooting:"
echo "   - Check logs in /tmp/ directory"
echo "   - Ensure all GPUs are available: nvidia-smi"
echo "   - Verify ports are not in use: netstat -tlnp | grep 8[0-2][0-9][0-9]"
echo "   - Stop all services: pkill -f vllm && pkill -f pd_proxy_server.py"
echo ""
echo "✅ Setup complete! Happy inferencing! 🚀"
