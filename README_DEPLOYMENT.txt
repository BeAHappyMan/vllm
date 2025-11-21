vLLM 8-GPU PD Separation Deployment Guide
==========================================

This deployment provides Prefill-Decode (PD) separation with Expert Parallel (EP) communication
for optimal performance on 8 GPUs with layer-by-layer KV cache transmission.

📋 ARCHITECTURE
===============
- Prefill Nodes: 4 GPUs (0,1,2,3) → Ports 8100-8103
- Decode Nodes:  4 GPUs (4,5,6,7) → Ports 8200-8203  
- Proxy Server:  Load balancer → Port 8000
- KV Transfer:   P2P NCCL with layer-by-layer streaming

🚀 QUICK START
==============
1. Run the quick start script:
   ./quick_start.sh

2. Or manually deploy:
   ./deploy_8gpu_pd_separation.sh

3. Send test requests:
   python3 send_requests.py --single

📁 FILES INCLUDED
=================
- deploy_8gpu_pd_separation.sh : Main deployment script
- pd_proxy_server.py          : Load balancing proxy server
- send_requests.py            : Request testing utility
- quick_start.sh              : Interactive setup script
- README_DEPLOYMENT.txt       : This documentation

⚙️ CONFIGURATION
================
Environment Variables:
- MODEL_NAME        : Model to use (default: meta-llama/Meta-Llama-3.1-8B-Instruct)
- MAX_MODEL_LEN     : Maximum sequence length (default: 4096)
- GPU_MEMORY_UTIL   : GPU memory utilization (default: 0.8)
- KV_BUFFER_SIZE    : KV cache buffer size for layer-by-layer transfer (default: 1.0)
- VLLM_HOST_IP      : Host IP address (auto-detected)

Key Parameters:
- kv_parallel_size: 8 (total P+D nodes)
- kv_buffer_size: 1.0 (enables layer-by-layer transmission)
- enable_expert_parallel: true (EP communication)
- nccl_num_channels: 8 (NCCL optimization)

🔧 MANUAL DEPLOYMENT
====================
1. Start Prefill Nodes (4 GPUs):
   
   # GPU 0 - Prefill Node 0
   CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
     --port 8100 --host 0.0.0.0 --max-model-len 4096 \
     --gpu-memory-utilization 0.8 --enable-expert-parallel \
     --trust-remote-code --enforce-eager \
     --kv-transfer-config '{
       "kv_connector": "P2pNcclConnector",
       "kv_role": "kv_producer", 
       "kv_rank": 0,
       "kv_parallel_size": 8,
       "kv_buffer_size": 1.0,
       "kv_connector_extra_config": {
         "nccl_num_channels": "8",
         "send_type": "PUT_ASYNC"
       }
     }' &

   # Repeat for GPUs 1,2,3 with ranks 1,2,3 and ports 8101,8102,8103

2. Start Decode Nodes (4 GPUs):
   
   # GPU 4 - Decode Node 0  
   CUDA_VISIBLE_DEVICES=4 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
     --port 8200 --host 0.0.0.0 --max-model-len 4096 \
     --gpu-memory-utilization 0.8 --enable-expert-parallel \
     --trust-remote-code --enforce-eager \
     --kv-transfer-config '{
       "kv_connector": "P2pNcclConnector",
       "kv_role": "kv_consumer",
       "kv_rank": 4, 
       "kv_parallel_size": 8,
       "kv_buffer_size": 1.0,
       "kv_connector_extra_config": {
         "nccl_num_channels": "8",
         "send_type": "PUT_ASYNC"
       }
     }' &

   # Repeat for GPUs 5,6,7 with ranks 5,6,7 and ports 8201,8202,8203

3. Start Proxy Server:
   python3 pd_proxy_server.py \
     --host 0.0.0.0 --port 8000 \
     --prefill-servers "localhost:8100,localhost:8101,localhost:8102,localhost:8103" \
     --decode-servers "localhost:8200,localhost:8201,localhost:8202,localhost:8203" &

📊 TESTING & MONITORING
=======================
1. Health Check:
   curl http://localhost:8000/health

2. Single Request Test:
   python3 send_requests.py --single

3. Load Testing:
   python3 send_requests.py --num-requests 50 --concurrency 10

4. Streaming Test:
   python3 send_requests.py --single --stream

5. Chat Completion Test:
   python3 send_requests.py --single --request-type chat

6. Monitor GPU Usage:
   watch -n 1 nvidia-smi

7. Monitor Logs:
   tail -f /tmp/prefill_*.log
   tail -f /tmp/decode_*.log  
   tail -f /tmp/proxy.log

🔍 TROUBLESHOOTING
==================
Common Issues:

1. "Port already in use":
   - Check: netstat -tlnp | grep 8[0-2][0-9][0-9]
   - Fix: Kill existing processes or change ports

2. "CUDA out of memory":
   - Reduce GPU_MEMORY_UTIL (e.g., 0.6 instead of 0.8)
   - Reduce MAX_MODEL_LEN
   - Ensure no other processes using GPUs

3. "NCCL initialization failed":
   - Check GPU topology: nvidia-smi topo -m
   - Verify NCCL installation
   - Try setting NCCL_IB_DISABLE=1

4. "Connection refused":
   - Wait longer for services to start (30-60 seconds)
   - Check firewall settings
   - Verify host IP configuration

5. "Model download failed":
   - Check internet connection
   - Verify Hugging Face access
   - Set HF_TOKEN if using gated models

Log Locations:
- Prefill nodes: /tmp/prefill_[0-3].log
- Decode nodes: /tmp/decode_[0-3].log
- Proxy server: /tmp/proxy.log

Stop All Services:
pkill -f vllm && pkill -f pd_proxy_server.py

🎯 PERFORMANCE TUNING
=====================
1. KV Cache Layer-by-Layer Transfer:
   - kv_buffer_size: 1.0 (immediate transfer after each layer)
   - Lower values = more frequent transfers, higher bandwidth usage
   - Higher values = less frequent transfers, higher memory usage

2. NCCL Optimization:
   - nccl_num_channels: 8 (adjust based on network topology)
   - NCCL_ALGO=RING or NCCL_ALGO=TREE
   - NCCL_PROTO=LL128 for high bandwidth

3. Memory Optimization:
   - gpu_memory_utilization: 0.8 (adjust based on model size)
   - mem_pool_size_gb: 16 (adjust based on available memory)

4. Expert Parallel (EP):
   - Enabled by default with --enable-expert-parallel
   - Optimizes MoE model communication
   - Reduces inter-GPU communication overhead

📈 EXPECTED PERFORMANCE
======================
With proper configuration, you should see:
- High GPU utilization (>90%) across all 8 GPUs
- Low latency prefill phase
- High throughput decode phase  
- Efficient KV cache transfer between P and D nodes
- Load balancing across multiple nodes

🔗 API ENDPOINTS
================
Proxy Server (localhost:8000):
- GET  /health                 : Health check
- GET  /v1/models             : List available models
- POST /v1/completions        : Text completion
- POST /v1/chat/completions   : Chat completion

Direct Node Access:
- Prefill nodes: localhost:8100-8103
- Decode nodes: localhost:8200-8203

📚 REFERENCES
=============
- vLLM Documentation: https://docs.vllm.ai
- PD Separation Paper: https://arxiv.org/abs/2309.06180
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- P2P Communication: https://developer.nvidia.com/gpudirect

🎉 SUCCESS INDICATORS
=====================
✅ All 8 GPU processes started successfully
✅ Proxy server responds to health checks  
✅ KV cache transfer working between P and D nodes
✅ Load balancing distributing requests evenly
✅ High GPU utilization with low memory usage
✅ Fast response times for both prefill and decode

Happy inferencing! 🚀
