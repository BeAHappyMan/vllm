#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Request sender for testing 8-GPU PD separation deployment
Supports both streaming and non-streaming requests
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Any

import aiohttp


async def send_completion_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
) -> Dict[str, Any]:
    """Send a completion request"""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                return {
                    "error": f"HTTP {response.status}: {error_text}",
                    "prompt": prompt,
                    "duration": time.time() - start_time
                }
            
            if stream:
                # Handle streaming response
                generated_text = ""
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    generated_text += delta['content']
                        except json.JSONDecodeError:
                            continue
                
                return {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "duration": time.time() - start_time,
                    "stream": True
                }
            else:
                # Handle non-streaming response
                result = await response.json()
                generated_text = ""
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0].get('text', '')
                
                return {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "duration": time.time() - start_time,
                    "stream": False,
                    "usage": result.get('usage', {})
                }
    
    except Exception as e:
        return {
            "error": str(e),
            "prompt": prompt,
            "duration": time.time() - start_time
        }


async def send_chat_completion_request(
    session: aiohttp.ClientSession,
    url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
) -> Dict[str, Any]:
    """Send a chat completion request"""
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                return {
                    "error": f"HTTP {response.status}: {error_text}",
                    "messages": messages,
                    "duration": time.time() - start_time
                }
            
            if stream:
                # Handle streaming response
                generated_text = ""
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    generated_text += delta['content']
                        except json.JSONDecodeError:
                            continue
                
                return {
                    "messages": messages,
                    "generated_text": generated_text,
                    "duration": time.time() - start_time,
                    "stream": True
                }
            else:
                # Handle non-streaming response
                result = await response.json()
                generated_text = ""
                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0].get('message', {})
                    generated_text = message.get('content', '')
                
                return {
                    "messages": messages,
                    "generated_text": generated_text,
                    "duration": time.time() - start_time,
                    "stream": False,
                    "usage": result.get('usage', {})
                }
    
    except Exception as e:
        return {
            "error": str(e),
            "messages": messages,
            "duration": time.time() - start_time
        }


async def run_benchmark(
    base_url: str,
    num_requests: int,
    concurrency: int,
    request_type: str,
    max_tokens: int,
    temperature: float,
    stream: bool
):
    """Run benchmark with multiple concurrent requests"""
    
    # Sample prompts for testing
    prompts = [
        "Explain the concept of artificial intelligence in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the benefits of renewable energy sources?",
        "Describe the process of photosynthesis.",
        "How does machine learning differ from traditional programming?",
        "What are the main challenges in space exploration?",
        "Explain quantum computing in layman's terms.",
        "Write a poem about the beauty of nature.",
        "What is the importance of biodiversity?",
        "Describe the history of the internet."
    ]
    
    # Sample chat messages
    chat_messages = [
        [{"role": "user", "content": "Hello! Can you help me understand neural networks?"}],
        [{"role": "user", "content": "What's the difference between AI and machine learning?"}],
        [{"role": "user", "content": "Explain deep learning in simple terms."}],
        [{"role": "user", "content": "How do transformers work in NLP?"}],
        [{"role": "user", "content": "What are the applications of computer vision?"}],
    ]
    
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        
        for i in range(num_requests):
            if request_type == "completion":
                prompt = prompts[i % len(prompts)]
                url = f"{base_url}/v1/completions"
                task = send_completion_request(
                    session, url, prompt, max_tokens, temperature, stream
                )
            else:  # chat completion
                messages = chat_messages[i % len(chat_messages)]
                url = f"{base_url}/v1/chat/completions"
                task = send_chat_completion_request(
                    session, url, messages, max_tokens, temperature, stream
                )
            
            tasks.append(task)
            
            # Control concurrency
            if len(tasks) >= concurrency:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        print(f"❌ Error: {result}")
                    elif "error" in result:
                        print(f"❌ Request failed: {result['error']}")
                    else:
                        duration = result['duration']
                        text_preview = result['generated_text'][:100] + "..." if len(result['generated_text']) > 100 else result['generated_text']
                        print(f"✅ Request completed in {duration:.2f}s: {text_preview}")
                
                tasks = []
        
        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"❌ Error: {result}")
                elif "error" in result:
                    print(f"❌ Request failed: {result['error']}")
                else:
                    duration = result['duration']
                    text_preview = result['generated_text'][:100] + "..." if len(result['generated_text']) > 100 else result['generated_text']
                    print(f"✅ Request completed in {duration:.2f}s: {text_preview}")


async def test_single_request(base_url: str, request_type: str, stream: bool):
    """Test a single request"""
    
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if request_type == "completion":
            url = f"{base_url}/v1/completions"
            prompt = "Explain the concept of machine learning in simple terms."
            result = await send_completion_request(
                session, url, prompt, max_tokens=50, temperature=0.7, stream=stream
            )
        else:
            url = f"{base_url}/v1/chat/completions"
            messages = [{"role": "user", "content": "Hello! Tell me about artificial intelligence."}]
            result = await send_chat_completion_request(
                session, url, messages, max_tokens=50, temperature=0.7, stream=stream
            )
        
        if "error" in result:
            print(f"❌ Request failed: {result['error']}")
        else:
            print(f"✅ Request completed successfully!")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Generated text: {result['generated_text']}")
            if 'usage' in result:
                print(f"Usage: {result['usage']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="vLLM PD Separation Request Sender")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the proxy server"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to send"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent requests"
    )
    parser.add_argument(
        "--request-type",
        type=str,
        choices=["completion", "chat"],
        default="completion",
        help="Type of request to send"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Send only a single test request"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 vLLM PD Separation Request Sender")
    print(f"Target: {args.base_url}")
    print(f"Request type: {args.request_type}")
    print(f"Streaming: {args.stream}")
    
    if args.single:
        print("Sending single test request...")
        asyncio.run(test_single_request(args.base_url, args.request_type, args.stream))
    else:
        print(f"Sending {args.num_requests} requests with concurrency {args.concurrency}...")
        asyncio.run(run_benchmark(
            args.base_url,
            args.num_requests,
            args.concurrency,
            args.request_type,
            args.max_tokens,
            args.temperature,
            args.stream
        ))


if __name__ == "__main__":
    main()
