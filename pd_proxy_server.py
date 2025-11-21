#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PD Separation Proxy Server for 8-GPU deployment
Handles load balancing between 4 Prefill nodes and 4 Decode nodes
"""

import argparse
import asyncio
import json
import logging
import os
import random
import uuid
from typing import Any, Dict, List

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM PD Separation Proxy", version="1.0.0")

# Global state
prefill_servers: List[str] = []
decode_servers: List[str] = []
request_counter = 0

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)  # 6 hours


def generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    return f"pd_proxy_{uuid.uuid4().hex}"


async def forward_request(
    url: str, 
    data: Dict[str, Any], 
    request_id: str,
    stream: bool = False
) -> Any:
    """Forward request to target server"""
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
    }
    
    # Add authorization if available
    if "OPENAI_API_KEY" in os.environ:
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    
    logger.info(f"Forwarding request {request_id} to {url}")
    
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Request {request_id} failed: {response.status} - {error_text}")
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Upstream server error: {error_text}"
                )
            
            if stream:
                async for chunk in response.content.iter_chunked(1024):
                    yield chunk
            else:
                content = await response.read()
                yield content


def select_server(servers: List[str]) -> str:
    """Select server using round-robin load balancing"""
    global request_counter
    if not servers:
        raise HTTPException(status_code=503, detail="No servers available")
    
    server = servers[request_counter % len(servers)]
    request_counter += 1
    return server


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "prefill_servers": len(prefill_servers),
        "decode_servers": len(decode_servers),
        "total_requests": request_counter
    }


@app.get("/v1/models")
async def list_models():
    """List available models by forwarding to a prefill server"""
    if not prefill_servers:
        raise HTTPException(status_code=503, detail="No prefill servers available")
    
    server = prefill_servers[0]
    url = f"http://{server}/v1/models"
    
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(status_code=response.status, detail="Failed to get models")


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def handle_completion(request: Request):
    """Handle completion requests with PD separation"""
    try:
        # Parse request
        original_data = await request.json()
        request_id = generate_request_id()
        
        logger.info(f"Processing request {request_id}")
        
        # Select servers
        prefill_server = select_server(prefill_servers)
        decode_server = select_server(decode_servers)
        
        logger.info(f"Request {request_id}: Prefill -> {prefill_server}, Decode -> {decode_server}")
        
        # Prepare prefill request (limit to 1 token for prefill only)
        prefill_data = original_data.copy()
        prefill_data["max_tokens"] = 1
        if "max_completion_tokens" in prefill_data:
            prefill_data["max_completion_tokens"] = 1
        
        # Step 1: Send to prefill server
        prefill_url = f"http://{prefill_server}{request.url.path}"
        
        # Execute prefill (consume all chunks but don't return them)
        async for _ in forward_request(prefill_url, prefill_data, request_id):
            pass
        
        logger.info(f"Request {request_id}: Prefill completed, starting decode")
        
        # Step 2: Send to decode server with original request
        decode_url = f"http://{decode_server}{request.url.path}"
        
        # Check if streaming is requested
        stream = original_data.get("stream", False)
        
        if stream:
            # Return streaming response
            return StreamingResponse(
                forward_request(decode_url, original_data, request_id, stream=True),
                media_type="text/plain"
            )
        else:
            # Return non-streaming response
            response_chunks = []
            async for chunk in forward_request(decode_url, original_data, request_id):
                response_chunks.append(chunk)
            
            # Combine all chunks and parse JSON
            response_data = b''.join(response_chunks)
            return json.loads(response_data.decode('utf-8'))
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="vLLM PD Separation Proxy Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--prefill-servers",
        type=str,
        required=True,
        help="Comma-separated list of prefill server addresses (host:port)"
    )
    parser.add_argument(
        "--decode-servers",
        type=str,
        required=True,
        help="Comma-separated list of decode server addresses (host:port)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    global prefill_servers, decode_servers
    
    args = parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse server lists
    prefill_servers = [s.strip() for s in args.prefill_servers.split(",") if s.strip()]
    decode_servers = [s.strip() for s in args.decode_servers.split(",") if s.strip()]
    
    if not prefill_servers:
        raise ValueError("At least one prefill server must be specified")
    if not decode_servers:
        raise ValueError("At least one decode server must be specified")
    
    logger.info(f"Starting PD Separation Proxy Server")
    logger.info(f"Prefill servers: {prefill_servers}")
    logger.info(f"Decode servers: {decode_servers}")
    logger.info(f"Listening on {args.host}:{args.port}")
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
