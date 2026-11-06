"""
vLLM server lifecycle management.

Handles starting, stopping, and health checking vLLM server processes
across multiple GPUs.

Note: This module is optional — you can also start vLLM servers manually
via the launch_vllm.sh script and pass --vllm-endpoints to the relevant
script (collect_*_responses.py, run_full_judging.py, run_refusal_check.py).
"""

import asyncio
import subprocess
import signal
import os
import sys
from typing import List, Optional, Dict
from dataclasses import dataclass

import httpx

from .model_configs import get_model_config, ModelConfig


@dataclass
class VLLMServerInfo:
    """Information about a running vLLM server instance."""
    process: subprocess.Popen
    gpu_id: int
    port: int
    endpoint: str
    model_path: str


class VLLMServerManager:
    """
    Manages vLLM server processes across multiple GPUs.
    
    Usage:
        manager = VLLMServerManager()
        endpoints = await manager.start_servers(
            model_path="/models/hyperclovax",
            gpus=[0, 1, 2, 3],
            base_port=8000
        )
        # ... use endpoints for inference ...
        await manager.stop_servers()
    """
    
    def __init__(self):
        self._servers: List[VLLMServerInfo] = []
        self._shutdown_event = asyncio.Event()
    
    async def start_servers(
        self,
        model_path: str,
        gpus: List[int],
        base_port: int = 8000,
        model_name: Optional[str] = None,
        wait_for_ready: bool = True,
        ready_timeout: float = 300.0,
    ) -> List[str]:
        """
        Start vLLM server instances across specified GPUs.
        
        Args:
            model_path: Path to model directory or HuggingFace model ID
            gpus: List of GPU IDs to use (one server per GPU)
            base_port: Starting port number (increments per GPU)
            model_name: Optional model name for config lookup
            wait_for_ready: Whether to wait for servers to be healthy
            ready_timeout: Timeout for waiting for servers to be ready
            
        Returns:
            List of endpoint URLs (e.g., ["http://localhost:8000/v1", ...])
        """
        # Get model config for launch options
        config = get_model_config(model_name or model_path)
        
        endpoints = []
        for i, gpu_id in enumerate(gpus):
            port = base_port + i
            endpoint = f"http://localhost:{port}/v1"
            
            # Build vLLM command
            cmd = self._build_vllm_command(
                model_path=model_path,
                port=port,
                config=config,
            )
            
            # Set environment for specific GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            print(f"Starting vLLM server on GPU {gpu_id}, port {port}...")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            server_info = VLLMServerInfo(
                process=process,
                gpu_id=gpu_id,
                port=port,
                endpoint=endpoint,
                model_path=model_path,
            )
            self._servers.append(server_info)
            endpoints.append(endpoint)
        
        # Wait for all servers to be ready
        if wait_for_ready:
            await self._wait_for_servers_ready(endpoints, timeout=ready_timeout)
        
        return endpoints
    
    def _build_vllm_command(
        self,
        model_path: str,
        port: int,
        config: ModelConfig,
    ) -> List[str]:
        """Build vLLM serve command with appropriate options."""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--host", "0.0.0.0",
        ]
        
        # Add config-based options
        if config.gpu_memory_utilization:
            cmd.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
        
        if config.max_model_len:
            cmd.extend(["--max-model-len", str(config.max_model_len)])
        
        if config.dtype and config.dtype != "auto":
            cmd.extend(["--dtype", config.dtype])
        
        if config.chat_template:
            cmd.extend(["--chat-template", config.chat_template])
        
        # Add any extra launch args from config
        for key, value in config.extra_launch_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    async def _wait_for_servers_ready(
        self,
        endpoints: List[str],
        timeout: float = 300.0,
        poll_interval: float = 5.0,
    ) -> None:
        """Wait for all servers to respond to health checks."""
        start_time = asyncio.get_event_loop().time()
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Timeout waiting for vLLM servers to be ready after {timeout}s"
                    )
                
                # Check all endpoints
                ready_count = 0
                for endpoint in endpoints:
                    try:
                        # vLLM health endpoint
                        health_url = endpoint.replace("/v1", "/health")
                        resp = await client.get(health_url)
                        if resp.status_code == 200:
                            ready_count += 1
                    except Exception:
                        pass  # Not ready yet
                
                if ready_count == len(endpoints):
                    print(f"All {len(endpoints)} vLLM servers are ready!")
                    return
                
                print(f"Waiting for servers... {ready_count}/{len(endpoints)} ready")
                await asyncio.sleep(poll_interval)
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all running servers.
        
        Returns:
            Dict mapping endpoint URL to health status
        """
        results = {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            for server in self._servers:
                try:
                    health_url = server.endpoint.replace("/v1", "/health")
                    resp = await client.get(health_url)
                    results[server.endpoint] = resp.status_code == 200
                except Exception:
                    results[server.endpoint] = False
        return results
    
    async def stop_servers(self) -> None:
        """Stop all running vLLM server processes."""
        for server in self._servers:
            if server.process.poll() is None:  # Still running
                print(f"Stopping vLLM server on port {server.port}...")
                server.process.send_signal(signal.SIGTERM)
                try:
                    server.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.process.kill()
                    server.process.wait()
        
        self._servers.clear()
        print("All vLLM servers stopped.")
    
    def get_endpoints(self) -> List[str]:
        """Get list of all server endpoints."""
        return [s.endpoint for s in self._servers]
    
    @property
    def num_servers(self) -> int:
        """Number of running servers."""
        return len(self._servers)


# Convenience functions for simple use cases

async def start_vllm_servers(
    model_path: str,
    gpus: List[int],
    base_port: int = 8000,
    model_name: Optional[str] = None,
) -> List[str]:
    """
    Start vLLM servers and return endpoints.
    
    Convenience wrapper around VLLMServerManager.
    
    Args:
        model_path: Path to model or HuggingFace ID
        gpus: List of GPU IDs
        base_port: Starting port
        model_name: Optional name for config lookup
        
    Returns:
        List of endpoint URLs
    """
    manager = VLLMServerManager()
    return await manager.start_servers(
        model_path=model_path,
        gpus=gpus,
        base_port=base_port,
        model_name=model_name,
    )


async def check_endpoints_health(endpoints: List[str]) -> bool:
    """
    Check if all vLLM endpoints are healthy.
    
    Args:
        endpoints: List of vLLM endpoint URLs
        
    Returns:
        True if all endpoints are healthy
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in endpoints:
            try:
                health_url = endpoint.replace("/v1", "/health")
                resp = await client.get(health_url)
                if resp.status_code != 200:
                    return False
            except Exception:
                return False
    return True

