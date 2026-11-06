"""
Load balancer for distributing requests across multiple vLLM server instances.

Provides round-robin distribution for optimal GPU utilization when running
multiple vLLM instances across GPUs.
"""

import threading
from typing import List
from openai import AsyncOpenAI


class VLLMLoadBalancer:
    """
    Thread-safe round-robin load balancer for vLLM endpoints.
    
    Distributes requests evenly across available vLLM server instances
    to maximize throughput on multi-GPU systems.
    
    Usage:
        balancer = VLLMLoadBalancer([
            "http://localhost:8000/v1",
            "http://localhost:8001/v1",
        ])
        client = balancer.get_client()
        # Use client for OpenAI-compatible API calls
    """
    
    def __init__(self, endpoints: List[str]):
        """
        Initialize load balancer with vLLM endpoints.
        
        Args:
            endpoints: List of vLLM server URLs (e.g., ["http://localhost:8000/v1"])
        """
        if not endpoints:
            raise ValueError("At least one endpoint is required")
        
        self._endpoints = endpoints
        self._counter = 0
        self._lock = threading.Lock()
        
        # Pre-create clients for each endpoint (reuse connections)
        self._clients = [
            AsyncOpenAI(base_url=endpoint, api_key="not-needed")
            for endpoint in endpoints
        ]
    
    def get_client(self) -> AsyncOpenAI:
        """
        Get next client in round-robin rotation.
        
        Thread-safe: uses lock to ensure correct distribution
        across concurrent requests.
        
        Returns:
            AsyncOpenAI client configured for next vLLM endpoint
        """
        with self._lock:
            client = self._clients[self._counter % len(self._clients)]
            self._counter += 1
            return client
    
    def get_endpoint(self) -> str:
        """
        Get next endpoint URL in round-robin rotation.
        
        Returns:
            Next vLLM endpoint URL
        """
        with self._lock:
            endpoint = self._endpoints[self._counter % len(self._endpoints)]
            self._counter += 1
            return endpoint
    
    @property
    def endpoints(self) -> List[str]:
        """List of all configured endpoints."""
        return self._endpoints.copy()
    
    @property
    def num_endpoints(self) -> int:
        """Number of configured endpoints."""
        return len(self._endpoints)
    
    def reset(self) -> None:
        """Reset the round-robin counter to 0."""
        with self._lock:
            self._counter = 0

