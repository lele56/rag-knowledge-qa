# evaluation/metrics/performance.py
"""性能指标统计"""
import statistics
from typing import List, Dict


class PerformanceMetrics:
    """性能指标统计"""

    @staticmethod
    def summarize(latencies_ms: List[float]) -> Dict[str, float]:
        """计算延迟分布统计"""
        if not latencies_ms:
            return {}
        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)
        return {
            "count": n,
            "mean_ms": round(statistics.mean(sorted_lat), 1),
            "median_ms": round(statistics.median(sorted_lat), 1),
            "p50_ms": round(sorted_lat[int(n * 0.50)] if n > 1 else sorted_lat[0], 1),
            "p95_ms": round(sorted_lat[min(int(n * 0.95), n - 1)], 1),
            "p99_ms": round(sorted_lat[min(int(n * 0.99), n - 1)], 1),
            "min_ms": round(sorted_lat[0], 1),
            "max_ms": round(sorted_lat[-1], 1),
            "total_ms": round(sum(sorted_lat), 1),
        }

    @staticmethod
    def throughput(total_queries: int, total_time_ms: float) -> float:
        """每秒查询数 (QPS)"""
        if total_time_ms <= 0:
            return 0.0
        return total_queries / (total_time_ms / 1000.0)