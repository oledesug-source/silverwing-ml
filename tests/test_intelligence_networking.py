"""Tests for enhanced intelligence.networking — rate limiter, retry, circuit breaker, middleware, connection pool."""

import time

from intelligence.networking.connection_pool import ConnectionPool
from intelligence.networking.http_client import HttpClient, HttpResponse
from intelligence.networking.load_balancer import Backend, LoadBalancer
from intelligence.networking.middleware import (
    AuthMiddleware,
    CorsMiddleware,
    LoggingMiddleware,
    MiddlewareContext,
    MiddlewarePipeline,
    RequestIdMiddleware,
)
from intelligence.networking.rate_limiter import (
    PerKeyRateLimiter,
    SlidingWindowLimiter,
    TokenBucket,
)
from intelligence.networking.retry import CircuitBreaker, CircuitState, RetryConfig, RetryExecutor
from intelligence.networking.server import Request, Response, Server


class TestHttpClient:
    def test_response_ok(self) -> None:
        resp = HttpResponse(status_code=200, body='{"ok": true}')
        assert resp.ok
        assert resp.json() == {"ok": True}

    def test_response_error(self) -> None:
        assert not HttpResponse(status_code=500, body="").ok
        assert not HttpResponse(status_code=404, body="").ok


class TestServer:
    def test_health_endpoint(self) -> None:
        server = Server(host="127.0.0.1", port=18800)
        server.start()
        time.sleep(0.1)
        client = HttpClient(base_url=server.url)
        resp = client.get("/health")
        assert resp.ok
        assert resp.json()["status"] == "ok"
        server.stop()

    def test_404(self) -> None:
        server = Server(host="127.0.0.1", port=18801)
        server.start()
        time.sleep(0.1)
        resp = HttpClient(base_url=server.url).get("/nope")
        assert resp.status_code == 404
        server.stop()


class TestLoadBalancer:
    def test_round_robin(self) -> None:
        lb = LoadBalancer(strategy="round_robin")
        lb.add_backend(Backend(url="http://b1:8000", name="b1"))
        lb.add_backend(Backend(url="http://b2:8000", name="b2"))
        assert lb.next_backend().name == "b1"
        assert lb.next_backend().name == "b2"
        assert lb.next_backend().name == "b1"

    def test_no_healthy(self) -> None:
        lb = LoadBalancer()
        lb.add_backend(Backend(url="http://b1:8000", healthy=False))
        try:
            lb.next_backend()
            assert False
        except RuntimeError:
            pass


class TestTokenBucket:
    def test_acquire(self) -> None:
        bucket = TokenBucket(rate=10, burst=5)
        for _ in range(5):
            assert bucket.acquire()
        assert not bucket.acquire()

    def test_refill(self) -> None:
        bucket = TokenBucket(rate=100, burst=1)
        bucket.acquire()
        time.sleep(0.02)
        assert bucket.acquire()


class TestSlidingWindowLimiter:
    def test_allowed(self) -> None:
        limiter = SlidingWindowLimiter(per_second=5, per_minute=100)
        result = limiter.check("key1")
        assert result.allowed

    def test_rate_limit(self) -> None:
        limiter = SlidingWindowLimiter(per_second=2, per_minute=100)
        limiter.check("k")
        limiter.check("k")
        result = limiter.check("k")
        assert not result.allowed

    def test_reset(self) -> None:
        limiter = SlidingWindowLimiter(per_second=1, per_minute=100)
        limiter.check("k")
        limiter.reset("k")
        result = limiter.check("k")
        assert result.allowed


class TestPerKeyRateLimiter:
    def test_default_limit(self) -> None:
        limiter = PerKeyRateLimiter(default_per_second=10)
        result = limiter.check("user1")
        assert result.allowed
        assert result.limit == 10

    def test_custom_limit(self) -> None:
        limiter = PerKeyRateLimiter(default_per_second=1)
        limiter.set_limit("premium", per_second=100)
        assert limiter.check("premium").limit == 100
        assert limiter.check("user1").limit == 1


class TestRetryExecutor:
    def test_success(self) -> None:
        executor = RetryExecutor(RetryConfig(max_retries=2))
        result = executor.execute(lambda: 42)
        assert result.success
        assert result.value == 42
        assert result.attempts == 1

    def test_retry_then_success(self) -> None:
        attempts = [0]
        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("transient")
            return "ok"

        executor = RetryExecutor(RetryConfig(max_retries=3, base_delay_ms=1, jitter=False))
        result = executor.execute(flaky)
        assert result.success
        assert result.value == "ok"
        assert result.attempts == 3

    def test_all_retries_fail(self) -> None:
        def always_fail():
            raise ValueError("permanent")

        executor = RetryExecutor(RetryConfig(max_retries=2, base_delay_ms=1, jitter=False))
        result = executor.execute(always_fail)
        assert not result.success
        assert result.attempts == 3


class TestCircuitBreaker:
    def test_closed_allows(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.allow_request()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_half_open_recovery(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_success_decrements_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 1


class TestConnectionPool:
    def test_acquire_release(self) -> None:
        counter = [0]
        def factory():
            counter[0] += 1
            return f"conn-{counter[0]}"

        pool = ConnectionPool(factory=factory, max_size=3)
        c1 = pool.acquire()
        c2 = pool.acquire()
        assert c1 != c2
        pool.release(c1)
        c3 = pool.acquire()
        assert c3.connection == c1.connection
        pool.close_all()

    def test_stats(self) -> None:
        pool = ConnectionPool(factory=lambda: "x", max_size=2)
        c = pool.acquire()
        stats = pool.stats
        assert stats["in_use"] == 1
        pool.release(c)
        pool.close_all()


class TestServerExtended:
    def test_custom_route_post(self) -> None:
        server = Server(host="127.0.0.1", port=18810)
        server.route("POST", "/predict")(lambda req: Response(body={"echo": req.json()}))
        server.start()
        time.sleep(0.1)
        resp = HttpClient(base_url=server.url).post("/predict", data={"x": 1})
        assert resp.ok
        assert resp.json()["echo"]["x"] == 1
        server.stop()

    def test_custom_route_put(self) -> None:
        server = Server(host="127.0.0.1", port=18811)
        server.route("PUT", "/update")(lambda req: Response(body={"updated": True}))
        server.start()
        time.sleep(0.1)
        resp = HttpClient(base_url=server.url).put("/update", data={"v": 2})
        assert resp.ok
        server.stop()

    def test_custom_route_delete(self) -> None:
        server = Server(host="127.0.0.1", port=18812)
        server.route("DELETE", "/remove")(lambda req: Response(status=204))
        server.start()
        time.sleep(0.1)
        resp = HttpClient(base_url=server.url).delete("/remove")
        assert resp.status_code == 204
        server.stop()

    def test_server_url(self) -> None:
        server = Server(host="127.0.0.1", port=9999)
        assert server.url == "http://127.0.0.1:9999"

    def test_handle_not_found(self) -> None:
        server = Server()
        resp = server.handle(Request(method="GET", path="/nope"))
        assert resp.status == 404

    def test_handle_custom_route(self) -> None:
        server = Server()
        server.route("GET", "/hi")(lambda req: Response(body={"msg": "hello"}))
        resp = server.handle(Request(method="GET", path="/hi"))
        assert resp.status == 200
        assert resp.to_json() == b'{"msg": "hello"}'


class TestLoadBalancerExtended:
    def test_least_connections(self) -> None:
        lb = LoadBalancer(strategy="least_connections")
        b1 = Backend(url="http://b1:8000", name="b1", active_connections=5)
        b2 = Backend(url="http://b2:8000", name="b2", active_connections=1)
        lb.add_backend(b1)
        lb.add_backend(b2)
        chosen = lb.next_backend()
        assert chosen.name == "b2"

    def test_weighted(self) -> None:
        lb = LoadBalancer(strategy="weighted")
        b1 = Backend(url="http://b1:8000", name="b1", weight=100)
        b2 = Backend(url="http://b2:8000", name="b2", weight=0)
        lb.add_backend(b1)
        lb.add_backend(b2)
        chosen = lb.next_backend()
        assert chosen.name == "b1"

    def test_remove_backend(self) -> None:
        lb = LoadBalancer()
        lb.add_backend(Backend(url="http://b1:8000"))
        lb.add_backend(Backend(url="http://b2:8000"))
        lb.remove_backend("http://b1:8000")
        chosen = lb.next_backend()
        assert chosen.url == "http://b2:8000"

    def test_health_check(self) -> None:
        lb = LoadBalancer(health_check_interval=0)
        b = Backend(url="http://b1:8000", total_requests=10, total_errors=0)
        lb.add_backend(b)
        results = lb.health_check()
        assert results["http://b1:8000"] is True

    def test_health_check_high_error_rate(self) -> None:
        lb = LoadBalancer(health_check_interval=0)
        b = Backend(url="http://b1:8000", total_requests=10, total_errors=8)
        lb.add_backend(b)
        results = lb.health_check()
        assert results["http://b1:8000"] is False

    def test_record_success_and_error(self) -> None:
        lb = LoadBalancer()
        b = Backend(url="http://b1:8000")
        lb.add_backend(b)
        chosen = lb.next_backend()
        lb.record_success(chosen)
        assert chosen.total_requests == 1
        chosen2 = lb.next_backend()
        lb.record_error(chosen2)
        assert chosen2.total_errors == 1

    def test_get_stats(self) -> None:
        lb = LoadBalancer()
        lb.add_backend(Backend(url="http://b1:8000", name="gpu1"))
        stats = lb.get_stats()
        assert len(stats) == 1
        assert stats[0]["name"] == "gpu1"

    def test_invalid_strategy(self) -> None:
        try:
            LoadBalancer(strategy="random_bad")
            assert False
        except ValueError:
            pass


class TestHttpClientExtended:
    def test_url_building_with_params(self) -> None:
        client = HttpClient(base_url="http://localhost:8000")
        url = client._build_url("/predict", params={"key": "val"})
        assert url == "http://localhost:8000/predict?key=val"

    def test_url_building_no_params(self) -> None:
        client = HttpClient(base_url="http://localhost:8000")
        url = client._build_url("/health")
        assert url == "http://localhost:8000/health"

    def test_base_url_strips_trailing_slash(self) -> None:
        client = HttpClient(base_url="http://localhost:8000/")
        assert client._base_url == "http://localhost:8000"


class TestConnectionPoolExtended:
    def test_max_age_expiry(self) -> None:
        counter = [0]
        def factory():
            counter[0] += 1
            return f"conn-{counter[0]}"
        pool = ConnectionPool(factory=factory, max_size=2, max_age_seconds=0.01)
        c = pool.acquire()
        pool.release(c)
        time.sleep(0.02)
        c2 = pool.acquire()
        assert c2.created_at > c.created_at
        pool.close_all()

    def test_validation_callback(self) -> None:
        counter = [0]
        alive = [True]
        def factory():
            counter[0] += 1
            return f"conn-{counter[0]}"
        pool = ConnectionPool(
            factory=factory,
            max_size=2,
            validate=lambda c: alive[0],
        )
        c = pool.acquire()
        pool.release(c)
        alive[0] = False
        c2 = pool.acquire()
        assert c2.created_at > c.created_at
        pool.close_all()

    def test_close_all(self) -> None:
        pool = ConnectionPool(factory=lambda: "conn", max_size=3)
        pool.acquire()
        pool.acquire()
        pool.close_all()
        stats = pool.stats
        assert stats["total"] == 0

    def test_stats_initial(self) -> None:
        pool = ConnectionPool(factory=lambda: "x", max_size=2)
        stats = pool.stats
        assert stats["created"] == 0
        assert stats["reused"] == 0
        assert stats["available"] == 0
        pool.close_all()


class TestMiddleware:
    def test_logging_middleware(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(LoggingMiddleware())
        ctx = pipeline.process(MiddlewareContext(method="GET", path="/test"))
        assert "elapsed_ms" in ctx.metadata
        assert "log" in ctx.metadata

    def test_cors_middleware(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(CorsMiddleware(allow_origins=["*"]))
        ctx = pipeline.process(MiddlewareContext(
            method="GET", path="/api",
            headers={"Origin": "http://example.com"},
        ))
        assert "Access-Control-Allow-Origin" in ctx.response_headers

    def test_cors_options_preflight(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(CorsMiddleware())
        ctx = pipeline.process(MiddlewareContext(
            method="OPTIONS", path="/api",
            headers={"Origin": "http://example.com"},
        ))
        assert ctx.status == 204
        assert ctx.stopped

    def test_auth_middleware(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(AuthMiddleware(verify_token=lambda t: t == "valid-token"))
        ctx = pipeline.process(MiddlewareContext(
            method="GET", path="/protected",
            headers={"Authorization": "Bearer invalid"},
        ))
        assert ctx.status == 401
        assert ctx.stopped

    def test_auth_skip_paths(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(AuthMiddleware(verify_token=lambda t: False))
        ctx = pipeline.process(MiddlewareContext(method="GET", path="/health"))
        assert not ctx.stopped

    def test_request_id_middleware(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(RequestIdMiddleware())
        ctx = pipeline.process(MiddlewareContext(method="GET", path="/"))
        assert "request_id" in ctx.metadata
        assert "X-Request-Id" in ctx.response_headers

    def test_pipeline_stops(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(AuthMiddleware(verify_token=lambda t: False))
        pipeline.add(LoggingMiddleware())

        def after(ctx):
            ctx.metadata["after_ran"] = True
            return ctx

        pipeline.add(after)

        ctx = pipeline.process(MiddlewareContext(method="GET", path="/secret"))
        assert ctx.status == 401
        assert ctx.stopped
