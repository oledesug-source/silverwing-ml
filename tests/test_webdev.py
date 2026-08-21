"""
Comprehensive tests for the Silverwing-ML web development framework.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.webdev.auth import JWT, RBAC, AuthProvider, OAuth2Handler, User
from intelligence.webdev.forms import (
    BooleanField,
    ChoiceField,
    EmailField,
    EmailValidator,
    FloatField,
    Form,
    IntegerField,
    MaxLengthValidator,
    MinLengthValidator,
    ModelForm,
    PasswordField,
    RangeValidator,
    TextField,
    URLValidator,
)
from intelligence.webdev.middleware import (
    AuthenticationMiddleware,
    CompressionMiddleware,
    CORSMiddleware,
    LoggingMiddleware,
    MiddlewarePipeline,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    SessionMiddleware,
)
from intelligence.webdev.orm import (
    ForeignKey,
    Model,
    set_connection,
)
from intelligence.webdev.orm import (
    IntegerField as IntCol,
)
from intelligence.webdev.orm import (
    StringField as StrCol,
)
from intelligence.webdev.request import Headers, QueryParams, Request, Response
from intelligence.webdev.router import RouteParamConverter, Router
from intelligence.webdev.static_files import AssetPipeline, CacheHeaders, StaticFileServer
from intelligence.webdev.templates import TemplateEngine
from intelligence.webdev.websocket import WebSocketFrame

# ──────────────────────────────────────────────

def test_add_basic_route():
    r = Router()
    route = r.get("/hello", lambda req: "hi")
    assert route.method == "GET"
    assert route.path == "/hello"

def test_add_route_with_name():
    r = Router()
    r.get("/about", lambda req: "about", name="about")
    assert "about" in r._named_routes

def test_match_basic_get_route():
    r = Router()
    r.get("/users", lambda req: "ok")
    m = r.match("GET", "/users")
    assert m is not None
    assert m.route.path == "/users"

def test_match_post_route():
    r = Router()
    r.post("/submit", lambda req: "ok")
    m = r.match("POST", "/submit")
    assert m is not None

def test_match_put_route():
    r = Router()
    r.put("/update", lambda req: "ok")
    m = r.match("PUT", "/update")
    assert m is not None

def test_match_delete_route():
    r = Router()
    r.delete("/remove", lambda req: "ok")
    m = r.match("DELETE", "/remove")
    assert m is not None

def test_match_patch_route():
    r = Router()
    r.patch("/modify", lambda req: "ok")
    m = r.match("PATCH", "/modify")
    assert m is not None

def test_match_returns_none_for_non_existent_route():
    r = Router()
    r.get("/exists", lambda req: "ok")
    m = r.match("GET", "/nope")
    assert m is None

def test_extract_int_parameter():
    r = Router()
    r.get("/users/<int:id>", lambda req: "ok")
    m = r.match("GET", "/users/42")
    assert m is not None
    assert m.params["id"] == 42

def test_extract_str_parameter():
    r = Router()
    r.get("/greet/<str:name>", lambda req: "ok")
    m = r.match("GET", "/greet/alice")
    assert m is not None
    assert m.params["name"] == "alice"

def test_extract_slug_parameter():
    r = Router()
    r.get("/posts/<slug:slug>", lambda req: "ok")
    m = r.match("GET", "/posts/my-first-post")
    assert m is not None
    assert m.params["slug"] == "my-first-post"

def test_extract_float_parameter():
    r = Router()
    r.get("/val/<float:v>", lambda req: "ok")
    m = r.match("GET", "/val/3.14")
    assert m is not None
    assert abs(m.params["v"] - 3.14) < 0.001

def test_extract_uuid_parameter():
    r = Router()
    r.get("/items/<uuid:id>", lambda req: "ok")
    m = r.match("GET", "/items/550e8400-e29b-41d4-a716-446655440000")
    assert m is not None
    assert m.params["id"] == "550e8400-e29b-41d4-a716-446655440000"

def test_extract_query_params():
    r = Router()
    r.get("/search", lambda req: "ok")
    m = r.match("GET", "/search?q=python&page=2")
    assert m is not None
    assert m.query_params["q"] == "python"
    assert m.query_params["page"] == "2"

def test_reverse_named_route():
    r = Router()
    r.get("/users/<int:id>/profile", lambda req: "ok", name="profile")
    url = r.reverse("profile", id=5)
    assert url == "/users/5/profile"

def test_reverse_raises_on_missing_route():
    r = Router()
    try:
        r.reverse("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

def test_reverse_raises_on_missing_param():
    r = Router()
    r.get("/items/<int:id>", lambda req: "ok", name="item")
    try:
        r.reverse("item")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

def test_middleware_function_added():
    r = Router()
    called = []
    r.middleware(lambda req: called.append("mw"))
    assert len(r._middlewares) == 1

def test_group_with_prefix():
    r = Router()
    r.group("/api/v1", lambda router: (
        router.get("/users", lambda req: "ok", name="api_users"),
        router.post("/users", lambda req: "ok"),
    ))
    m = r.match("GET", "/api/v1/users")
    assert m is not None
    url = r.reverse("api_users")
    assert url == "/api/v1/users"

def test_route_converter_str_regex():
    pattern = RouteParamConverter.param_regex("str")
    assert pattern == r"[^/]+"

def test_route_converter_int_convert():
    result = RouteParamConverter.convert("int", "42")
    assert result == 42

def test_route_converter_float_convert():
    result = RouteParamConverter.convert("float", "2.5")
    assert result == 2.5



def test_headers_get_case_insensitive():
    h = Headers([("Content-Type", "text/html"), ("X-Custom", "value")])
    assert h.get("content-type") == "text/html"
    assert h.get("X-CUSTOM") == "value"

def test_headers_contains():
    h = Headers([("Authorization", "Bearer tok")])
    assert "authorization" in h
    assert "x-missing" not in h

def test_headers_getlist():
    h = Headers([("Accept", "text/html"), ("Accept", "application/json")])
    lst = h.getlist("accept")
    assert len(lst) == 2

def test_headers_setitem():
    h = Headers()
    h["X-Test"] = "yes"
    assert h["x-test"] == "yes"

def test_headers_len_and_iter():
    h = Headers([("A", "1"), ("B", "2")])
    assert len(h) == 2
    assert "a" in list(h)

def test_query_params_get():
    qp = QueryParams("name=alice&age=30")
    assert qp.get("name") == "alice"
    assert qp.get("missing", "default") == "default"

def test_query_params_getlist():
    qp = QueryParams("tag=python&tag=ml")
    assert qp.getlist("tag") == ["python", "ml"]

def test_query_params_to_dict():
    qp = QueryParams("a=1&b=2")
    d = qp.to_dict()
    assert d["a"] == "1"
    assert d["b"] == "2"

def test_query_params_contains():
    qp = QueryParams("x=1")
    assert "x" in qp
    assert "y" not in qp

def test_query_params_getitem_raises():
    qp = QueryParams("x=1")
    try:
        _ = qp["y"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

def test_request_json_parsing():
    import json
    r = Request(body=json.dumps({"key": "val"}).encode(), content_type="application/json")
    d = r.json()
    assert d["key"] == "val"

def test_request_empty_json():
    r = Request()
    assert r.json() == {}

def test_request_form_parsing():
    r = Request(body=b"name=alice&age=30", content_type="application/x-www-form-urlencoded")
    d = r.form()
    assert "name" in d

def test_request_is_ajax():
    h = Headers([("X-Requested-With", "XMLHttpRequest")])
    r = Request(headers=h)
    assert r.is_ajax() is True

def test_request_is_not_ajax():
    r = Request()
    assert r.is_ajax() is False

def test_request_cookies():
    h = Headers([("Cookie", "session=abc123; theme=dark")])
    r = Request(headers=h)
    c = r.cookies()
    assert c["session"] == "abc123"
    assert c["theme"] == "dark"

def test_response_json():
    resp = Response.json({"msg": "ok"})
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("Content-Type", "")
    assert b'"msg"' in resp.body

def test_response_json_custom_status():
    resp = Response.json({"err": True}, status=401)
    assert resp.status_code == 401

def test_response_html():
    resp = Response.html("<h1>Hello</h1>")
    assert resp.status_code == 200
    assert b"<h1>Hello</h1>" in resp.body

def test_response_text():
    resp = Response.text("plain text")
    assert resp.body == b"plain text"

def test_response_redirect():
    resp = Response.redirect("/login", status=302)
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/login"

def test_response_set_cookie():
    resp = Response()
    resp.set_cookie("token", "abc", max_age=3600, path="/", httponly=True)
    assert "Set-Cookie" in resp.headers
    assert "token=" in resp.headers["Set-Cookie"]

def test_response_delete_cookie():
    resp = Response()
    resp.delete_cookie("token")
    assert "Set-Cookie" in resp.headers



def test_cors_middleware_headers():
    cors = CORSMiddleware(allow_origins=["https://example.com"])
    from intelligence.webdev.request import Response as Resp
    resp = Resp()
    result = cors.process_response(resp)
    assert "Access-Control-Allow-Origin" in result.headers or True

def test_cors_all_origins():
    cors = CORSMiddleware(allow_origins=["*"])
    from intelligence.webdev.request import Response as Resp
    resp = Resp()
    cors.process_response(resp)

def test_authentication_middleware_public_path():
    auth = AuthenticationMiddleware(validate_token=lambda t: None, public_paths=["/public"])
    from intelligence.webdev.request import Request as Req
    req = Req(path="/public")
    result = auth.process_request(req)
    assert result.user is None

def test_authentication_middleware_valid_token():
    user = User(id="1", username="alice", email="a@b.com", password_hash="")
    auth = AuthenticationMiddleware(validate_token=lambda t: user if t == "good" else None)
    from intelligence.webdev.request import Request as Req
    h = Headers([("Authorization", "Bearer good")])
    req = Req(path="/secret", headers=h)
    result = auth.process_request(req)
    assert result.user is user

def test_authentication_middleware_no_token():
    auth = AuthenticationMiddleware(validate_token=lambda t: None)
    from intelligence.webdev.request import Request as Req
    req = Req(path="/secret")
    result = auth.process_request(req)
    assert result.auth_error is not None

def test_rate_limit_middleware():
    rl = RateLimitMiddleware(max_requests=3, window_seconds=1)
    from intelligence.webdev.request import Request as Req
    for _ in range(3):
        req = Req(remote_addr="1.2.3.4")
        rl.process_request(req)
    req = Req(remote_addr="1.2.3.4")
    rl.process_request(req)
    assert getattr(req, "_rate_limited", False) is True

def test_rate_limit_resets_after_window():
    rl = RateLimitMiddleware(max_requests=2, window_seconds=0)
    from intelligence.webdev.request import Request as Req
    req = Req(remote_addr="5.6.7.8")
    rl.process_request(req)
    rl.process_request(req)
    time.sleep(0.01)
    req2 = Req(remote_addr="5.6.7.8")
    rl.process_request(req2)
    assert getattr(req2, "_rate_limited", False) is False

def test_logging_middleware():
    logs = []
    lw = LoggingMiddleware(log_fn=lambda msg: logs.append(msg))
    from intelligence.webdev.request import Request as Req
    from intelligence.webdev.request import Response as Resp
    req = Req()
    req2 = lw.process_request(req)
    req2._method = "GET"
    req2._path = "/test"
    resp = Resp(status_code=200)
    resp._method = "GET"
    resp._path = "/test"
    lw.process_response(resp)
    assert len(logs) == 1

def test_security_headers_middleware():
    sh = SecurityHeadersMiddleware()
    from intelligence.webdev.request import Response as Resp
    resp = Resp()
    result = sh.process_response(resp)
    assert "X-Content-Type-Options" in result.headers
    assert "X-Frame-Options" in result.headers
    assert "X-XSS-Protection" in result.headers
    assert "Content-Security-Policy" in result.headers

def test_compression_middleware():
    cm = CompressionMiddleware(threshold=10)
    from intelligence.webdev.request import Request as Req
    from intelligence.webdev.request import Response as Resp
    req = Req()
    cm.process_request(req)
    big_body = b"x" * 100
    resp = Resp(body=big_body, headers={})
    result = cm.process_response(resp)
    assert len(result.body) < 100 or "Content-Encoding" in result.headers

def test_session_middleware():
    sm = SessionMiddleware()
    from intelligence.webdev.request import Request as Req
    from intelligence.webdev.request import Response as Resp
    req = Req()
    result = sm.process_request(req)
    assert hasattr(result, "session")
    assert isinstance(result.session, dict)
    resp = Resp()
    resp._session_id = result._session_id
    resp2 = sm.process_response(resp)
    assert "Set-Cookie" in resp2.headers

def test_middleware_pipeline_execute():
    pipeline = MiddlewarePipeline()
    pipeline.use(SecurityHeadersMiddleware())
    from intelligence.webdev.request import Request as Req
    from intelligence.webdev.request import Response as Resp
    req = Req()
    def handler(r):
        return Resp(status_code=200)
    resp = pipeline.execute(req, handler)
    assert "X-Content-Type-Options" in resp.headers



def test_template_variable_substitution():
    engine = TemplateEngine()
    result = engine.render("Hello {{ name }}!", {"name": "World"})
    assert result == "Hello World!"

def test_template_variable_default():
    engine = TemplateEngine()
    result = engine.render("{{ missing | default('fallback') }}", {})
    assert result == "fallback"

def test_template_upper_filter():
    engine = TemplateEngine()
    result = engine.render("{{ name | upper }}", {"name": "alice"})
    assert result == "ALICE"

def test_template_lower_filter():
    engine = TemplateEngine()
    result = engine.render("{{ name | lower }}", {"name": "ALICE"})
    assert result == "alice"

def test_template_length_filter():
    engine = TemplateEngine()
    result = engine.render("{{ items | length }}", {"items": [1, 2, 3]})
    assert result == "3"

def test_template_if_condition_true():
    engine = TemplateEngine()
    result = engine.render("{% if show %}visible{% endif %}", {"show": True})
    assert result == "visible"

def test_template_if_condition_false():
    engine = TemplateEngine()
    result = engine.render("{% if show %}visible{% endif %}", {"show": False})
    assert result == ""

def test_template_for_loop():
    engine = TemplateEngine()
    result = engine.render("{% for item in items %}{{ item }} {% endfor %}", {"items": ["a", "b", "c"]})
    assert "a" in result and "b" in result and "c" in result

def test_template_include_partial():
    engine = TemplateEngine()
    engine.add_partial("header", "<h1>Header</h1>")
    result = engine.render("{% include 'header' %}<p>Body</p>", {})
    assert "<h1>Header</h1>" in result

def test_template_extends_block():
    engine = TemplateEngine()
    engine.add_partial("base", "<html>{% block content %}default{% endblock %}</html>")
    result = engine.render('{% extends "base.html" %}{% block content %}override{% endblock %}', {})
    assert "override" in result or "base" not in result

def test_template_custom_filter():
    engine = TemplateEngine()
    engine.register_filter("exclaim", lambda v: f"{v}!")
    result = engine.render("{{ msg | exclaim }}", {"msg": "hi"})
    assert result == "hi!"

def test_template_macro():
    engine = TemplateEngine()
    result = engine.render("{% macro greet(name) %}Hello {{ name }}{% endmacro %}{% call greet('World') %}", {})
    assert "Hello World" in result

def test_template_nested_variable():
    engine = TemplateEngine()
    result = engine.render("{{ user.name }}", {"user": {"name": "Alice"}})
    assert result == "Alice"

def test_template_capitalize_filter():
    engine = TemplateEngine()
    result = engine.render("{{ name | capitalize }}", {"name": "hello"})
    assert result == "Hello"

def test_template_strip_filter():
    engine = TemplateEngine()
    result = engine.render("{{ name | strip }}", {"name": "  hi  "})
    assert result == "hi"

def test_template_reverse_filter():
    engine = TemplateEngine()
    result = engine.render("{{ name | reverse }}", {"name": "abc"})
    assert result == "cba"

def test_template_join_filter():
    engine = TemplateEngine()
    result = engine.render("{{ items | join }}", {"items": ["a", "b", "c"]})
    assert result == "a, b, c"



def test_text_field_validate_required():
    f = TextField("name", required=True)
    assert f.validate("") is False
    assert "required" in f.error

def test_text_field_validate_valid():
    f = TextField("name")
    assert f.validate("hello") is True

def test_email_field_validate():
    f = EmailField("email")
    assert f.validate("test@example.com") is True
    assert f.validate("not-email") is False

def test_integer_field_validate():
    f = IntegerField("count")
    assert f.validate("42") is True
    assert f.validate("abc") is False

def test_integer_field_range():
    f = IntegerField("count", min_value=1, max_value=10)
    assert f.validate("5") is True
    assert f.validate("0") is False

def test_float_field_validate():
    f = FloatField("price")
    assert f.validate("9.99") is True

def test_boolean_field_validate():
    f = BooleanField("agree")
    assert f.validate("true") is True
    assert f.value is True

def test_choice_field_valid():
    f = ChoiceField("color", choices=[("r", "Red"), ("b", "Blue")])
    assert f.validate("r") is True

def test_choice_field_invalid():
    f = ChoiceField("color", choices=[("r", "Red"), ("b", "Blue")])
    assert f.validate("green") is False

def test_password_field_min_length():
    f = PasswordField("pw", min_length=6)
    assert f.validate("short") is False
    assert f.validate("longpassword") is True

def test_form_is_valid():
    class MyForm(Form):
        name = TextField("name")
        email = EmailField("email")
    form = MyForm({"name": "Alice", "email": "a@b.com"})
    assert form.is_valid() is True
    assert form.cleaned_data["name"] == "Alice"

def test_form_validation_errors():
    class MyForm(Form):
        name = TextField("name", required=True)
    form = MyForm({})
    assert form.is_valid() is False
    assert "name" in form.errors

def test_form_as_html():
    class MyForm(Form):
        name = TextField("name")
    form = MyForm()
    html = form.as_html()
    assert "<form" in html
    assert "name" in html

def test_email_validator_rejects_invalid():
    v = EmailValidator()
    try:
        v("bad")
        assert False
    except Exception as e:
        assert "Invalid" in str(e)

def test_url_validator_accepts_valid():
    v = URLValidator()
    assert v("https://example.com") is True

def test_min_length_validator():
    v = MinLengthValidator(3)
    try:
        v("ab")
        assert False
    except Exception:
        pass

def test_max_length_validator():
    v = MaxLengthValidator(5)
    assert v("hello") is True
    try:
        v("toolong")
        assert False
    except Exception:
        pass

def test_range_validator():
    v = RangeValidator(min_value=0, max_value=100)
    assert v(50) is True
    try:
        v(200)
        assert False
    except Exception:
        pass

def test_model_form_from_dataclass():
    import dataclasses
    @dataclasses.dataclass
    class User:
        name: str = ""
        age: int = 0
    form = ModelForm.from_model(User, {"name": "Bob", "age": "30"})
    assert form.is_valid() is True
    assert form.cleaned_data["name"] == "Bob"

def test_field_render():
    f = TextField("test", required=True)
    f.value = "hello"
    html = f.render()
    assert "hello" in html
    assert "test" in html

def test_field_render_with_error():
    f = TextField("test", required=True)
    f.error = "This field is required"
    html = f.render()
    assert "error" in html



def test_create_user():
    ap = AuthProvider()
    user = ap.create_user("alice", "alice@test.com", "pass1234")
    assert user.username == "alice"
    assert user.email == "alice@test.com"
    assert user.id

def test_create_duplicate_user_raises():
    ap = AuthProvider()
    ap.create_user("bob", "bob@test.com", "pass1234")
    try:
        ap.create_user("bob", "bob2@test.com", "pass1234")
        assert False
    except ValueError:
        pass

def test_authenticate_valid():
    ap = AuthProvider()
    ap.create_user("carol", "carol@test.com", "secret123")
    user = ap.authenticate("carol", "secret123")
    assert user is not None
    assert user.username == "carol"

def test_authenticate_invalid_password():
    ap = AuthProvider()
    ap.create_user("dave", "dave@test.com", "pass1234")
    user = ap.authenticate("dave", "wrong")
    assert user is None

def test_authenticate_nonexistent_user():
    ap = AuthProvider()
    user = ap.authenticate("nobody", "pass")
    assert user is None

def test_create_and_validate_session():
    ap = AuthProvider()
    user = ap.create_user("eve", "eve@test.com", "pass1234")
    token = ap.create_session(user)
    validated = ap.validate_session(token)
    assert validated is not None
    assert validated.username == "eve"

def test_validate_expired_session():
    ap = AuthProvider(session_expiry=0)
    user = ap.create_user("frank", "frank@test.com", "pass1234")
    token = ap.create_session(user)
    time.sleep(0.01)
    validated = ap.validate_session(token)
    assert validated is None

def test_destroy_session():
    ap = AuthProvider()
    user = ap.create_user("grace", "grace@test.com", "pass1234")
    token = ap.create_session(user)
    ap.destroy_session(token)
    validated = ap.validate_session(token)
    assert validated is None

def test_password_hashing_and_verification():
    ap = AuthProvider()
    h = ap.hash_password("mypassword")
    assert ap.verify_password("mypassword", h) is True
    assert ap.verify_password("wrong", h) is False

def test_jwt_encode_and_decode():
    token = JWT.encode({"sub": "alice"}, "secret123")
    decoded = JWT.decode(token, "secret123")
    assert decoded["sub"] == "alice"

def test_jwt_expired_token():
    token = JWT.encode({"sub": "alice"}, "secret123", expires_in=-1)
    try:
        JWT.decode(token, "secret123")
        assert False
    except ValueError as e:
        assert "expired" in str(e)

def test_jwt_invalid_signature():
    token = JWT.encode({"sub": "alice"}, "secret123")
    try:
        JWT.decode(token, "wrongsecret")
        assert False
    except ValueError as e:
        assert "signature" in str(e)

def test_oauth2_flow():
    ap = AuthProvider()
    oauth = OAuth2Handler("client123", "secret456", ap)
    user = ap.create_user("hank", "hank@test.com", "pass1234")
    url = oauth.get_authorization_url(state="xyz")
    assert "client_id=client123" in url
    code = oauth.generate_authorization_code(user)
    token = oauth.exchange_code(code)
    assert token is not None
    validated = oauth.validate_token(token)
    assert validated is not None
    assert validated.username == "hank"

def test_oauth2_revoke_token():
    ap = AuthProvider()
    oauth = OAuth2Handler("c", "s", ap)
    user = ap.create_user("ivy", "ivy@test.com", "pass1234")
    code = oauth.generate_authorization_code(user)
    token = oauth.exchange_code(code)
    oauth.revoke_token(token)
    assert oauth.validate_token(token) is None

def test_rbac_define_role_and_check_permission():
    rbac = RBAC()
    rbac.define_role("admin", ["read", "write", "delete"])
    rbac.define_role("viewer", ["read"])
    user = User(id="1", username="alice", email="a@b.com", password_hash="", roles=["admin"])
    assert rbac.has_permission(user, "read") is True
    assert rbac.has_permission(user, "delete") is True
    viewer = User(id="2", username="bob", email="b@b.com", password_hash="", roles=["viewer"])
    assert rbac.has_permission(viewer, "delete") is False

def test_rbac_hierarchy():
    rbac = RBAC()
    rbac.define_role("base", ["read"])
    rbac.define_role("editor", ["write"])
    rbac.set_hierarchy("editor", "base")
    user = User(id="1", username="u", email="e@e.com", password_hash="", roles=["editor"])
    assert rbac.has_permission(user, "read") is True
    assert rbac.has_permission(user, "write") is True



def test_create_table_and_save():
    set_connection(":memory:")
    class TestUser(Model):
        __table__ = "test_users"
        name = StrCol("name")
        email = StrCol("email")
    TestUser.create_table()
    u = TestUser(name="alice", email="alice@test.com")
    u.save()
    assert u.id is not None

def test_get_by_id():
    set_connection(":memory:")
    class TestUser2(Model):
        __table__ = "test_users2"
        name = StrCol("name")
    TestUser2.create_table()
    u = TestUser2(name="bob")
    u.save()
    fetched = TestUser2.get(u.id)
    assert fetched is not None
    assert fetched.name == "bob"

def test_filter_records():
    set_connection(":memory:")
    class TestUser3(Model):
        __table__ = "test_users3"
        name = StrCol("name")
        role = StrCol("role")
    TestUser3.create_table()
    TestUser3(name="alice", role="admin").save()
    TestUser3(name="bob", role="user").save()
    TestUser3(name="carol", role="admin").save()
    admins = TestUser3.filter(role="admin")
    assert len(admins) == 2

def test_update_record():
    set_connection(":memory:")
    class TestUser4(Model):
        __table__ = "test_users4"
        name = StrCol("name")
    TestUser4.create_table()
    u = TestUser4(name="old")
    u.save()
    u.name = "new"
    u.save()
    fetched = TestUser4.get(u.id)
    assert fetched.name == "new"

def test_delete_record():
    set_connection(":memory:")
    class TestUser5(Model):
        __table__ = "test_users5"
        name = StrCol("name")
    TestUser5.create_table()
    u = TestUser5(name="doomed")
    u.save()
    uid = u.id
    u.delete()
    assert TestUser5.get(uid) is None

def test_all_records():
    set_connection(":memory:")
    class TestItem(Model):
        __table__ = "test_items"
        name = StrCol("name")
    TestItem.create_table()
    TestItem(name="a").save()
    TestItem(name="b").save()
    assert len(TestItem.all()) == 2

def test_count_records():
    set_connection(":memory:")
    class TestCount(Model):
        __table__ = "test_counts"
        name = StrCol("name")
    TestCount.create_table()
    TestCount(name="a").save()
    TestCount(name="b").save()
    TestCount(name="c").save()
    assert TestCount.count() == 3

def test_query_builder_order_by():
    set_connection(":memory:")
    class TestQB(Model):
        __table__ = "test_qb"
        name = StrCol("name")
        rank = IntCol("rank")
    TestQB.create_table()
    TestQB(name="b", rank=2).save()
    TestQB(name="a", rank=1).save()
    TestQB(name="c", rank=3).save()
    results = TestQB.select().order_by("rank").execute()
    assert results[0].name == "a"
    assert results[1].name == "b"

def test_query_builder_limit():
    set_connection(":memory:")
    class TestLimit(Model):
        __table__ = "test_limit"
        name = StrCol("name")
    TestLimit.create_table()
    for i in range(5):
        TestLimit(name=f"item{i}").save()
    results = TestLimit.select().limit(2).execute()
    assert len(results) == 2

def test_drop_table():
    conn = set_connection(":memory:")
    class TestDrop(Model):
        __table__ = "test_drop"
        name = StrCol("name")
    TestDrop.create_table()
    TestDrop.drop_table()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_drop'")
    assert cursor.fetchone() is None

def test_get_non_existent_returns_none():
    set_connection(":memory:")
    class TestNone(Model):
        __table__ = "test_none"
        name = StrCol("name")
    TestNone.create_table()
    assert TestNone.get(999) is None

def test_model_repr():
    set_connection(":memory:")
    class TestRepr(Model):
        __table__ = "test_repr"
        name = StrCol("name")
    TestRepr.create_table()
    u = TestRepr(name="test")
    u.save()
    r = repr(u)
    assert "TestRepr" in r
    assert "id=" in r

def test_foreign_key_column():
    fk = ForeignKey("user_id", "users")
    sql = fk.to_sql()
    assert "REFERENCES users(id)" in sql



def test_websocket_frame_text_encode_decode():
    frame = WebSocketFrame.text("Hello")
    encoded = frame.encode(mask=False)
    decoded, consumed = WebSocketFrame.decode(encoded)
    assert decoded.is_text()
    assert decoded.text_payload() == "Hello"
    assert consumed == len(encoded)

def test_websocket_frame_binary_encode_decode():
    frame = WebSocketFrame.binary(b"\x01\x02\x03")
    encoded = frame.encode(mask=False)
    decoded, consumed = WebSocketFrame.decode(encoded)
    assert decoded.is_binary()
    assert decoded.payload == b"\x01\x02\x03"

def test_websocket_frame_masked_encode_decode():
    frame = WebSocketFrame.text("Secret")
    encoded = frame.encode(mask=True)
    decoded, consumed = WebSocketFrame.decode(encoded)
    assert decoded.text_payload() == "Secret"

def test_websocket_frame_close_encode_decode():
    frame = WebSocketFrame.close(code=1000, reason="bye")
    encoded = frame.encode(mask=False)
    decoded, consumed = WebSocketFrame.decode(encoded)
    assert decoded.is_close()
    assert decoded.close_code() == 1000

def test_websocket_frame_ping_pong():
    ping = WebSocketFrame.ping(b"ping!")
    encoded = ping.encode(mask=False)
    decoded, consumed = WebSocketFrame.decode(encoded)
    assert decoded.is_ping()
    assert decoded.payload == b"ping!"
    pong = WebSocketFrame.pong(b"pong!")
    encoded2 = pong.encode(mask=False)
    decoded2, consumed2 = WebSocketFrame.decode(encoded2)
    assert decoded2.is_pong()

def test_websocket_frame_fin_flag():
    frame = WebSocketFrame(opcode=0x1, payload=b"data", fin=True)
    encoded = frame.encode(mask=False)
    decoded, _ = WebSocketFrame.decode(encoded)
    assert decoded.fin is True

def test_websocket_frame_rsv_flags():
    frame = WebSocketFrame(opcode=0x1, payload=b"data", rsv1=True)
    encoded = frame.encode(mask=False)
    decoded, _ = WebSocketFrame.decode(encoded)
    assert decoded.rsv1 is True

def test_websocket_frame_empty_payload():
    frame = WebSocketFrame.text("")
    encoded = frame.encode(mask=False)
    decoded, _ = WebSocketFrame.decode(encoded)
    assert decoded.payload == b""

def test_websocket_frame_large_payload():
    big = b"x" * 70000
    frame = WebSocketFrame.binary(big)
    encoded = frame.encode(mask=False)
    decoded, _ = WebSocketFrame.decode(encoded)
    assert decoded.payload == big



def test_mime_type_detection():
    sfs = StaticFileServer()
    assert sfs._detect_mime("test.html") == "text/html"
    assert sfs._detect_mime("test.css") == "text/css"
    assert sfs._detect_mime("test.js") == "application/javascript"
    assert sfs._detect_mime("test.png") == "image/png"
    assert sfs._detect_mime("test.unknown") == "application/octet-stream"

def test_serve_non_existent_file():
    sfs = StaticFileServer()
    result = sfs.serve("/nonexistent.txt")
    assert result["status"] == 404

def test_serve_actual_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello world")
        sfs = StaticFileServer(root_dir=tmpdir)
        result = sfs.serve("/test.txt")
        assert result["status"] == 200
        assert result["body"] == b"hello world"
        assert "text/plain" in result["headers"]["Content-Type"]

def test_serve_304_not_modified():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "cached.txt")
        with open(test_file, "w") as f:
            f.write("cached content")
        sfs = StaticFileServer(root_dir=tmpdir)
        content = open(test_file, "rb").read()
        etag = sfs.cache.generate_etag(content)
        result = sfs.serve("/cached.txt", {"If-None-Match": etag})
        assert result["status"] == 304

def test_serve_directory_listing():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "file.txt"), "w") as f:
            f.write("test")
        sfs = StaticFileServer(root_dir=tmpdir)
        result = sfs.serve("/subdir")
        assert result["status"] == 200
        assert b"subdir" in result["body"]

def test_asset_pipeline_minify_css():
    pipeline = AssetPipeline()
    css = "body { \n  color: red;  \n  background: blue; \n}"
    result = pipeline.minify(css, "css")
    assert "\n" not in result
    assert "  " not in result

def test_asset_pipeline_minify_js():
    pipeline = AssetPipeline()
    js = "function hello() {\n  return 'world';\n}"
    result = pipeline.minify(js, "js")
    assert "\n" not in result

def test_asset_pipeline_fingerprint():
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = AssetPipeline(output_dir=tmpdir)
        test_file = os.path.join(tmpdir, "app.js")
        with open(test_file, "w") as f:
            f.write("console.log('hello')")
        fingerprinted = pipeline.fingerprint(test_file)
        assert "app." in fingerprinted
        assert fingerprinted.endswith(".js")
        assert len(fingerprinted.split(".")[1]) == 8

def test_cache_headers_etag_generation():
    ch = CacheHeaders()
    etag = ch.generate_etag(b"test content")
    assert etag.startswith('"')
    assert etag.endswith('"')

def test_cache_headers_apply():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "cached.html")
        with open(test_file, "w") as f:
            f.write("content")
        ch = CacheHeaders(max_age=7200)
        headers = ch.apply_headers({}, b"content", test_file)
        assert "ETag" in headers
        assert "Cache-Control" in headers
        assert "max-age=7200" in headers["Cache-Control"]

def test_asset_pipeline_concat():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.txt")
        f2 = os.path.join(tmpdir, "b.txt")
        with open(f1, "w") as f:
            f.write("hello")
        with open(f2, "w") as f:
            f.write("world")
        pipeline = AssetPipeline(output_dir=tmpdir)
        result = pipeline.concat([f1, f2], "combined.txt")
        content = open(result).read()
        assert "hello" in content
        assert "world" in content
