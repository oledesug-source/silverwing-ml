"""Tests for enhanced intelligence.security — RBAC, JWT, audit, encryption, auth, crypto, validation."""

import os
import time

from intelligence.security.audit import AuditAction, AuditLog
from intelligence.security.auth import Authenticator
from intelligence.security.crypto import Checksum, Hasher
from intelligence.security.encryption import AesCipher, EncryptedPayload
from intelligence.security.jwt import TokenClaims, TokenService
from intelligence.security.rbac import RBAC, Permission, Role, Subject
from intelligence.security.validation import InputValidator, Sanitizer


class TestAuthenticator:
    def test_create_and_validate_key(self) -> None:
        auth = Authenticator()
        api_key, raw_key = auth.create_key("test-app", scopes=["read"])
        assert api_key.valid
        assert auth.validate_key(api_key.key_id, raw_key)

    def test_revoke_key(self) -> None:
        auth = Authenticator()
        api_key, raw_key = auth.create_key("app")
        auth.revoke_key(api_key.key_id)
        assert not auth.validate_key(api_key.key_id, raw_key)

    def test_hmac_signature(self) -> None:
        auth = Authenticator()
        sig = auth.sign_request("POST", "/api", b"data")
        assert auth.verify_signature("POST", "/api", b"data", sig)
        assert not auth.verify_signature("POST", "/api", b"other", sig)

    def test_token_lifecycle(self) -> None:
        auth = Authenticator()
        token = auth.create_token("user-1", ttl_seconds=60)
        assert token.valid
        validated = auth.validate_token(token.token_id)
        assert validated is not None


class TestRBAC:
    def test_basic_permissions(self) -> None:
        rbac = RBAC()
        rbac.add_role(Role(
            name="researcher",
            permissions={Permission.READ, Permission.TRAIN},
        ))
        subject = Subject(subject_id="u1", roles=["researcher"])
        assert rbac.has_permission(subject, Permission.READ)
        assert not rbac.has_permission(subject, Permission.DELETE)

    def test_inheritance(self) -> None:
        rbac = RBAC()
        rbac.add_role(Role(name="base", permissions={Permission.READ}))
        rbac.add_role(Role(name="admin", permissions={Permission.ADMIN}, inherits_from=["base"]))
        subject = Subject(subject_id="u1", roles=["admin"])
        assert rbac.has_permission(subject, Permission.READ)
        assert rbac.has_permission(subject, Permission.ADMIN)

    def test_require_permission(self) -> None:
        rbac = RBAC()
        rbac.add_role(Role(name="viewer", permissions={Permission.READ}))
        subject = Subject(subject_id="u1", roles=["viewer"])
        rbac.require_permission(subject, Permission.READ)
        try:
            rbac.require_permission(subject, Permission.DELETE)
            assert False
        except PermissionError:
            pass

    def test_has_any_permission(self) -> None:
        rbac = RBAC()
        rbac.add_role(Role(name="writer", permissions={Permission.WRITE}))
        subject = Subject(subject_id="u1", roles=["writer"])
        assert rbac.has_any_permission(subject, {Permission.READ, Permission.WRITE})
        assert not rbac.has_any_permission(subject, {Permission.READ, Permission.ADMIN})

    def test_list_remove_roles(self) -> None:
        rbac = RBAC()
        rbac.add_role(Role(name="r1", permissions=set()))
        rbac.add_role(Role(name="r2", permissions=set()))
        assert len(rbac.list_roles()) == 2
        rbac.remove_role("r1")
        assert len(rbac.list_roles()) == 1


class TestJWT:
    def test_create_and_validate(self) -> None:
        svc = TokenService(secret="test-secret")
        token = svc.create_token(subject="user-1", scopes=["read"], ttl_seconds=60)
        claims = svc.validate_token(token)
        assert claims is not None
        assert claims.subject == "user-1"
        assert "read" in claims.scopes

    def test_expired_token(self) -> None:
        svc = TokenService(secret="test")
        token = svc.create_token(subject="u1", ttl_seconds=-1)
        assert svc.validate_token(token) is None

    def test_invalid_signature(self) -> None:
        svc1 = TokenService(secret="key1")
        svc2 = TokenService(secret="key2")
        token = svc1.create_token(subject="u1")
        assert svc2.validate_token(token) is None

    def test_refresh_token(self) -> None:
        svc = TokenService(secret="test")
        token = svc.create_token(subject="u1", ttl_seconds=60)
        refreshed = svc.refresh_token(token, ttl_seconds=120)
        assert refreshed is not None
        claims = svc.validate_token(refreshed)
        assert claims is not None

    def test_scope_check(self) -> None:
        svc = TokenService(secret="test")
        token = svc.create_token(subject="u1", scopes=["read", "write"])
        assert svc.has_scope(token, "read")
        assert not svc.has_scope(token, "admin")

    def test_wildcard_scope(self) -> None:
        svc = TokenService(secret="test")
        token = svc.create_token(subject="u1", scopes=["*"])
        assert svc.has_scope(token, "anything")


class TestAuthenticatorExtended:
    def test_token_expiry(self) -> None:
        auth = Authenticator()
        token = auth.create_token("user-1", ttl_seconds=0.01)
        assert token.valid
        time.sleep(0.02)
        assert not auth.validate_token(token.token_id)

    def test_list_keys(self) -> None:
        auth = Authenticator()
        auth.create_key("app1")
        auth.create_key("app2")
        keys = auth.list_keys()
        assert len(keys) == 2

    def test_invalid_raw_key(self) -> None:
        auth = Authenticator()
        api_key, raw_key = auth.create_key("app")
        assert not auth.validate_key(api_key.key_id, "wrong-key")

    def test_revoke_nonexistent(self) -> None:
        auth = Authenticator()
        assert not auth.revoke_key("nonexistent")

    def test_key_with_ttl(self) -> None:
        auth = Authenticator()
        api_key, raw_key = auth.create_key("app", ttl_seconds=0.01)
        assert api_key.expires_at is not None
        time.sleep(0.02)
        assert not api_key.valid


class TestRBACExtended:
    def test_get_role(self) -> None:
        rbac = RBAC()
        role = Role(name="analyst", permissions={Permission.READ})
        rbac.add_role(role)
        assert rbac.get_role("analyst") is role
        assert rbac.get_role("nonexistent") is None

    def test_remove_nonexistent(self) -> None:
        rbac = RBAC()
        assert not rbac.remove_role("ghost")

    def test_circular_inheritance(self) -> None:
        rbac = RBAC()
        rbac.add_role(Role(name="a", permissions={Permission.READ}, inherits_from=["b"]))
        rbac.add_role(Role(name="b", permissions={Permission.WRITE}, inherits_from=["a"]))
        subject = Subject(subject_id="u1", roles=["a"])
        perms = rbac.get_permissions(subject)
        assert Permission.READ in perms
        assert Permission.WRITE in perms

    def test_multiple_roles(self) -> None:
        rbac = RBAC()
        rbac.add_role(Role(name="reader", permissions={Permission.READ}))
        rbac.add_role(Role(name="writer", permissions={Permission.WRITE}))
        subject = Subject(subject_id="u1", roles=["reader", "writer"])
        assert rbac.has_permission(subject, Permission.READ)
        assert rbac.has_permission(subject, Permission.WRITE)

    def test_unknown_role(self) -> None:
        rbac = RBAC()
        subject = Subject(subject_id="u1", roles=["ghost"])
        assert not rbac.has_permission(subject, Permission.READ)

    def test_empty_permissions(self) -> None:
        rbac = RBAC()
        subject = Subject(subject_id="u1", roles=[])
        assert rbac.get_permissions(subject) == set()


class TestJWTExtended:
    def test_claims_to_dict(self) -> None:
        claims = TokenClaims(subject="u1", scopes=["read"], expires_at=123.0)
        d = claims.to_dict()
        assert d["sub"] == "u1"
        assert d["scopes"] == ["read"]
        assert d["exp"] == 123.0

    def test_claims_from_dict(self) -> None:
        d = {"sub": "u1", "iat": 1.0, "exp": 2.0, "iss": "test", "aud": "api", "scopes": ["write"], "jti": "abc", "meta": {}}
        claims = TokenClaims.from_dict(d)
        assert claims.subject == "u1"
        assert claims.scopes == ["write"]

    def test_claims_expired(self) -> None:
        claims = TokenClaims(subject="u1", expires_at=0.0)
        assert not claims.expired

    def test_claims_expired_past(self) -> None:
        claims = TokenClaims(subject="u1", expires_at=1.0)
        # 1.0 is in the past (1970), so expired
        assert claims.expired

    def test_claims_remaining_seconds_inf(self) -> None:
        claims = TokenClaims(subject="u1", expires_at=0.0)
        assert claims.remaining_seconds == float("inf")

    def test_token_service_no_secret(self) -> None:
        svc = TokenService()
        token = svc.create_token(subject="u1")
        claims = svc.validate_token(token)
        assert claims is not None
        assert claims.subject == "u1"


class TestValidationExtended:
    def test_numeric_validation(self) -> None:
        v = InputValidator()
        v.add_rule("temp", required=True, min_val=0.0, max_val=2.0)
        assert v.validate({"temp": 1.5}).valid
        assert not v.validate({"temp": 3.0}).valid
        assert not v.validate({"temp": -1.0}).valid

    def test_type_checking(self) -> None:
        v = InputValidator()
        v.add_rule("name", required=True, allowed_types=(str,))
        assert v.validate({"name": "hello"}).valid
        assert not v.validate({"name": 123}).valid

    def test_min_length(self) -> None:
        v = InputValidator()
        v.add_rule("code", required=True, min_length=5)
        assert not v.validate({"code": "hi"}).valid
        assert v.validate({"code": "hello"}).valid

    def test_pattern(self) -> None:
        v = InputValidator()
        v.add_rule("email", required=True, pattern=r"^[a-z]+@[a-z]+\.[a-z]+$")
        assert v.validate({"email": "user@test.com"}).valid
        assert not v.validate({"email": "bad"}).valid

    def test_optional_field(self) -> None:
        v = InputValidator()
        v.add_rule("opt", required=False)
        assert v.validate({}).valid

    def test_sanitizer_number_valid(self) -> None:
        s = Sanitizer()
        assert s.number("42.5", 0, 100) == 42.5

    def test_sanitizer_number_out_of_range(self) -> None:
        s = Sanitizer()
        assert s.number("200", 0, 100) is None

    def test_sanitizer_number_invalid(self) -> None:
        s = Sanitizer()
        assert s.number("abc") is None

    def test_strip_html(self) -> None:
        s = Sanitizer()
        assert s.strip_html("<b>hello</b>") == "hello"


class TestCryptoExtended:
    def test_multi_algorithm_hash(self) -> None:
        h = Hasher(algorithms=("sha256", "sha512"))
        h.update(b"test")
        result = h.digest()
        assert "sha256" in result
        assert "sha512" in result
        assert result["sha256"] != result["sha512"]

    def test_hash_string(self) -> None:
        result = Hasher.hash_string("hello world")
        assert len(result["sha256"]) == 64

    def test_generate_salt(self) -> None:
        salt = Hasher.generate_salt(16)
        assert len(salt) == 32

    def test_reset(self) -> None:
        h = Hasher()
        h.update(b"test")
        result1 = h.digest()
        h.update(b"more")
        result2 = h.digest()
        assert result1["sha256"] != result2["sha256"]

    def test_invalid_algorithm(self) -> None:
        try:
            Hasher(algorithms=("bad_algo",))
            assert False
        except ValueError:
            pass


class TestEncryptionExtended:
    def test_generate_key(self) -> None:
        key = AesCipher.generate_key()
        assert len(key) == 32

    def test_from_password_with_salt(self) -> None:
        c1 = AesCipher.from_password("pass", salt="fixed-salt")
        c2 = AesCipher.from_password("pass", salt="fixed-salt")
        enc = c1.encrypt(b"test")
        dec = c2.decrypt(enc)
        assert dec == b"test"

    def test_empty_plaintext(self) -> None:
        cipher = AesCipher.from_password("test")
        enc = cipher.encrypt(b"")
        dec = cipher.decrypt(enc)
        assert dec == b""

    def test_large_data(self) -> None:
        cipher = AesCipher.from_password("test")
        data = b"x" * 200
        enc = cipher.encrypt(data)
        dec = cipher.decrypt(enc)
        assert dec == data

    def test_associated_data(self) -> None:
        cipher = AesCipher.from_password("test")
        enc = cipher.encrypt(b"secret", associated_data=b"meta")
        dec = cipher.decrypt(enc, associated_data=b"meta")
        assert dec == b"secret"
        try:
            cipher.decrypt(enc, associated_data=b"wrong")
            assert False
        except ValueError:
            pass

    def test_payload_serialization(self) -> None:
        cipher = AesCipher.from_password("test")
        enc = cipher.encrypt(b"data")
        b64 = enc.to_base64()
        restored = EncryptedPayload.from_base64(b64)
        dec = cipher.decrypt(restored)
        assert dec == b"data"

    def test_key_from_bytes(self) -> None:
        key = os.urandom(32)
        cipher = AesCipher(key)
        enc = cipher.encrypt(b"test")
        dec = cipher.decrypt(enc)
        assert dec == b"test"

    def test_password_hash_deterministic(self) -> None:
        h1, s1 = AesCipher.hash_password("pw", salt="fixed")
        h2, s2 = AesCipher.hash_password("pw", salt="fixed")
        assert h1 == h2
        assert s1 == s2


class TestHasher:
    def test_hash_file(self, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_bytes(b"content")
        result = Hasher.hash_file(str(f))
        assert len(result["sha256"]) == 64

    def test_checksum_verify(self) -> None:
        data = b"test"
        h = Hasher(algorithms=("sha256",))
        h.update(data)
        checksum = Checksum(algorithm="sha256", digest=h.digest()["sha256"])
        assert checksum.verify(data)
        assert not checksum.verify(b"wrong")


class TestEncryption:
    def test_encrypt_decrypt(self) -> None:
        cipher = AesCipher.from_password("secret")
        plaintext = b"hello world encryption test"
        encrypted = cipher.encrypt(plaintext)
        decrypted = cipher.decrypt(encrypted)
        assert decrypted == plaintext

    def test_wrong_password_fails(self) -> None:
        c1 = AesCipher.from_password("pass1")
        c2 = AesCipher.from_password("pass2")
        encrypted = c1.encrypt(b"secret")
        try:
            c2.decrypt(encrypted)
            assert False
        except ValueError:
            pass

    def test_tamper_detection(self) -> None:
        cipher = AesCipher.from_password("test")
        encrypted = cipher.encrypt(b"data")
        tampered = EncryptedPayload(
            ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce,
            tag=b"x" * 16,
        )
        try:
            cipher.decrypt(tampered)
            assert False
        except ValueError:
            pass

    def test_base64_roundtrip(self) -> None:
        cipher = AesCipher.generate_key()
        enc = AesCipher(cipher)
        encrypted = enc.encrypt(b"roundtrip test")
        b64 = encrypted.to_base64()
        decrypted = enc.decrypt(EncryptedPayload.from_base64(b64))
        assert decrypted == b"roundtrip test"

    def test_password_hash(self) -> None:
        h, salt = AesCipher.hash_password("mypassword")
        assert AesCipher.verify_password("mypassword", h, salt)
        assert not AesCipher.verify_password("wrong", h, salt)


class TestValidation:
    def test_valid_input(self) -> None:
        v = InputValidator()
        v.add_rule("prompt", required=True, max_length=100)
        result = v.validate({"prompt": "Hello"})
        assert result.valid

    def test_missing_required(self) -> None:
        v = InputValidator()
        v.add_rule("prompt", required=True)
        result = v.validate({})
        assert not result.valid

    def test_sanitizer_text(self) -> None:
        s = Sanitizer()
        clean = s.text("<script>alert(1)</script>test")
        assert "<script>" not in clean

    def test_sanitizer_filename(self) -> None:
        s = Sanitizer()
        clean = s.filename("../../etc/passwd")
        assert ".." not in clean
        assert "/" not in clean

    def test_sql_injection(self) -> None:
        s = Sanitizer()
        result = s.sql_input("'; DROP TABLE x; --")
        assert not result.valid


class TestAuditLog:
    def test_record_and_query(self) -> None:
        log = AuditLog(max_entries=100)
        log.record(AuditAction.AUTH_SUCCESS, subject="user-1")
        log.record(AuditAction.DATA_READ, subject="user-1", resource="corpus")
        log.record(AuditAction.AUTH_FAILURE, subject="user-2")

        results = log.query(subject="user-1")
        assert len(results) == 2

    def test_count(self) -> None:
        log = AuditLog()
        log.record(AuditAction.API_CALL)
        log.record(AuditAction.API_CALL)
        assert log.count == 2

    def test_get_recent(self) -> None:
        log = AuditLog()
        for i in range(20):
            log.record(AuditAction.API_CALL, resource=f"r{i}")
        recent = log.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].resource == "r19"

    def test_clear(self) -> None:
        log = AuditLog()
        log.record(AuditAction.API_CALL)
        log.clear()
        assert log.count == 0

    def test_hash_chain(self) -> None:
        log = AuditLog()
        log.record(AuditAction.AUTH_SUCCESS, subject="u1")
        log.record(AuditAction.DATA_READ, subject="u1")
        assert log.verify_chain()

    def test_action_filter(self) -> None:
        log = AuditLog()
        log.record(AuditAction.AUTH_SUCCESS, subject="u1")
        log.record(AuditAction.AUTH_FAILURE, subject="u2")
        log.record(AuditAction.DATA_READ, subject="u1")

        results = log.query(action=AuditAction.AUTH_SUCCESS)
        assert len(results) == 1
