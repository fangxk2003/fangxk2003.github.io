#!/usr/bin/env python3
"""Sync only daily Garmin calorie totals through the scoped Calories endpoint.

The scheduled worker never receives a Garmin password. Initial connection runs
interactively, and access/refresh tokens exist only in memory or encrypted form.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from getpass import getpass
import json
import logging
import math
import os
import re
import stat
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


OWNER_ID = "8c0b4ad9-f74f-4951-bb9d-a4a3d4d1f686"
PURPOSE = b"calories-garmin-session-v1"
SESSION_FIELDS = ("di_token", "di_refresh_token", "di_client_id")
MAX_RESPONSE_BYTES = 131_072
MAX_SESSION_BYTES = 65_536
MAX_DAILY_CALORIES = 100_000
PUBLIC_MESSAGES = {
    "success": "Synced {count} daily calorie records.",
    "busy": "Another sync is running. Try again shortly.",
    "not_connected": "Garmin is not connected yet. Run the connect command first.",
    "needs_login": "Garmin needs a fresh sign-in. Run the connect command again.",
    "upstream_unavailable": "Sync is temporarily unavailable. Saved records remain available.",
    "missing_data": "Garmin returned no complete daily calorie records. Saved values are unchanged.",
}


class WorkerFailure(Exception):
    """An intentionally generic error safe to display in private workflow logs."""

    def __init__(self, code: str = "upstream_unavailable") -> None:
        self.code = code
        super().__init__(PUBLIC_MESSAGES[code])


class ConfigurationError(Exception):
    """A safe configuration error that never includes the supplied values."""


@dataclass(frozen=True)
class Config:
    endpoint: str
    credential: str
    timezone: str = "Europe/Stockholm"

    def validate(self) -> Config:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".supabase.co")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.port not in (None, 443)
        ):
            raise ConfigurationError("Set a valid HTTPS Supabase sync endpoint.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.credential):
            raise ConfigurationError("The sync credential must contain 64 lowercase hexadecimal characters.")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ConfigurationError("Set a valid IANA timezone.") from None
        return self

    @property
    def associated_data(self) -> bytes:
        # The endpoint identifies the project; this integration has one owner.
        return PURPOSE + b"\0" + self.endpoint.rstrip("/").encode() + b"\0" + OWNER_ID.encode()


@dataclass(frozen=True)
class RunResult:
    status: str
    count: int = 0
    cleanup_pending: bool = False


def load_config(path: str | None, environ: dict[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    if path:
        # No token/password files are written by this worker. The optional config
        # contains the scoped worker credential, so require owner-only access.
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as config_file:
                info = os.fstat(config_file.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                    raise ConfigurationError("The config must be a regular owner-only file (chmod 600).")
                if hasattr(os, "getuid") and info.st_uid != os.getuid():
                    raise ConfigurationError("The config must be owned by your user account.")
                raw = config_file.read(16_385)
                if len(raw) > 16_384:
                    raise ConfigurationError("The config file is too large.")
                values = json.loads(raw)
                if not isinstance(values, dict):
                    raise ConfigurationError("The config must be a JSON object.")
        except ConfigurationError:
            raise
        except (OSError, ValueError, UnicodeError):
            raise ConfigurationError("Could not read the private JSON config file.") from None
    endpoint = env.get("GARMIN_SYNC_ENDPOINT", values.get("endpoint", ""))
    credential = env.get("GARMIN_SYNC_CREDENTIAL", values.get("credential", ""))
    timezone = env.get("GARMIN_TIMEZONE", values.get("timezone", "Europe/Stockholm"))
    if not all(isinstance(value, str) for value in (endpoint, credential, timezone)):
        raise ConfigurationError("Configuration fields must be strings.")
    try:
        return Config(endpoint.rstrip("/"), credential, timezone).validate()
    except ValueError:
        raise ConfigurationError("Set a valid HTTPS Supabase sync endpoint.") from None


def _key(config: Config) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=PURPOSE + b"/encryption"
    ).derive(bytes.fromhex(config.credential))


def _validate_session(raw: str) -> str:
    try:
        if not isinstance(raw, str) or len(raw.encode()) > MAX_SESSION_BYTES:
            raise ValueError
        data = json.loads(raw)
        if not isinstance(data, dict) or any(
            not isinstance(data.get(field), str) or not data[field]
            for field in SESSION_FIELDS
        ):
            raise ValueError
        # Drop anything beyond the three resumable native DI token fields.
        return json.dumps({field: data[field] for field in SESSION_FIELDS}, separators=(",", ":"))
    except (TypeError, ValueError, UnicodeError):
        raise WorkerFailure("needs_login") from None


def encrypt_session(raw: str, config: Config) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    plaintext = _validate_session(raw).encode()
    nonce = os.urandom(12)
    ciphertext = AESGCM(_key(config)).encrypt(nonce, plaintext, config.associated_data)
    # Version is outside the ciphertext to allow a future format migration.
    encoded = base64.urlsafe_b64encode(nonce + ciphertext).decode().rstrip("=")
    return "v1." + encoded


def decrypt_session(envelope: str, config: Config) -> str:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        if not isinstance(envelope, str) or not re.fullmatch(r"v1\.[A-Za-z0-9_-]+", envelope):
            raise ValueError
        if len(envelope) > MAX_RESPONSE_BYTES:
            raise ValueError
        encoded = envelope[3:]
        data = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        if len(data) < 29:
            raise ValueError
        raw = AESGCM(_key(config)).decrypt(data[:12], data[12:], config.associated_data)
        return _validate_session(raw.decode())
    except (InvalidTag, ValueError, TypeError, UnicodeError, binascii.Error):
        raise WorkerFailure("needs_login") from None


class EndpointClient:
    def __init__(self, config: Config) -> None:
        self.config = config

    def call(self, operation: str, **fields: Any) -> dict[str, Any]:
        payload = json.dumps(
            {"operation": operation, "credential": self.config.credential, **fields},
            separators=(",", ":"), allow_nan=False,
        ).encode()
        request = Request(
            self.config.endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        # Repeat only operations with an existing lease. Never acquire twice if
        # the response is lost, which could strand a second lease.
        attempts = 1 if operation == "acquire" else 3
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=12) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError
                result = json.loads(raw)
                if not isinstance(result, dict) or result.get("error"):
                    raise ValueError
                return result
            except HTTPError as error:
                # Do not inspect/log server bodies; they may contain request data.
                if operation == "release" and error.code == 409:
                    # A prior release can succeed while its response is lost.
                    # A stale lease is already unable to write or lock the row.
                    return {"ok": True}
                if error.code < 500 or attempt + 1 == attempts:
                    raise WorkerFailure() from None
            except (URLError, OSError):
                if attempt + 1 == attempts:
                    raise WorkerFailure() from None
            except (ValueError, UnicodeError):
                raise WorkerFailure() from None
            time.sleep(attempt + 1)
        raise WorkerFailure()


def normalize_rows(raw: Any, start: date, end: date) -> list[dict[str, Any]]:
    """Keep complete daily totals, never replace missing Garmin values with zero."""
    if not isinstance(raw, list):
        raise WorkerFailure("missing_data")
    rows: dict[str, dict[str, Any]] = {}
    for record in raw:
        if not isinstance(record, dict):
            continue
        calendar_date = record.get("calendarDate")
        try:
            if not isinstance(calendar_date, str):
                continue
            day = date.fromisoformat(calendar_date)
            if day.isoformat() != calendar_date or not start <= day <= end:
                continue
        except ValueError:
            continue
        values = (record.get("active"), record.get("resting"))
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or value > MAX_DAILY_CALORIES
            for value in values
        ):
            continue
        active, resting = map(float, values)
        total = math.fsum((active, resting))
        if total > MAX_DAILY_CALORIES:
            continue
        normalized = {"date": calendar_date, "active": active, "resting": resting, "total": total}
        if calendar_date in rows and rows[calendar_date] != normalized:
            # Conflicting duplicate dates cannot safely be resolved by list order.
            raise WorkerFailure("missing_data")
        rows[calendar_date] = normalized
    if not rows:
        raise WorkerFailure("missing_data")
    return [rows[key] for key in sorted(rows)]


def _new_garmin(**kwargs: Any) -> Any:
    from garminconnect import Garmin

    return Garmin(**kwargs)


class RefreshGuard:
    """Observe refresh outcomes hidden by the pinned client's suppression.

    garminconnect 0.3.17 can swallow a refresh timeout/500 and subsequently
    report an expired access token as an authentication failure. Only an actual
    refresh rejection proves the saved refresh credential should be replaced.
    """

    def __init__(self, client: Any) -> None:
        self.rejected = False
        self.native = client.client
        self.original_post = getattr(self.native, "_http_post", None)
        refresh_url = getattr(self.native, "_di_token_url", None)
        if not callable(self.original_post) or not isinstance(refresh_url, str):
            return

        def observed_post(url: str, **kwargs: Any) -> Any:
            is_refresh = (
                url == refresh_url and isinstance(kwargs.get("data"), dict)
                and kwargs["data"].get("grant_type") == "refresh_token"
            )
            try:
                response = self.original_post(url, **kwargs)
            except Exception:
                if is_refresh:
                    self.rejected = False
                raise
            if is_refresh:
                status = getattr(response, "status_code", None)
                self.rejected = status in (401, 403)
                if status == 400:
                    # OAuth invalid_grant/invalid_token is definitive; other
                    # 400s could reflect an upstream client/protocol problem.
                    try:
                        data = response.json()
                        self.rejected = isinstance(data, dict) and data.get("error") in (
                            "invalid_grant", "invalid_token",
                        )
                    except Exception:
                        self.rejected = False
            return response

        self.native._http_post = observed_post

    def restore(self) -> None:
        if callable(self.original_post):
            self.native._http_post = self.original_post


def _error_code(error: Exception, guard: RefreshGuard | None = None, *, interactive: bool = False) -> str:
    if isinstance(error, WorkerFailure):
        return error.code
    if guard is not None and guard.rejected:
        return "needs_login"
    from garminconnect import GarminConnectAuthenticationError

    if interactive and isinstance(error, GarminConnectAuthenticationError):
        return "needs_login"
    # A class-only 401 may be an expired access token after a swallowed refresh
    # outage. Preserve the valid refresh credential unless rejection is proven.
    return "upstream_unavailable"


def _checkpoint(client: Any, endpoint: Any, lease_id: str, config: Config) -> None:
    session = encrypt_session(client.client.dumps(), config)
    endpoint.call("checkpoint", lease_id=lease_id, encrypted_session=session)


def _acquire(endpoint: Any) -> dict[str, Any]:
    result = endpoint.call("acquire")
    if result.get("busy") is True:
        return result
    try:
        UUID(result.get("lease_id"))
    except (ValueError, TypeError, AttributeError):
        raise WorkerFailure() from None
    return result


def _release(endpoint: Any, lease_id: str, error_code: str | None) -> bool:
    try:
        endpoint.call("release", lease_id=lease_id, error_code=error_code)
        return True
    except Exception:
        # The lease expires server-side. Cleanup transport trouble must not
        # incorrectly turn an already-imported batch into an import failure.
        return False


def _run_with_lease(
    config: Config, endpoint: Any, acquired: dict[str, Any], client: Any,
    days: int, *, already_logged_in: bool = False, today: date | None = None,
) -> RunResult:
    lease_id = acquired["lease_id"]
    error_code: str | None = None
    result = RunResult("upstream_unavailable")
    cleanup_pending = False
    guard = RefreshGuard(client)
    try:
        if not already_logged_in:
            envelope = acquired.get("encrypted_session")
            if envelope is None:
                raise WorkerFailure("not_connected")
            session = decrypt_session(envelope, config)
            login_error: Exception | None = None
            try:
                client.login(session)
            except Exception as error:
                login_error = error
            finally:
                # login() refreshes DI tokens before loading the profile. Keep a
                # valid rotated session even if that later network request fails.
                # A rejected login is not a candidate to replace the last session.
                if login_error is None or _error_code(login_error, guard) != "needs_login":
                    try:
                        candidate = _validate_session(client.client.dumps())
                        if login_error is None or candidate != session:
                            _checkpoint(client, endpoint, lease_id, config)
                    except WorkerFailure as persistence_error:
                        if persistence_error.code != "needs_login" or login_error is None:
                            raise
            if login_error is not None:
                raise login_error
        else:
            _checkpoint(client, endpoint, lease_id, config)
        today = today or datetime.now(ZoneInfo(config.timezone)).date()
        start = today - timedelta(days=days - 1)
        try:
            raw = client.get_calories_daily(start.isoformat(), today.isoformat())
            rows = normalize_rows(raw, start, today)
        finally:
            # A request may refresh credentials before failing. The replacement
            # token must survive even if no calorie records can be imported.
            _checkpoint(client, endpoint, lease_id, config)
        endpoint.call("import", lease_id=lease_id, rows=rows)
        result = RunResult("success", len(rows))
    except Exception as error:
        error_code = _error_code(error, guard)
        result = RunResult(error_code)
    finally:
        guard.restore()
        cleanup_pending = not _release(endpoint, lease_id, error_code)
    return RunResult(result.status, result.count, cleanup_pending)


def run_sync(
    config: Config, days: int = 8, *, endpoint: Any | None = None,
    client_factory: Callable[..., Any] = _new_garmin, today: date | None = None,
) -> RunResult:
    endpoint = endpoint or EndpointClient(config)
    acquired = _acquire(endpoint)
    if acquired.get("busy") is True:
        return RunResult("busy")
    if acquired.get("encrypted_session") is None:
        released = _release(endpoint, acquired["lease_id"], "not_connected")
        return RunResult("not_connected", cleanup_pending=not released)
    try:
        client = client_factory()
    except Exception:
        _release(endpoint, acquired["lease_id"], "upstream_unavailable")
        raise WorkerFailure() from None
    return _run_with_lease(config, endpoint, acquired, client, days, today=today)


def run_connect(
    config: Config, days: int = 30, *, endpoint: Any | None = None,
    client_factory: Callable[..., Any] = _new_garmin,
    input_email: Callable[[str], str] = input,
    input_secret: Callable[[str], str] = getpass,
    today: date | None = None,
) -> RunResult:
    # Obtain the interactive login before starting the limited-duration lease.
    email = input_email("Garmin email: ").strip()
    password = input_secret("Garmin password (hidden): ")
    client = client_factory(
        email=email, password=password,
        prompt_mfa=lambda: input_secret("Garmin MFA code (hidden): "),
    )
    password = ""
    # Garmin.login() honors GARMINTOKENS; suppress it so a pre-existing shell
    # setting cannot make this bootstrap write plaintext tokens to a file.
    previous_tokenstore = os.environ.pop("GARMINTOKENS", None)
    try:
        client.login()
        _validate_session(client.client.dumps())
    except Exception as error:
        return RunResult(_error_code(error, interactive=True))
    finally:
        if previous_tokenstore is not None:
            os.environ["GARMINTOKENS"] = previous_tokenstore
    endpoint = endpoint or EndpointClient(config)
    acquired = _acquire(endpoint)
    if acquired.get("busy") is True:
        return RunResult("busy")
    return _run_with_lease(
        config, endpoint, acquired, client, days, already_logged_in=True, today=today,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sync", "connect"))
    parser.add_argument("--config", help="Path to an owner-only JSON configuration file.")
    parser.add_argument("--days", type=int, help="Days to refresh (1–366; sync: 8, connect: 30).")
    args = parser.parse_args(argv)
    days = args.days if args.days is not None else (30 if args.command == "connect" else 8)
    if not 1 <= days <= 366:
        parser.error("--days must be between 1 and 366")
    # The dependency may put response snippets in debug logs. Never enable them.
    logging.basicConfig(level=logging.CRITICAL)
    logging.disable(logging.CRITICAL)
    try:
        config = load_config(args.config)
        result = run_connect(config, days) if args.command == "connect" else run_sync(config, days)
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:
        # Future dependency/transport errors may contain credentials or upstream
        # response bodies. Do not print exceptions or tracebacks in this CLI.
        print(PUBLIC_MESSAGES["upstream_unavailable"], file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Connection cancelled.", file=sys.stderr)
        return 1
    print(PUBLIC_MESSAGES[result.status].format(count=result.count))
    if result.cleanup_pending:
        print("Temporary sync lock cleanup is pending; the lock expires automatically.")
    if result.status in ("success", "not_connected"):
        return 0
    if result.status == "busy" and args.command == "sync":
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
