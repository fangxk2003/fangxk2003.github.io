"""Credential-free tests for the worker's privacy and synchronization guarantees."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from datetime import date, datetime
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from garminconnect import GarminConnectAuthenticationError, GarminConnectConnectionError

import worker


CONFIG = worker.Config(
    "https://wjobefnfswnftvthybiv.supabase.co/functions/v1/calories-garmin-sync",
    "a1" * 32,
)
TODAY = date(2026, 10, 12)
LEASE_ID = "cd0474e7-bf27-44da-b45c-60af1d7e1c3a"


def token_json(access: str = "synthetic-access", refresh: str = "synthetic-refresh") -> str:
    return json.dumps({
        "di_token": access, "di_refresh_token": refresh, "di_client_id": "synthetic-client",
    })


class FakeEndpoint:
    def __init__(self, envelope: str | None = None, *, busy: bool = False) -> None:
        self.envelope = envelope
        self.busy = busy
        self.calls: list[tuple[str, dict]] = []

    def call(self, operation: str, **fields):
        self.calls.append((operation, fields))
        if operation == "acquire":
            return {"busy": True} if self.busy else {
                "lease_id": LEASE_ID, "encrypted_session": self.envelope,
            }
        if operation == "checkpoint":
            self.envelope = fields["encrypted_session"]
        return {"ok": True}


class FakeGarmin:
    def __init__(self, *, fetch_error: Exception | None = None, login_error: Exception | None = None):
        self.client = self
        self.session = token_json()
        self.fetch_error = fetch_error
        self.login_error = login_error
        self.login_arguments: list[tuple] = []
        self.requested_ranges: list[tuple[str, str]] = []
        self._di_token_url = "https://diauth.garmin.com/synthetic-test-token-url"
        self._http_post = lambda *args, **kwargs: None
        self.rows = [{"calendarDate": TODAY.isoformat(), "active": 510, "resting": 1900, "total": 1}]

    def dumps(self):
        return self.session

    def login(self, *arguments):
        self.login_arguments.append(arguments)
        if self.login_error:
            raise self.login_error
        self.session = token_json("after-login", "refresh-after-login")

    def get_calories_daily(self, start, end):
        self.requested_ranges.append((start, end))
        self.session = token_json("after-fetch", "refresh-after-fetch")
        if self.fetch_error:
            raise self.fetch_error
        return self.rows


class EncryptionTests(unittest.TestCase):
    def test_round_trip_and_random_nonce_without_plaintext(self):
        first = worker.encrypt_session(token_json(), CONFIG)
        second = worker.encrypt_session(token_json(), CONFIG)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("v1."))
        self.assertNotIn("synthetic-refresh", first)
        self.assertNotIn(CONFIG.credential, first)
        self.assertEqual(json.loads(worker.decrypt_session(first, CONFIG)), json.loads(token_json()))

    def test_wrong_credential_wrong_project_and_tampering_fail_closed(self):
        envelope = worker.encrypt_session(token_json(), CONFIG)
        for config in (
            worker.Config(CONFIG.endpoint, "b2" * 32),
            worker.Config("https://other-project.supabase.co/functions/v1/calories-garmin-sync", CONFIG.credential),
        ):
            with self.assertRaises(worker.WorkerFailure) as caught:
                worker.decrypt_session(envelope, config)
            self.assertEqual(caught.exception.code, "needs_login")
        offset = 25
        replacement = "A" if envelope[offset] != "A" else "B"
        altered = envelope[:offset] + replacement + envelope[offset + 1:]
        with self.assertRaises(worker.WorkerFailure):
            worker.decrypt_session(altered, CONFIG)

    def test_owner_is_bound_into_associated_data(self):
        envelope = worker.encrypt_session(token_json(), CONFIG)
        with patch.object(worker, "OWNER_ID", "different-owner"):
            with self.assertRaises(worker.WorkerFailure):
                worker.decrypt_session(envelope, CONFIG)

    def test_nonresumable_session_is_not_uploaded(self):
        for raw in ("not-json", "{}", json.dumps({"di_token": "cookie-only"}), token_json(refresh="")):
            with self.assertRaises(worker.WorkerFailure) as caught:
                worker.encrypt_session(raw, CONFIG)
            self.assertEqual(caught.exception.code, "needs_login")

    def test_extra_account_fields_are_removed(self):
        raw = json.loads(token_json())
        raw.update({"full_name": "should-not-persist", "password": "should-not-persist"})
        result = json.loads(worker.decrypt_session(worker.encrypt_session(json.dumps(raw), CONFIG), CONFIG))
        self.assertEqual(set(result), set(worker.SESSION_FIELDS))


class NormalizationTests(unittest.TestCase):
    def test_complete_values_and_zero_are_kept_and_total_is_recalculated(self):
        rows = worker.normalize_rows([
            {"calendarDate": "2026-10-12", "active": 0, "resting": 0, "total": 900, "heartRate": 75},
            {"calendarDate": "2026-10-11", "active": 513.5, "resting": 1801.25, "total": None},
        ], date(2026, 10, 11), TODAY)
        self.assertEqual(rows, [
            {"date": "2026-10-11", "active": 513.5, "resting": 1801.25, "total": 2314.75},
            {"date": "2026-10-12", "active": 0.0, "resting": 0.0, "total": 0.0},
        ])

    def test_missing_invalid_or_out_of_range_records_do_not_replace_good_data(self):
        invalid = [
            {"calendarDate": "2026-10-12", "active": 3, "resting": None},
            {"calendarDate": "2026-10-12", "active": None, "resting": 1900},
            {"calendarDate": "2026-10-12", "active": True, "resting": 1900},
            {"calendarDate": "2026-10-12", "active": float("nan"), "resting": 1900},
            {"calendarDate": "2026-10-12", "active": float("inf"), "resting": 1900},
            {"calendarDate": "2026-10-12", "active": -1, "resting": 1900},
            {"calendarDate": "2026-10-12", "active": "200", "resting": 1900},
            {"calendarDate": "2026-10-12", "active": 99_999, "resting": 1900},
            {"calendarDate": "2026-02-30", "active": 200, "resting": 1900},
            {"calendarDate": "20261012", "active": 200, "resting": 1900},
            {"calendarDate": "2026-10-13", "active": 200, "resting": 1900},
        ]
        good = {"calendarDate": "2026-10-11", "active": 200, "resting": 1900}
        rows = worker.normalize_rows(invalid + [good], date(2026, 10, 11), TODAY)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total"], 2100)

    def test_no_complete_data_is_distinct_from_zero(self):
        for raw in (None, {}, [], [{"calendarDate": "2026-10-12", "active": None, "resting": 1900}]):
            with self.assertRaises(worker.WorkerFailure) as caught:
                worker.normalize_rows(raw, TODAY, TODAY)
            self.assertEqual(caught.exception.code, "missing_data")

    def test_conflicting_duplicates_are_rejected(self):
        rows = [
            {"calendarDate": "2026-10-12", "active": 100, "resting": 1900},
            {"calendarDate": "2026-10-12", "active": 200, "resting": 1900},
        ]
        with self.assertRaises(worker.WorkerFailure):
            worker.normalize_rows(rows, TODAY, TODAY)


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = FakeEndpoint(worker.encrypt_session(token_json(), CONFIG))
        self.garmin = FakeGarmin()

    def run_sync(self):
        return worker.run_sync(CONFIG, endpoint=self.endpoint, client_factory=lambda: self.garmin, today=TODAY)

    def test_success_imports_only_calories_and_keeps_rotated_tokens(self):
        result = self.run_sync()
        self.assertEqual(result, worker.RunResult("success", 1))
        self.assertEqual(self.garmin.requested_ranges, [("2026-10-05", "2026-10-12")])
        self.assertEqual([op for op, _ in self.endpoint.calls], ["acquire", "checkpoint", "checkpoint", "import", "release"])
        self.assertEqual(json.loads(worker.decrypt_session(self.endpoint.envelope, CONFIG))["di_refresh_token"], "refresh-after-fetch")
        self.assertEqual(self.endpoint.calls[-2][1]["rows"], [
            {"date": "2026-10-12", "active": 510.0, "resting": 1900.0, "total": 2410.0},
        ])
        self.assertIsNone(self.endpoint.calls[-1][1]["error_code"])

    def test_fetch_failure_still_checkpoints_rotated_refresh_token_and_releases(self):
        self.garmin.fetch_error = GarminConnectConnectionError("sensitive-upstream-response")
        result = self.run_sync()
        self.assertEqual(result.status, "upstream_unavailable")
        self.assertNotIn("import", [op for op, _ in self.endpoint.calls])
        self.assertEqual(json.loads(worker.decrypt_session(self.endpoint.envelope, CONFIG))["di_refresh_token"], "refresh-after-fetch")
        self.assertEqual(self.endpoint.calls[-1][1]["error_code"], "upstream_unavailable")

    def test_auth_failure_does_not_destroy_previous_ciphertext(self):
        previous = self.endpoint.envelope
        self.garmin.login_error = GarminConnectAuthenticationError("sensitive-account-error")
        self.assertEqual(self.run_sync().status, "upstream_unavailable")
        self.assertEqual(self.endpoint.envelope, previous)
        self.assertEqual(self.endpoint.calls[-1][1]["error_code"], "upstream_unavailable")

    def test_swallowed_refresh_outage_then_profile_401_does_not_clear_session(self):
        class Response:
            status_code = 500
        self.garmin._http_post = lambda *args, **kwargs: Response()
        def login(*args):
            self.garmin._http_post(self.garmin._di_token_url, data={"grant_type": "refresh_token"})
            raise GarminConnectAuthenticationError("401 after swallowed refresh outage")
        self.garmin.login = login
        previous = self.endpoint.envelope
        self.assertEqual(self.run_sync().status, "upstream_unavailable")
        self.assertEqual(self.endpoint.envelope, previous)
        self.assertEqual(self.endpoint.calls[-1][1]["error_code"], "upstream_unavailable")

    def test_swallowed_refresh_timeout_then_profile_401_does_not_clear_session(self):
        def post(*args, **kwargs):
            raise TimeoutError("sensitive network details")
        self.garmin._http_post = post
        def login(*args):
            try:
                self.garmin._http_post(self.garmin._di_token_url, data={"grant_type": "refresh_token"})
            except TimeoutError:
                pass
            raise GarminConnectAuthenticationError("401 after swallowed timeout")
        self.garmin.login = login
        previous = self.endpoint.envelope
        self.assertEqual(self.run_sync().status, "upstream_unavailable")
        self.assertEqual(self.endpoint.envelope, previous)

    def test_verified_refresh_rejection_requests_new_login(self):
        class Response:
            def __init__(self, status, error):
                self.status_code = status
                self.error = error
            def json(self):
                return {"error": self.error}
        for status, error in ((401, None), (403, None), (400, "invalid_grant"), (400, "invalid_token")):
            self.endpoint = FakeEndpoint(worker.encrypt_session(token_json(), CONFIG))
            self.garmin = FakeGarmin()
            self.garmin._http_post = lambda *args, **kwargs: Response(status, error)
            def login(*args):
                self.garmin._http_post(self.garmin._di_token_url, data={"grant_type": "refresh_token"})
                raise GarminConnectAuthenticationError("refresh rejected")
            self.garmin.login = login
            previous = self.endpoint.envelope
            self.assertEqual(self.run_sync().status, "needs_login")
            self.assertEqual(self.endpoint.envelope, previous)
            self.assertEqual(self.endpoint.calls[-1][1]["error_code"], "needs_login")

    def test_login_refresh_then_profile_network_failure_preserves_new_token(self):
        def login(*arguments):
            self.garmin.session = token_json("login-partial", "rotated-before-profile-failure")
            raise GarminConnectConnectionError("private profile response must never be logged")
        self.garmin.login = login
        self.assertEqual(self.run_sync().status, "upstream_unavailable")
        self.assertEqual(json.loads(worker.decrypt_session(self.endpoint.envelope, CONFIG))["di_refresh_token"], "rotated-before-profile-failure")
        self.assertEqual([op for op, _ in self.endpoint.calls], ["acquire", "checkpoint", "release"])

    def test_missing_data_never_imports_a_zero(self):
        self.garmin.rows = [{"calendarDate": "2026-10-12", "active": 400, "resting": None, "total": 400}]
        self.assertEqual(self.run_sync().status, "missing_data")
        self.assertNotIn("import", [op for op, _ in self.endpoint.calls])
        self.assertEqual(self.endpoint.calls[-1][1]["error_code"], "missing_data")

    def test_busy_acquire_does_not_start_garmin_or_release_someone_elses_lease(self):
        endpoint = FakeEndpoint(busy=True)
        def should_not_run():
            self.fail("Garmin was initialized while another worker owns the lease")
        result = worker.run_sync(CONFIG, endpoint=endpoint, client_factory=should_not_run)
        self.assertEqual(result.status, "busy")
        self.assertEqual([op for op, _ in endpoint.calls], ["acquire"])

    def test_unconnected_run_is_harmless_without_garmin_login(self):
        endpoint = FakeEndpoint()
        result = worker.run_sync(CONFIG, endpoint=endpoint, client_factory=lambda: self.fail("Should not initialize Garmin"))
        self.assertEqual(result.status, "not_connected")
        self.assertEqual([op for op, _ in endpoint.calls], ["acquire", "release"])

    def test_constructing_client_failure_releases_lease(self):
        def fail():
            raise RuntimeError("sensitive internal details")
        with self.assertRaises(worker.WorkerFailure):
            worker.run_sync(CONFIG, endpoint=self.endpoint, client_factory=fail)
        self.assertEqual(self.endpoint.calls[-1][0], "release")

    def test_connect_logs_in_before_acquire_and_suppresses_plaintext_tokenstore(self):
        seen = []
        original_login = self.garmin.login
        def login(*args):
            self.assertNotIn("GARMINTOKENS", os.environ)
            self.assertEqual(self.endpoint.calls, [])
            original_login(*args)
        self.garmin.login = login
        def factory(**kwargs):
            seen.append(kwargs)
            return self.garmin
        with patch.dict(os.environ, {"GARMINTOKENS": "/synthetic/should-not-write"}):
            result = worker.run_connect(
                CONFIG, endpoint=self.endpoint, client_factory=factory,
                input_email=lambda _: " test@example.invalid ",
                input_secret=lambda _: "synthetic-password", today=TODAY,
            )
            self.assertEqual(os.environ["GARMINTOKENS"], "/synthetic/should-not-write")
        self.assertEqual(result.status, "success")
        self.assertEqual(seen[0]["email"], "test@example.invalid")
        self.assertEqual(self.garmin.requested_ranges, [("2026-09-13", "2026-10-12")])

    def test_bad_ciphertext_releases_with_needs_login(self):
        self.endpoint.envelope = "v1.corrupted"
        self.assertEqual(self.run_sync().status, "needs_login")
        self.assertEqual(self.endpoint.calls[-1][1]["error_code"], "needs_login")

    def test_release_network_failure_preserves_confirmed_import_result(self):
        original_call = self.endpoint.call
        def call(operation, **fields):
            if operation == "release":
                raise worker.WorkerFailure()
            return original_call(operation, **fields)
        self.endpoint.call = call
        result = self.run_sync()
        self.assertEqual(result.status, "success")
        self.assertEqual(result.count, 1)
        self.assertTrue(result.cleanup_pending)

    def test_release_network_failure_preserves_original_auth_error(self):
        original_call = self.endpoint.call
        def call(operation, **fields):
            if operation == "release":
                raise worker.WorkerFailure()
            return original_call(operation, **fields)
        self.endpoint.call = call
        self.garmin.login_error = GarminConnectAuthenticationError("sensitive-auth-error")
        result = self.run_sync()
        self.assertEqual(result.status, "upstream_unavailable")
        self.assertTrue(result.cleanup_pending)


class TransportTests(unittest.TestCase):
    def test_lost_release_response_then_stale_lease_is_successful_cleanup(self):
        endpoint = worker.EndpointClient(CONFIG)
        with patch.object(worker, "urlopen", side_effect=[
            URLError("sensitive transport details"),
            HTTPError(CONFIG.endpoint, 409, "stale lease", None, None),
        ]), patch.object(worker.time, "sleep"):
            self.assertEqual(endpoint.call("release", lease_id=LEASE_ID, error_code=None), {"ok": True})

    def test_stale_lease_on_write_still_fails(self):
        for operation in ("checkpoint", "import"):
            with patch.object(worker, "urlopen", side_effect=HTTPError(CONFIG.endpoint, 409, "stale lease", None, None)):
                with self.assertRaises(worker.WorkerFailure):
                    worker.EndpointClient(CONFIG).call(operation, lease_id=LEASE_ID)


class ConfigurationAndOutputTests(unittest.TestCase):
    def test_config_requires_private_permissions_and_environment_can_override_timezone(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"endpoint": CONFIG.endpoint, "credential": CONFIG.credential}))
            path.chmod(0o600)
            result = worker.load_config(str(path), {"GARMIN_TIMEZONE": "Asia/Shanghai"})
            self.assertEqual(result.timezone, "Asia/Shanghai")
            path.chmod(0o644)
            with self.assertRaises(worker.ConfigurationError):
                worker.load_config(str(path), {})

    def test_config_rejects_symlinks_and_non_supabase_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text("{}")
            path.chmod(0o600)
            alias = Path(folder) / "alias.json"
            alias.symlink_to(path)
            with self.assertRaises(worker.ConfigurationError):
                worker.load_config(str(alias), {})
        for endpoint in ("http://example.com", "https://attacker.invalid/sync", "https://supabase.co.evil.invalid/sync"):
            with self.assertRaises(worker.ConfigurationError):
                worker.Config(endpoint, CONFIG.credential).validate()

    def test_cli_output_never_contains_source_errors_credentials_or_values(self):
        out = io.StringIO()
        with patch.object(worker, "load_config", return_value=CONFIG), patch.object(
            worker, "run_sync", return_value=worker.RunResult("upstream_unavailable"),
        ), redirect_stdout(out), redirect_stderr(out):
            code = worker.main(["sync"])
        self.assertEqual(code, 1)
        self.assertNotIn(CONFIG.credential, out.getvalue())
        self.assertNotIn(CONFIG.endpoint, out.getvalue())
        self.assertEqual(out.getvalue().strip(), worker.PUBLIC_MESSAGES["upstream_unavailable"])

    def test_cli_initial_not_connected_is_a_successful_noop(self):
        with patch.object(worker, "load_config", return_value=CONFIG), patch.object(
            worker, "run_sync", return_value=worker.RunResult("not_connected"),
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(["sync"]), 0)

    def test_unexpected_dependency_exception_is_redacted(self):
        output = io.StringIO()
        with patch.object(worker, "load_config", return_value=CONFIG), patch.object(
            worker, "run_sync", side_effect=RuntimeError(CONFIG.credential + " private health details"),
        ), redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(worker.main(["sync"]), 1)
        self.assertNotIn(CONFIG.credential, output.getvalue())
        self.assertNotIn("private health", output.getvalue())

    def test_clock_uses_configured_timezone_for_date_range(self):
        endpoint = FakeEndpoint(worker.encrypt_session(token_json(), CONFIG))
        garmin = FakeGarmin()
        with patch.object(worker, "datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 12, 0, 30)
            self.assertEqual(worker.run_sync(CONFIG, endpoint=endpoint, client_factory=lambda: garmin).status, "success")
        self.assertEqual(clock.now.call_args.args[0].key, "Europe/Stockholm")
        self.assertEqual(garmin.requested_ranges[-1][1], "2026-10-12")


if __name__ == "__main__":
    unittest.main()
