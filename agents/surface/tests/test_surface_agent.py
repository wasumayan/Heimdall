from agents.surface.surface_agent import (
    EnvironmentSnapshot,
    CookieRecord,
    ConsoleEvent,
    NetworkEvent,
    Severity,
    StorageEntry,
    SurfaceAgent,
    SurfaceAgentConfig,
)


def build_environment(
    href: str = "https://example.test/app",
    search: str = "",
    hash_fragment: str = "",
) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        route=href,
        timestamp=0.0,
        window_keys_before=["location"],
        window_keys_after=["location", "__HEIMDALL__"],
        new_globals=["__HEIMDALL__"],
        location={
            "href": href,
            "origin": "https://example.test",
            "pathname": "/app",
            "search": search,
            "hash": hash_fragment,
            "referrer": None,
            "title": "Example",
            "referrerPolicy": None,
        },
    )


def test_storage_jwt_triggers_critical():
    agent = SurfaceAgent(SurfaceAgentConfig())
    jwt_value = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvZSIsImlhdCI6MTUxNjIzOTAyMn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    entries = [
        StorageEntry(
            area="localStorage",
            key="auth_token",
            value=jwt_value,
        )
    ]
    findings = agent._analyze_storage(
        build_environment(),
        entries,
        agent.config.secret_redaction,
    )
    assert findings, "Expected a finding for JWT in localStorage"
    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].evidence["pattern"] == "jwt"


def test_cookie_without_http_only_flagged_high():
    agent = SurfaceAgent(SurfaceAgentConfig())
    cookies = [
        CookieRecord(
            name="sessionid",
            value="ghp_testpat_abcdefghijklmnopqrstuvwxyz123456",
            domain="example.test",
            path="/",
            secure=False,
            http_only=False,
            same_site="Lax",
            expires=None,
        )
    ]
    findings = agent._analyze_cookies(
        build_environment(),
        cookies,
        agent.config.secret_redaction,
    )
    assert findings, "Expected a finding for cookie missing HttpOnly"
    finding = findings[0]
    assert finding.severity in (Severity.CRITICAL, Severity.HIGH)
    assert finding.evidence["flags"]["HttpOnly"] is False


def test_url_token_detection_and_referrer_policy():
    agent = SurfaceAgent(SurfaceAgentConfig())
    env_with_token = EnvironmentSnapshot(
        route="https://app.example.test/#access_token=sk_test_PLACEHOLDER_DO_NOT_USE",
        timestamp=0.0,
        window_keys_before=[],
        window_keys_after=[],
        new_globals=[],
        location={
            "href": "https://app.example.test/#access_token=sk_test_PLACEHOLDER_DO_NOT_USE",
            "origin": "https://app.example.test",
            "pathname": "/",
            "search": "",
            "hash": "#access_token=sk_test_PLACEHOLDER_DO_NOT_USE",
            "referrer": None,
            "title": "App",
            "referrerPolicy": None,
        },
    )
    url_findings = agent._analyze_url(
        env_with_token,
        [],
        agent.config.secret_redaction,
    )
    token_findings = [f for f in url_findings if f.kind == "UrlExposure"]
    assert token_findings, "Expected URL exposure finding for hash token"
    assert token_findings[0].severity == Severity.CRITICAL

    env_without_token = build_environment(
        href="https://app.example.test/dashboard?view=summary",
        search="?view=summary",
    )
    policy_events = [
        NetworkEvent(
            direction="response",
            url="https://app.example.test/dashboard",
            status=200,
            headers={"referrer-policy": "unsafe-url"},
        )
    ]
    policy_findings = agent._analyze_url(
        env_without_token,
        policy_events,
        agent.config.secret_redaction,
    )
    referrer_findings = [f for f in policy_findings if f.kind == "ReferrerPolicy"]
    assert referrer_findings, "Expected referrer policy warning when policy is unsafe"
    assert referrer_findings[0].severity == Severity.HIGH


def test_console_log_secret_detected():
    agent = SurfaceAgent(SurfaceAgentConfig())
    console_events = [
        ConsoleEvent(
            type="log",
            text="Bearer ghp_examplepat_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
            location=None,
            timestamp=1.0,
        )
    ]
    findings = agent._analyze_console(
        build_environment(),
        console_events,
        agent.config.secret_redaction,
    )
    assert findings, "Expected console leak finding"
    assert findings[0].kind == "ConsoleLeak"
    assert findings[0].severity == Severity.CRITICAL



