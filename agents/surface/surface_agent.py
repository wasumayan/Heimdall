"""Surface layer secret exposure agent for Heimdall.

This module runs within a browser context (Playwright-powered) to analyze a
running web application for exposed credentials or sensitive material. It
intentionally mirrors the structure of the network and injection agents:
  * a Severity enum shared across findings
  * dataclass-based findings and result payloads
  * a high-level agent facade with a CLI entrypoint
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from playwright.sync_api import Browser, BrowserContext, Page, Request, Response, sync_playwright  # type: ignore[import]
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Browser = BrowserContext = Page = Request = Response = None  # type: ignore[assignment]
    sync_playwright = None  # type: ignore[assignment]


class Severity(str, Enum):
    """Consistent severity labels for downstream consumers."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


@dataclass
class SurfaceFinding:
    """Represents a secret-exposure finding surfaced by the agent."""

    id: str
    kind: str
    severity: Severity
    confidence: float
    title: str
    description: str
    recommendation: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


@dataclass
class EnvironmentSnapshot:
    """Minimal snapshot of the browser environment before/after scanning."""

    route: str
    timestamp: float
    window_keys_before: List[str]
    window_keys_after: List[str]
    new_globals: List[str]
    location: Dict[str, Any]


@dataclass
class StorageEntry:
    """Represents a key/value pair discovered in web storage."""

    area: str
    key: str
    value: Optional[str]
    truncated: bool = False


@dataclass
class CookieRecord:
    """Represents a cookie accessible to JavaScript."""

    name: str
    value: str
    domain: str
    path: str
    secure: bool
    http_only: bool
    same_site: Optional[str]
    expires: Optional[float]


@dataclass
class ConsoleEvent:
    """Represents a console log captured during observation."""

    type: str
    text: str
    location: Optional[Dict[str, Any]]
    timestamp: float


@dataclass
class NetworkEvent:
    """Represents a captured network request or response."""

    direction: str  # "request" or "response"
    url: str
    method: Optional[str] = None
    status: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    body_preview: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SurfaceAgentConfig:
    """Runtime configuration for the surface agent."""

    headless: bool = True
    wait_until: str = "networkidle"
    post_navigation_wait: float = 2.0
    observation_timeout: float = 5.0
    user_agent: str = "Heimdall-Surface-Agent/0.1"
    ignore_https_errors: bool = False
    max_storage_value_chars: int = 4096
    max_console_payload_chars: int = 4096
    pattern_confidence_floor: float = 0.55
    sensitive_key_indicators: Tuple[str, ...] = (
        "token",
        "auth",
        "secret",
        "key",
        "session",
        "bearer",
        "passwd",
    )
    secret_redaction: Tuple[int, int] = (4, 4)


@dataclass
class SurfaceScanResult:
    """Top-level payload returned by the agent."""

    target: str
    resolved_url: str
    environment: EnvironmentSnapshot
    storage_entries: List[StorageEntry]
    cookies: List[CookieRecord]
    console_events: List[ConsoleEvent]
    network_events: List[NetworkEvent]
    findings: List[SurfaceFinding]
    generated_at: float = field(default_factory=time.time)
    _summary_cache: Optional[str] = field(default=None, init=False, repr=False)
    _summary_source: str = field(default="heuristic", init=False, repr=False)

    @property
    def summary(self) -> Dict[str, Any]:
        counts = {sev.value: 0 for sev in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return {
            "target": self.target,
            "resolved_url": self.resolved_url,
            "finding_count": len(self.findings),
            "severity_counts": counts,
            "new_globals": self.environment.new_globals,
        }

    @property
    def summary_text(self) -> str:
        if self._summary_cache is not None:
            return self._summary_cache

        summary, source = LLMSummaryGenerator.generate(self)
        self._summary_cache = summary
        self._summary_source = source
        return summary

    @property
    def summary_source(self) -> str:
        _ = self.summary_text
        return self._summary_source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": "SurfaceAgent",
            "target": self.target,
            "resolved_url": self.resolved_url,
            "environment": asdict(self.environment),
            "storage_entries": [asdict(entry) for entry in self.storage_entries],
            "cookies": [asdict(cookie) for cookie in self.cookies],
            "console_events": [asdict(event) for event in self.console_events],
            "network_events": [asdict(event) for event in self.network_events],
            "findings": [finding.to_dict() for finding in self.findings],
            "summary": self.summary,
            "summary_text": self.summary_text,
            "summary_source": self.summary_source,
            "generated_at": self.generated_at,
        }


class LLMSummaryGenerator:
    """Produces natural-language summaries with a deterministic fallback."""

    _DEFAULT_MODEL = "grok-2-latest"
    _DEFAULT_TIMEOUT = 30

    @classmethod
    def generate(cls, scan: SurfaceScanResult) -> Tuple[str, str]:
        llm_summary = cls._try_xai_summary(scan)
        if llm_summary:
            return llm_summary, "xai"
        return cls._heuristic_summary(scan), "heuristic"

    @classmethod
    def _try_xai_summary(cls, scan: SurfaceScanResult) -> Optional[str]:
        try:  # pragma: no cover - optional path
            from openai import OpenAI  # type: ignore[import]
        except ImportError:  # pragma: no cover
            return None

        api_key = cls._env("TWEuw27mz6ciyYF4Pugy8k9qmNv0CAKdGeAAe1FIkeJX")  # obfuscated guard
        if not api_key:
            return None

        if not scan.findings:
            return None

        model = cls._env("XAI_SUMMARY_MODEL", cls._DEFAULT_MODEL)
        base_url = cls._env("XAI_API_BASE", "https://api.x.ai/v1")
        timeout = float(cls._env("XAI_SUMMARY_TIMEOUT", cls._DEFAULT_TIMEOUT))

        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception:  # pragma: no cover - API init issues
            return None

        payload = cls._build_payload(scan)
        system_prompt = (
            "You are a security analyst summarizing frontend secret exposure."
            " Write 2-4 sentences, grouping similar findings, citing severities"
            " and affected storage areas or routes. Finish with remediation advice."
        )
        user_prompt = json.dumps(payload, indent=2)

        try:
            completion = client.chat.completions.create(  # pragma: no cover - optional
                model=model,
                timeout=timeout,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None

        if not completion.choices:
            return None
        message = completion.choices[0].message
        content = message.content if message else None
        return content.strip() if content else None

    @staticmethod
    def _env(key: str, default: Optional[str] = None) -> Optional[str]:
        import os

        return os.environ.get(key, default)

    @staticmethod
    def _build_payload(scan: SurfaceScanResult) -> Dict[str, Any]:
        findings = [
            {
                "id": finding.id,
                "kind": finding.kind,
                "severity": finding.severity.value,
                "confidence": round(finding.confidence, 2),
                "title": finding.title,
                "description": finding.description,
                "location": finding.evidence.get("location"),
            }
            for finding in scan.findings
        ]
        return {
            "target": scan.target,
            "resolved_url": scan.resolved_url,
            "findings": findings,
            "new_globals": scan.environment.new_globals,
        }

    @staticmethod
    def _heuristic_summary(scan: SurfaceScanResult) -> str:
        if not scan.findings:
            return (
                f"No frontend secret exposure detected for {scan.environment.route}."
            )

        top_finding = min(
            scan.findings,
            key=lambda finding: (
                _SEVERITY_RANK.get(finding.severity, 99),
                (1 - finding.confidence),
            ),
        )
        location = top_finding.evidence.get("location")
        loc_clause = f" at {location}" if location else ""
        return (
            f"Found {len(scan.findings)} exposure issue(s). "
            f"Highest severity {top_finding.severity.value.upper()} - "
            f"{top_finding.title}{loc_clause}."
        )


@dataclass
class SecretPattern:
    """Definition of a secret pattern searched across artifacts."""

    id: str
    regex: re.Pattern[str]
    description: str
    default_severity: Severity
    base_confidence: float


@dataclass
class SecretMatch:
    """Concrete match for a secret pattern."""

    pattern_id: str
    match: str
    redacted: str
    confidence: float
    severity: Severity
    pattern_description: str


class SecretDetector:
    """Searches arbitrary strings for secret-like material."""

    _PATTERNS: Tuple[SecretPattern, ...] = (
        SecretPattern(
            id="jwt",
            regex=re.compile(
                r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"
            ),
            description="JWT token-like value",
            default_severity=Severity.CRITICAL,
            base_confidence=0.96,
        ),
        SecretPattern(
            id="sk_live",
            regex=re.compile(r"\bsk_(?:live|test)_[0-9a-zA-Z]{16,}\b"),
            description="Stripe-style secret key",
            default_severity=Severity.CRITICAL,
            base_confidence=0.95,
        ),
        SecretPattern(
            id="aws_access_key",
            regex=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            description="AWS access key ID",
            default_severity=Severity.HIGH,
            base_confidence=0.92,
        ),
        SecretPattern(
            id="google_api_key",
            regex=re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b"),
            description="Google API key",
            default_severity=Severity.HIGH,
            base_confidence=0.9,
        ),
        SecretPattern(
            id="slack_token",
            regex=re.compile(r"\bxox[baprs]-[0-9a-zA-Z]{10,}\b"),
            description="Slack token",
            default_severity=Severity.CRITICAL,
            base_confidence=0.92,
        ),
        SecretPattern(
            id="github_pat",
            regex=re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{36}\b"),
            description="GitHub personal access token",
            default_severity=Severity.CRITICAL,
            base_confidence=0.93,
        ),
        SecretPattern(
            id="long_hex",
            regex=re.compile(r"\b[a-f0-9]{32,}\b", re.IGNORECASE),
            description="Long hexadecimal token",
            default_severity=Severity.HIGH,
            base_confidence=0.75,
        ),
        SecretPattern(
            id="base64_secret",
            regex=re.compile(
                r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b"
            ),
            description="Base64-like secret",
            default_severity=Severity.MEDIUM,
            base_confidence=0.7,
        ),
        SecretPattern(
            id="basic_auth",
            regex=re.compile(r"\bBasic\s+[A-Za-z0-9+/=]{20,}"),
            description="Basic auth credential",
            default_severity=Severity.CRITICAL,
            base_confidence=0.88,
        ),
    )

    @classmethod
    def scan_text(cls, value: str, redaction: Tuple[int, int]) -> List[SecretMatch]:
        matches: List[SecretMatch] = []
        if not value:
            return matches

        for pattern in cls._PATTERNS:
            for match in pattern.regex.findall(value):
                redacted = redact(match, *redaction)
                matches.append(
                    SecretMatch(
                        pattern_id=pattern.id,
                        match=match,
                        redacted=redacted,
                        confidence=pattern.base_confidence,
                        severity=pattern.default_severity,
                        pattern_description=pattern.description,
                    )
                )
        return matches

    @classmethod
    def heuristic_key_flag(
        cls, key: str, value: Optional[str], indicators: Sequence[str], redaction: Tuple[int, int]
    ) -> Optional[SecretMatch]:
        if value is None:
            return None
        lowered = key.lower()
        if not any(indicator in lowered for indicator in indicators):
            return None
        if len(value) < 12:
            return None
        redacted = redact(value, *redaction)
        severity = Severity.HIGH if "refresh" in lowered or "session" in lowered else Severity.MEDIUM
        confidence = 0.65 if "token" in lowered else 0.6
        return SecretMatch(
            pattern_id="heuristic_key_indicator",
            match=value,
            redacted=redacted,
            confidence=confidence,
            severity=severity,
            pattern_description=f"Key name '{key}' suggests sensitive data",
        )


def redact(value: str, prefix: int, suffix: int) -> str:
    """Show only the first/last few characters of a secret."""
    if len(value) <= prefix + suffix:
        return value
    return f"{value[:prefix]}…{value[-suffix:]}"


class SurfaceAgent:
    """Facade that orchestrates browser-driven surface scanning."""

    def __init__(self, config: Optional[SurfaceAgentConfig] = None):
        self.config = config or SurfaceAgentConfig()

    def scan(self, url: str) -> SurfaceScanResult:
        if sync_playwright is None:
            raise RuntimeError(
                "Playwright is required for the surface agent. Install via "
                "`pip install playwright` and run `playwright install chromium`."
            )

        storage_entries: List[StorageEntry] = []
        cookies: List[CookieRecord] = []
        console_events: List[ConsoleEvent] = []
        network_events: List[NetworkEvent] = []

        with sync_playwright() as p:
            browser = self._launch_browser(p)
            try:
                context = browser.new_context(
                    user_agent=self.config.user_agent,
                    ignore_https_errors=self.config.ignore_https_errors,
                )
                page = context.new_page()
                window_keys_before = self._safe_eval_list(page, "Object.keys(window)")

                self._attach_observers(page, console_events, network_events)

                page.goto(url, wait_until=self.config.wait_until)
                if self.config.post_navigation_wait > 0:
                    page.wait_for_timeout(self.config.post_navigation_wait * 1000)
                if self.config.observation_timeout > 0:
                    page.wait_for_timeout(self.config.observation_timeout * 1000)

                environment = self._capture_environment(page, window_keys_before)
                storage_entries = self._collect_storage(page)
                cookies = self._collect_cookies(context)

                findings = self._analyze_findings(
                    environment,
                    storage_entries,
                    cookies,
                    console_events,
                    network_events,
                )

                resolved_url = page.url
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

        return SurfaceScanResult(
            target=url,
            resolved_url=resolved_url,
            environment=environment,
            storage_entries=storage_entries,
            cookies=cookies,
            console_events=console_events,
            network_events=network_events,
            findings=findings,
        )

    def _launch_browser(self, playwright) -> Browser:
        browser = playwright.chromium.launch(headless=self.config.headless)
        return browser

    def _safe_eval_list(self, page: Page, script: str) -> List[str]:
        try:
            result = page.evaluate(script)
        except Exception:
            return []
        if isinstance(result, list):
            return [str(item) for item in result]
        return []

    def _attach_observers(
        self,
        page: Page,
        console_events: List[ConsoleEvent],
        network_events: List[NetworkEvent],
    ) -> None:
        redaction = self.config.secret_redaction

        def on_console(message):
            try:
                location = message.location
            except Exception:
                location = None
            text = message.text
            if text and len(text) > self.config.max_console_payload_chars:
                text = text[: self.config.max_console_payload_chars] + "…"
            console_events.append(
                ConsoleEvent(
                    type=message.type,
                    text=text,
                    location=location,
                    timestamp=time.time(),
                )
            )

        def on_request(request: Request):
            headers = {k.lower(): v for k, v in request.headers.items()}
            body_preview: Optional[str] = None
            try:
                post_data = request.post_data
                if post_data:
                    body_preview = post_data[:512] + ("…" if len(post_data) > 512 else "")
            except Exception:
                body_preview = None

            network_events.append(
                NetworkEvent(
                    direction="request",
                    url=request.url,
                    method=request.method,
                    headers=headers,
                    body_preview=body_preview,
                )
            )

        def on_response(response: Response):
            try:
                headers = {k.lower(): v for k, v in response.headers.items()}
                status = response.status
            except Exception:
                headers = {}
                status = None

            network_events.append(
                NetworkEvent(
                    direction="response",
                    url=response.url,
                    status=status,
                    headers=headers,
                )
            )

        page.on("console", on_console)
        page.on("request", on_request)
        page.on("response", on_response)

    def _capture_environment(self, page: Page, window_keys_before: List[str]) -> EnvironmentSnapshot:
        window_keys_after = self._safe_eval_list(page, "Object.keys(window)")
        new_globals = sorted(set(window_keys_after) - set(window_keys_before))
        location = page.evaluate(
            """() => ({
                href: window.location.href,
                origin: window.location.origin,
                pathname: window.location.pathname,
                search: window.location.search,
                hash: window.location.hash,
                referrer: document.referrer || null,
                title: document.title || null,
                referrerPolicy: (document.querySelector('meta[name="referrer"]')?.content) || null
            })"""
        )
        return EnvironmentSnapshot(
            route=str(location.get("href", page.url)),
            timestamp=time.time(),
            window_keys_before=window_keys_before,
            window_keys_after=window_keys_after,
            new_globals=new_globals,
            location=location,
        )

    def _collect_storage(self, page: Page) -> List[StorageEntry]:
        payload = page.evaluate(
            f"""(maxChars) => {{
                const areas = [];
                const serializer = (areaName, storage) => {{
                    try {{
                        const entries = [];
                        if (!storage) {{
                            return entries;
                        }}
                        const length = storage.length;
                        for (let i = 0; i < length; i++) {{
                            const key = storage.key(i);
                            const rawValue = storage.getItem(key);
                            if (rawValue === null || rawValue === undefined) {{
                                entries.push({{ area: areaName, key, value: null, truncated: false }});
                                continue;
                            }}
                            const value = String(rawValue);
                            const truncated = value.length > maxChars;
                            entries.push({{
                                area: areaName,
                                key,
                                value: truncated ? value.slice(0, maxChars) : value,
                                truncated,
                            }});
                        }}
                        return entries;
                    }} catch (err) {{
                        return [{{ area: areaName, key: "__error__", value: String(err), truncated: false }}];
                    }}
                }};

                try {{
                    areas.push(...serializer("localStorage", window.localStorage));
                }} catch (err) {{
                    areas.push({{ area: "localStorage", key: "__error__", value: String(err), truncated: false }});
                }}
                try {{
                    areas.push(...serializer("sessionStorage", window.sessionStorage));
                }} catch (err) {{
                    areas.push({{ area: "sessionStorage", key: "__error__", value: String(err), truncated: false }});
                }}
                return areas;
            }}"""
            ,
            self.config.max_storage_value_chars,
        )

        entries: List[StorageEntry] = []
        if isinstance(payload, list):
            for item in payload:
                area = str(item.get("area"))
                key = str(item.get("key"))
                value = item.get("value")
                truncated = bool(item.get("truncated", False))
                entries.append(
                    StorageEntry(
                        area=area,
                        key=key,
                        value=None if value is None else str(value),
                        truncated=truncated,
                    )
                )
        return entries

    def _collect_cookies(self, context: BrowserContext) -> List[CookieRecord]:
        cookies: List[CookieRecord] = []
        try:
            context_cookies = context.cookies()
        except Exception:
            context_cookies = []
        for cookie in context_cookies:
            cookies.append(
                CookieRecord(
                    name=cookie.get("name", ""),
                    value=cookie.get("value", ""),
                    domain=cookie.get("domain", ""),
                    path=cookie.get("path", ""),
                    secure=bool(cookie.get("secure", False)),
                    http_only=bool(cookie.get("httpOnly", False)),
                    same_site=cookie.get("sameSite"),
                    expires=cookie.get("expires"),
                )
            )
        return cookies

    def _analyze_findings(
        self,
        environment: EnvironmentSnapshot,
        storage_entries: List[StorageEntry],
        cookies: List[CookieRecord],
        console_events: List[ConsoleEvent],
        network_events: List[NetworkEvent],
    ) -> List[SurfaceFinding]:
        findings: List[SurfaceFinding] = []
        redaction = self.config.secret_redaction

        findings.extend(
            self._analyze_storage(environment, storage_entries, redaction)
        )
        findings.extend(
            self._analyze_cookies(environment, cookies, redaction)
        )
        findings.extend(
            self._analyze_console(environment, console_events, redaction)
        )
        findings.extend(
            self._analyze_network(environment, network_events, redaction)
        )
        findings.extend(
            self._analyze_url(environment, network_events, redaction)
        )
        findings.extend(
            self._analyze_globals(environment)
        )

        return findings

    def _analyze_storage(
        self,
        environment: EnvironmentSnapshot,
        storage_entries: List[StorageEntry],
        redaction: Tuple[int, int],
    ) -> List[SurfaceFinding]:
        findings: List[SurfaceFinding] = []
        for entry in storage_entries:
            if entry.key == "__error__":
                findings.append(
                    SurfaceFinding(
                        id=f"surface_storage_error_{entry.area}",
                        kind="StorageAccess",
                        severity=Severity.MEDIUM,
                        confidence=0.5,
                        title=f"{entry.area} access failed",
                        description=f"Unable to enumerate {entry.area}: {entry.value}",
                        recommendation=f"Ensure {entry.area} is accessible or handled gracefully.",
                        evidence={
                            "source": entry.area,
                            "error": entry.value,
                        },
                    )
                )
                continue

            matches = []
            seen_patterns: set[Tuple[str, str]] = set()
            if entry.value:
                matches = SecretDetector.scan_text(entry.value, redaction)
                heuristic = SecretDetector.heuristic_key_flag(
                    entry.key,
                    entry.value,
                    self.config.sensitive_key_indicators,
                    redaction,
                )
                if heuristic:
                    matches.append(heuristic)

            for match in matches:
                if match.confidence < self.config.pattern_confidence_floor:
                    continue
                key_pattern = (match.pattern_id, match.redacted)
                if key_pattern in seen_patterns:
                    continue
                seen_patterns.add(key_pattern)
                severity = match.severity
                if entry.area == "sessionStorage" and "refresh" in entry.key.lower():
                    severity = Severity.CRITICAL
                elif entry.area == "localStorage" and severity.value in ("medium", "info"):
                    severity = Severity.HIGH
                findings.append(
                    SurfaceFinding(
                        id=f"surface_storage_{entry.area}_{entry.key}_{match.pattern_id}",
                        kind="StorageExposure",
                        severity=severity,
                        confidence=match.confidence,
                        title=f"Sensitive value stored in {entry.area}",
                        description=f"{match.pattern_description} detected at key '{entry.key}'.",
                        recommendation="Move secrets to HttpOnly cookies or server-side storage.",
                        evidence={
                            "location": {
                                "area": entry.area,
                                "key": entry.key,
                                "route": environment.route,
                            },
                            "valuePreview": match.redacted,
                            "pattern": match.pattern_id,
                            "truncated": entry.truncated,
                        },
                    )
                )
        return findings

    def _analyze_cookies(
        self,
        environment: EnvironmentSnapshot,
        cookies: List[CookieRecord],
        redaction: Tuple[int, int],
    ) -> List[SurfaceFinding]:
        findings: List[SurfaceFinding] = []
        for cookie in cookies:
            if cookie.http_only:
                continue

            severity = Severity.HIGH
            confidence = 0.85

            matches = [
                m
                for m in SecretDetector.scan_text(cookie.value, redaction)
                if m.confidence >= self.config.pattern_confidence_floor
            ]
            heuristic = SecretDetector.heuristic_key_flag(
                cookie.name,
                cookie.value,
                self.config.sensitive_key_indicators,
                redaction,
            )
            if heuristic:
                if heuristic.confidence >= self.config.pattern_confidence_floor:
                    matches.append(heuristic)

            if matches:
                matches.sort(
                    key=lambda m: (
                        _SEVERITY_RANK.get(m.severity, 99),
                        -m.confidence,
                    )
                )
                match = matches[0]
                severity = Severity.CRITICAL if match.severity == Severity.CRITICAL else Severity.HIGH
                confidence = max(match.confidence, confidence)
                preview = match.redacted
                pattern_id = match.pattern_id
            else:
                preview = redact(cookie.value, *redaction) if cookie.value else ""
                pattern_id = "cookie_no_pattern"

            findings.append(
                SurfaceFinding(
                    id=f"surface_cookie_{cookie.name}",
                    kind="CookieSettings",
                    severity=severity,
                    confidence=confidence,
                    title=f"Cookie '{cookie.name}' accessible to JavaScript",
                    description="Cookie lacking HttpOnly flag exposes session data to frontend scripts.",
                    recommendation="Set HttpOnly, Secure, and SameSite appropriately on sensitive cookies.",
                    evidence={
                        "location": {
                            "cookie": cookie.name,
                            "domain": cookie.domain,
                            "route": environment.route,
                        },
                        "flags": {
                            "HttpOnly": cookie.http_only,
                            "Secure": cookie.secure,
                            "SameSite": cookie.same_site,
                        },
                        "valuePreview": preview,
                        "pattern": pattern_id,
                    },
                )
            )
        return findings

    def _analyze_console(
        self,
        environment: EnvironmentSnapshot,
        console_events: List[ConsoleEvent],
        redaction: Tuple[int, int],
    ) -> List[SurfaceFinding]:
        findings: List[SurfaceFinding] = []
        for event in console_events:
            matches = SecretDetector.scan_text(event.text, redaction)
            if not matches:
                continue
            match = matches[0]
            if match.confidence < self.config.pattern_confidence_floor:
                continue
            findings.append(
                SurfaceFinding(
                    id=f"surface_console_{int(event.timestamp * 1000)}",
                    kind="ConsoleLeak",
                    severity=match.severity,
                    confidence=match.confidence,
                    title="Sensitive value logged to console",
                    description=f"{match.pattern_description} observed via console.{event.type}.",
                    recommendation="Strip sensitive data from client-side logs.",
                    evidence={
                        "location": {
                            "route": environment.route,
                            "consoleType": event.type,
                        },
                        "valuePreview": match.redacted,
                        "pattern": match.pattern_id,
                    },
                )
            )
        return findings

    def _analyze_network(
        self,
        environment: EnvironmentSnapshot,
        network_events: List[NetworkEvent],
        redaction: Tuple[int, int],
    ) -> List[SurfaceFinding]:
        findings: List[SurfaceFinding] = []
        for event in network_events:
            payloads: List[Tuple[str, Optional[str]]] = []
            if event.direction == "request":
                for header, value in event.headers.items():
                    payloads.append((f"header:{header}", value))
                payloads.append(("body", event.body_preview))
            else:
                for header, value in event.headers.items():
                    payloads.append((f"header:{header}", value))

            for source, value in payloads:
                if not value:
                    continue
                matches = SecretDetector.scan_text(value, redaction)
                if not matches:
                    continue
                match = matches[0]
                if match.confidence < self.config.pattern_confidence_floor:
                    continue
                severity = Severity.CRITICAL if "authorization" in source or event.direction == "request" else match.severity
                findings.append(
                    SurfaceFinding(
                        id=f"surface_network_{hash((event.url, source)) & 0xFFFF}",
                        kind="NetworkExposure",
                        severity=severity,
                        confidence=match.confidence,
                        title="Sensitive value observed in network activity",
                        description=f"{match.pattern_description} detected in {event.direction} {source}.",
                        recommendation="Avoid embedding bearer tokens or API keys in client-visible requests.",
                        evidence={
                            "location": {
                                "url": event.url,
                                "method": event.method,
                                "direction": event.direction,
                                "route": environment.route,
                                "source": source,
                                "status": event.status,
                            },
                            "valuePreview": match.redacted,
                            "pattern": match.pattern_id,
                        },
                    )
                )
        return findings

    def _analyze_url(
        self,
        environment: EnvironmentSnapshot,
        network_events: List[NetworkEvent],
        redaction: Tuple[int, int],
    ) -> List[SurfaceFinding]:
        findings: List[SurfaceFinding] = []
        url = environment.location.get("href") or ""
        search = environment.location.get("search") or ""
        hash_fragment = environment.location.get("hash") or ""

        url_components = {
            "full": url,
            "search": search,
            "hash": hash_fragment,
        }
        for component_name, value in url_components.items():
            if not value:
                continue
            matches = SecretDetector.scan_text(value, redaction)
            if not matches:
                continue
            match = matches[0]
            if match.confidence < self.config.pattern_confidence_floor:
                continue
            findings.append(
                SurfaceFinding(
                    id=f"surface_url_{component_name}",
                    kind="UrlExposure",
                    severity=Severity.CRITICAL,
                    confidence=match.confidence,
                    title="Sensitive token present in URL",
                    description=f"{match.pattern_description} detected in URL {component_name}.",
                    recommendation="Avoid placing tokens in URLs. Use POST bodies or HttpOnly cookies.",
                    evidence={
                        "location": {
                            "component": component_name,
                            "route": environment.route,
                        },
                        "valuePreview": match.redacted,
                        "pattern": match.pattern_id,
                    },
                )
            )
        if search and not findings:
            policy = self._infer_referrer_policy(environment, network_events)
            if policy in (None, "unsafe-url", "no-referrer-when-downgrade"):
                findings.append(
                    SurfaceFinding(
                        id="surface_referrer_policy",
                        kind="ReferrerPolicy",
                        severity=Severity.HIGH,
                        confidence=0.6,
                        title="Potential referrer leakage risk",
                        description="Sensitive parameters might leak via permissive Referrer-Policy.",
                        recommendation="Set Referrer-Policy to `strict-origin` or stricter when sensitive parameters appear in URLs.",
                        evidence={
                            "location": {"route": environment.route},
                            "policy": policy,
                        },
                    )
                )
        return findings

    def _infer_referrer_policy(
        self,
        environment: EnvironmentSnapshot,
        network_events: List[NetworkEvent],
    ) -> Optional[str]:
        for event in network_events:
            if event.direction != "response":
                continue
            policy = event.headers.get("referrer-policy")
            if policy:
                return str(policy).lower()

        policy = environment.location.get("referrerPolicy")
        if policy:
            return str(policy).lower()
        return None

    def _analyze_globals(self, environment: EnvironmentSnapshot) -> List[SurfaceFinding]:
        if not environment.new_globals:
            return []
        return [
            SurfaceFinding(
                id="surface_new_globals",
                kind="GlobalScope",
                severity=Severity.MEDIUM,
                confidence=0.5,
                title="New global variables introduced",
                description="Application introduced additional globals after boot; inspect for exposed state.",
                recommendation="Namespace client state and avoid leaking secrets on window.*.",
                evidence={
                    "location": {"route": environment.route},
                    "globals": environment.new_globals[:50],
                },
            )
        ]


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Browser surface agent for detecting frontend secret exposure."
    )
    parser.add_argument("url", help="Target URL to open within the browser context.")
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run the browser in headful mode (default: headless).",
    )
    parser.add_argument(
        "--wait-until",
        default="networkidle",
        choices=["load", "domcontentloaded", "networkidle", "commit"],
        help="Wait condition passed to Playwright page.goto.",
    )
    parser.add_argument(
        "--post-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after initial navigation before analyzing storage.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )

    args = parser.parse_args()
    config = SurfaceAgentConfig(
        headless=not args.headful,
        wait_until=args.wait_until,
        post_navigation_wait=args.post_wait,
    )
    agent = SurfaceAgent(config)
    result = agent.scan(args.url)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"[target] {result.target}")
    print(f"[resolved] {result.resolved_url}")
    print(f"[findings] {len(result.findings)}")
    print(f"[summary] {result.summary_text} (source: {result.summary_source})")
    for finding in result.findings:
        location = finding.evidence.get("location")
        location_suffix = f" at {location}" if location else ""
        print(
            f"- ({finding.severity.value.upper()}) {finding.title}"
            f"{location_suffix}: {finding.description}"
        )
    if not result.findings:
        print("No frontend secret exposure detected.")


if __name__ == "__main__":  # pragma: no cover
    run_cli()


