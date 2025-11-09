"""Injection/XSS detection agent for Heimdall.

This module couples HTTP inspection with Playwright-driven browser automation
to uncover reflected, stored, and DOM-based cross-site scripting issues. It is
designed to remain standalone, mirroring the cleanliness of
`agents.network.network_agent` while focusing on injection-specific logic.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple
import time
from html import escape as html_escape
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx  # type: ignore[import]
from bs4 import BeautifulSoup  # type: ignore[import]
if TYPE_CHECKING:  # pragma: no cover
    from playwright.sync_api import Page  # type: ignore[import]

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
class InjectionFinding:
    """Represents an injection-related security issue surfaced by the agent."""

    id: str
    severity: Severity
    title: str
    description: str
    recommendation: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


@dataclass
class HTTPRequestRecord:
    """Captured HTTP request/response metadata for later analysis."""

    method: str
    url: str
    status_code: Optional[int]
    headers: Dict[str, str]
    elapsed_ms: Optional[float]
    body: Optional[str] = None
    error: Optional[str] = None

    def simplified_headers(self) -> Dict[str, str]:
        return {k.lower(): v for k, v in self.headers.items()}


@dataclass
class FormDescriptor:
    """Simple representation of a form discovered in static HTML."""

    action: str
    method: str
    inputs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DOMEventRecord:
    """Represents a noteworthy DOM mutation or console event during scanning."""

    timestamp: float
    event_type: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InjectionAttempt:
    """Tracks a single payload attempt against a particular sink/source pair."""

    payload: str
    context: str
    target_url: str
    method: str
    input_name: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InjectionAgentConfig:
    """Runtime configuration for the injection agent."""

    timeout: float = 12.0
    verify_ssl: bool = True
    follow_redirects: bool = True
    headless: bool = True
    user_agent: str = "Heimdall-Injection-Agent/0.1"
    methods: Tuple[str, ...] = ("GET", "POST")
    extra_paths: Tuple[str, ...] = ()
    payloads: Tuple[str, ...] = (
        "__HEIMDALL_REFLECT_JOBID__<svg/onload=alert(1)>",
        "\"'><svg/onload=alert(1)>",
        "<img src=x onerror=alert('heimdall')>",
    )
    max_forms: int = 25
    mutation_observer_timeout: float = 6.0


@dataclass
class InjectionScanResult:
    """Top-level payload returned by the injection agent."""

    target: str
    resolved_url: str
    http_records: List[HTTPRequestRecord]
    dom_events: List[DOMEventRecord]
    attempts: List[InjectionAttempt]
    findings: List[InjectionFinding]
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
            "http_response_count": len(self.http_records),
            "dom_event_count": len(self.dom_events),
            "attempt_count": len(self.attempts),
            "severity_counts": counts,
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
            "target": self.target,
            "resolved_url": self.resolved_url,
            "http_records": [asdict(record) for record in self.http_records],
            "dom_events": [asdict(event) for event in self.dom_events],
            "attempts": [asdict(attempt) for attempt in self.attempts],
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
    def generate(cls, scan: InjectionScanResult) -> Tuple[str, str]:
        llm_summary = cls._try_xai_summary(scan)
        if llm_summary:
            return llm_summary, "xai"
        return cls._heuristic_summary(scan), "heuristic"

    @classmethod
    def _try_xai_summary(cls, scan: InjectionScanResult) -> Optional[str]:
        # Import locally to avoid mandatory dependency when not configured.
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None

        import json
        import os

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            return None
        if not scan.findings:
            return None

        model = os.environ.get("XAI_SUMMARY_MODEL", cls._DEFAULT_MODEL)
        base_url = os.environ.get("XAI_API_BASE", "https://api.x.ai/v1")
        timeout = float(os.environ.get("XAI_SUMMARY_TIMEOUT", cls._DEFAULT_TIMEOUT))

        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            return None

        payload = cls._build_payload(scan)
        system_prompt = (
            "You are a security engineer summarizing web injection findings. "
            "Highlight the highest severity, note affected pages, and call out "
            "whether findings were confirmed via DOM execution."
        )
        user_prompt = (
            "Summarize these injection scan results for a technical stakeholder:\n"
            f"{json.dumps(payload, indent=2)}"
        )

        try:
            completion = client.chat.completions.create(
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
    def _build_payload(scan: InjectionScanResult) -> Dict[str, Any]:
        return {
            "target": scan.target,
            "resolved_url": scan.resolved_url,
            "findings": [
                {
                    "id": finding.id,
                    "title": finding.title,
                    "severity": finding.severity.value,
                    "description": finding.description,
                    "recommendation": finding.recommendation,
                }
                for finding in scan.findings
            ],
            "attempts": [
                {
                    "payload": attempt.payload,
                    "context": attempt.context,
                    "target_url": attempt.target_url,
                    "method": attempt.method,
                }
                for attempt in scan.attempts[:10]
            ],
        }

    @staticmethod
    def _heuristic_summary(scan: InjectionScanResult) -> str:
        if not scan.findings:
            friendly_target = scan.resolved_url or scan.target
            return f"No injection vectors detected for {friendly_target}."

        top_finding = min(
            scan.findings,
            key=lambda finding: (_SEVERITY_RANK.get(finding.severity, 99), finding.id),
        )
        return (
            f"Detected {len(scan.findings)} injection issue(s). "
            f"Highest severity {top_finding.severity.value.upper()} - "
            f"{top_finding.title}: {top_finding.description}"
        )


class InjectionAgent:
    """Facade orchestrating HTTP collection and browser-based injection checks."""

    def __init__(self, config: Optional[InjectionAgentConfig] = None):
        self.config = config or InjectionAgentConfig()
        self._http_fetcher = HTTPFetcher(self.config)

    def scan(
        self, url: str, extra_paths: Optional[Iterable[str]] = None
    ) -> InjectionScanResult:
        normalized = self._normalize_url(url)
        paths = list(self.config.extra_paths)
        if extra_paths:
            paths.extend(extra_paths)

        http_records = self._http_fetcher.collect(normalized, paths)
        static_findings, forms = StaticAnalyzer.analyze_records(http_records)

        dom_events, dynamic_attempts, dynamic_findings = self._run_playwright(
            normalized, http_records, forms
        )

        resolved_url = http_records[0].url if http_records else normalized
        findings = static_findings + dynamic_findings
        attempts = dynamic_attempts
        return InjectionScanResult(
            target=url,
            resolved_url=resolved_url,
            http_records=http_records,
            dom_events=dom_events,
            attempts=attempts,
            findings=findings,
        )

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _run_playwright(
        self,
        base_url: str,
        http_records: Sequence[HTTPRequestRecord],
        forms: Sequence[FormDescriptor],
    ) -> Tuple[List[DOMEventRecord], List[InjectionAttempt], List[InjectionFinding]]:
        try:
            scanner = PlaywrightScanner(self.config)
        except ImportError as exc:
            finding = InjectionFinding(
                id="injection_playwright_missing",
                severity=Severity.INFO,
                title="Playwright dependency missing",
                description="Playwright is not installed; dynamic injection checks were skipped.",
                recommendation="Install Playwright (pip install playwright) and run `playwright install`.",
                evidence={"error": str(exc)},
            )
            return [], [], [finding]

        return scanner.scan(base_url, http_records, forms)


class HTTPFetcher:
    """Collects HTTP responses for initial analysis."""

    def __init__(self, config: InjectionAgentConfig):
        self.config = config

    def collect(self, base_url: str, paths: Sequence[str]) -> List[HTTPRequestRecord]:
        records: List[HTTPRequestRecord] = []
        transport = httpx.HTTPTransport(retries=1)
        headers = {"User-Agent": self.config.user_agent}
        methods = tuple(dict.fromkeys(method.upper() for method in self.config.methods))

        with httpx.Client(
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            follow_redirects=self.config.follow_redirects,
            transport=transport,
        ) as client:
            for target_url in self._enumerate_targets(base_url, paths):
                for method in methods:
                    record = self._send_request(client, method, target_url, headers)
                    records.append(record)
        return records

    def _enumerate_targets(self, base_url: str, paths: Sequence[str]) -> List[str]:
        targets = [base_url]
        for path in paths:
            targets.append(urljoin(base_url, path))
        return targets

    def _send_request(
        self, client: httpx.Client, method: str, url: str, headers: Dict[str, str]
    ) -> HTTPRequestRecord:
        start = time.perf_counter()
        try:
            response = client.request(method, url, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000
            body: Optional[str]
            try:
                body = response.text
            except UnicodeDecodeError:
                body = None
            return HTTPRequestRecord(
                method=method,
                url=str(response.url),
                status_code=response.status_code,
                headers=dict(response.headers),
                elapsed_ms=round(elapsed, 2),
                body=body,
            )
        except httpx.HTTPError as exc:
            return HTTPRequestRecord(
                method=method,
                url=url,
                status_code=None,
                headers={},
                elapsed_ms=None,
                body=None,
                error=str(exc),
            )


class StaticAnalyzer:
    """Performs static HTML analysis on collected HTTP responses."""

    @classmethod
    def analyze_records(
        cls, records: Sequence[HTTPRequestRecord]
    ) -> Tuple[List[InjectionFinding], List[FormDescriptor]]:
        findings: List[InjectionFinding] = []
        forms: List[FormDescriptor] = []
        for record in records:
            if not cls._is_html_candidate(record):
                continue
            if record.body:
                document_forms, inline_evidence = cls._extract_forms(record)
                forms.extend(document_forms)
                findings.extend(cls._inline_handler_findings(record, inline_evidence))
            findings.extend(cls._header_findings(record))
        return findings, forms

    @staticmethod
    def _is_html_candidate(record: HTTPRequestRecord) -> bool:
        if not record.status_code or record.status_code >= 400:
            return False
        if not record.body:
            return False
        content_type = record.simplified_headers().get("content-type", "")
        return "html" in content_type or "<html" in record.body.lower()

    @classmethod
    def _extract_forms(
        cls, record: HTTPRequestRecord
    ) -> Tuple[List[FormDescriptor], Dict[str, Any]]:
        soup = BeautifulSoup(record.body, "html.parser")
        forms: List[FormDescriptor] = []
        inline_events: List[Dict[str, Any]] = []
        inline_scripts: List[str] = []

        for form in soup.find_all("form"):
            action = form.get("action") or record.url
            method = (form.get("method") or "GET").upper()
            inputs: List[Dict[str, Any]] = []
            for field in form.find_all(["input", "textarea", "select"]):
                inputs.append(
                    {
                        "name": field.get("name"),
                        "type": field.get("type"),
                        "placeholder": field.get("placeholder"),
                    }
                )
                inline_events.extend(cls._collect_inline_handlers(field))
            inline_events.extend(cls._collect_inline_handlers(form))
            forms.append(FormDescriptor(action=action, method=method, inputs=inputs))

        for tag in soup.find_all(True):
            inline_events.extend(cls._collect_inline_handlers(tag))
            if tag.name == "script" and not tag.get("src"):
                script_content = tag.get_text(strip=True)
                if script_content:
                    inline_scripts.append(script_content[:200])

        evidence = {"inline_events": inline_events, "inline_scripts": inline_scripts}
        return forms, evidence

    @staticmethod
    def _collect_inline_handlers(element: Any) -> List[Dict[str, Any]]:
        handlers: List[Dict[str, Any]] = []
        for attr, value in element.attrs.items():
            if attr.lower().startswith("on") and isinstance(value, str):
                handlers.append(
                    {
                        "attribute": attr,
                        "value": value[:200],
                        "element": element.name,
                    }
                )
        return handlers

    @classmethod
    def _inline_handler_findings(
        cls, record: HTTPRequestRecord, evidence: Dict[str, Any]
    ) -> List[InjectionFinding]:
        findings: List[InjectionFinding] = []
        inline_events: List[Dict[str, Any]] = evidence.get("inline_events", [])
        inline_scripts: List[str] = evidence.get("inline_scripts", [])

        if inline_events:
            findings.append(
                InjectionFinding(
                    id="injection_inline_event_handlers",
                    severity=Severity.MEDIUM,
                    title="Inline event handlers detected",
                    description="Inline `on*` handlers can enable DOM XSS when paired with untrusted data.",
                    recommendation="Refactor inline handlers to addEventListener and ensure proper encoding of dynamic inputs.",
                    evidence={"url": record.url, "inline_handlers": inline_events[:20]},
                )
            )

        if inline_scripts:
            findings.append(
                InjectionFinding(
                    id="injection_inline_scripts",
                    severity=Severity.LOW,
                    title="Inline script blocks detected",
                    description="Inline `<script>` blocks make CSP harder to lock down and may mix data with script contexts.",
                    recommendation="Move scripts to external files and enforce strict CSP without `unsafe-inline`.",
                    evidence={"url": record.url, "sample_scripts": inline_scripts[:5]},
                )
            )
        return findings

    @staticmethod
    def _header_findings(record: HTTPRequestRecord) -> List[InjectionFinding]:
        headers = record.simplified_headers()
        findings: List[InjectionFinding] = []
        csp = headers.get("content-security-policy")

        if not csp:
            findings.append(
                InjectionFinding(
                    id="injection_missing_csp",
                    severity=Severity.MEDIUM,
                    title="Missing Content-Security-Policy header",
                    description="Responses lack a Content-Security-Policy header, increasing XSS impact.",
                    recommendation="Deploy a strict CSP (default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none').",
                    evidence={"url": record.url},
                )
            )

        set_cookie = headers.get("set-cookie")
        if set_cookie and "httponly" not in set_cookie.lower():
            findings.append(
                InjectionFinding(
                    id="injection_cookie_missing_httponly",
                    severity=Severity.MEDIUM,
                    title="Session cookie missing HttpOnly",
                    description="Cookies without HttpOnly are exposed to XSS, increasing account takeover risk.",
                    recommendation="Set cookies with HttpOnly; Secure; SameSite=strict to limit XSS impact.",
                    evidence={"url": record.url, "set_cookie": set_cookie},
                )
            )
        return findings


class PlaywrightScanner:
    """Runs Playwright-driven injection attempts and collects findings."""

    def __init__(self, config: InjectionAgentConfig):
        # Delay heavy import until Playwright functionality is requested.
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise exc
        self._sync_playwright = sync_playwright
        self.config = config

    def scan(
        self,
        base_url: str,
        http_records: Sequence[HTTPRequestRecord],
        forms: Sequence[FormDescriptor],
    ) -> Tuple[List[DOMEventRecord], List[InjectionAttempt], List[InjectionFinding]]:
        from playwright.sync_api import (  # type: ignore
            Browser,
            ConsoleMessage,
            Dialog,
            Page,
            TimeoutError as PlaywrightTimeoutError,
        )

        dom_events: List[DOMEventRecord] = []
        attempts: List[InjectionAttempt] = []
        findings: List[InjectionFinding] = []

        targets = self._candidate_urls(base_url, http_records, forms)

        try:
            with self._sync_playwright() as p:
                browser: Browser = p.chromium.launch(headless=self.config.headless)
                context = browser.new_context(user_agent=self.config.user_agent)
                context.add_init_script(self._init_script())
                page: Page = context.new_page()
                self._attach_listeners(page, dom_events)

                for target in targets:
                    self._probe_query_payloads(
                        page, target, attempts, findings, dom_events
                    )
                    self._probe_form_payloads(
                        page, target, attempts, findings, dom_events
                    )

                browser.close()
        except PlaywrightTimeoutError as exc:
            findings.append(
                InjectionFinding(
                    id="injection_playwright_timeout",
                    severity=Severity.LOW,
                    title="Playwright navigation timed out",
                    description="Dynamic injection scan timed out waiting for a page load.",
                    recommendation="Increase the timeout or verify the target is reachable.",
                    evidence={"error": str(exc)},
                )
            )
        except Exception as exc:  # pragma: no cover - defensive catch
            findings.append(
                InjectionFinding(
                    id="injection_playwright_error",
                    severity=Severity.LOW,
                    title="Playwright scan error",
                    description="An unexpected error occurred during Playwright-based scanning.",
                    recommendation="Review the error and rerun with debug logging enabled.",
                    evidence={"error": repr(exc)},
                )
            )

        return dom_events, attempts, findings

    def _candidate_urls(
        self,
        base_url: str,
        http_records: Sequence[HTTPRequestRecord],
        forms: Sequence[FormDescriptor],
    ) -> List[str]:
        urls = {base_url}
        for record in http_records:
            if record.status_code and record.status_code < 400:
                urls.add(record.url)
        for form in forms:
            urls.add(form.action or base_url)
        return list(urls)

    def _attach_listeners(
        self, page: Page, dom_events: List[DOMEventRecord]
    ) -> None:
        from playwright.sync_api import ConsoleMessage, Dialog  # type: ignore

        def record_event(event_type: str, details: Dict[str, Any]) -> None:
            dom_events.append(
                DOMEventRecord(
                    timestamp=time.time(),
                    event_type=event_type,
                    details=details,
                )
            )

        def on_console(msg: ConsoleMessage) -> None:
            record_event(
                "console",
                {
                    "type": msg.type,
                    "text": msg.text,
                },
            )

        def on_page_error(exc: Exception) -> None:
            record_event("pageerror", {"error": repr(exc)})

        def on_dialog(dialog: Dialog) -> None:
            record_event(
                "dialog",
                {"type": dialog.type, "message": dialog.message},
            )
            try:
                dialog.dismiss()
            except Exception:
                dialog.accept()

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("dialog", on_dialog)

    def _probe_query_payloads(
        self,
        page: Page,
        url: str,
        attempts: List[InjectionAttempt],
        findings: List[InjectionFinding],
        dom_events: List[DOMEventRecord],
    ) -> None:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # type: ignore

        for payload in self.config.payloads:
            test_url = self._augment_url_with_payload(url, payload)
            attempts.append(
                InjectionAttempt(
                    payload=payload,
                    context="query-string",
                    target_url=test_url,
                    method="GET",
                )
            )
            try:
                page.goto(
                    test_url,
                    wait_until="load",
                    timeout=self.config.timeout * 1000,
                )
                page.wait_for_timeout(self.config.mutation_observer_timeout * 1000)
            except PlaywrightTimeoutError as exc:
                dom_events.append(
                    DOMEventRecord(
                        timestamp=time.time(),
                        event_type="timeout",
                        details={"url": test_url, "error": str(exc)},
                    )
                )
                continue

            detection = self._detect_reflection(page, payload)
            if detection:
                findings.append(self._build_reflection_finding(url, payload, detection))

    def _probe_form_payloads(
        self,
        page: Page,
        url: str,
        attempts: List[InjectionAttempt],
        findings: List[InjectionFinding],
        dom_events: List[DOMEventRecord],
    ) -> None:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # type: ignore

        for payload in self.config.payloads:
            try:
                page.goto(url, wait_until="load", timeout=self.config.timeout * 1000)
            except PlaywrightTimeoutError as exc:
                dom_events.append(
                    DOMEventRecord(
                        timestamp=time.time(),
                        event_type="timeout",
                        details={"url": url, "error": str(exc)},
                    )
                )
                return

            forms = page.query_selector_all("form")
            for idx, form in enumerate(forms[: self.config.max_forms]):
                attempt = InjectionAttempt(
                    payload=payload,
                    context="form-submit",
                    target_url=url,
                    method="POST",
                    input_name=None,
                )
                attempts.append(attempt)
                try:
                    self._fill_form_with_payload(form, payload)
                    with page.expect_navigation(
                        wait_until="load", timeout=self.config.timeout * 1000
                    ):
                        form.evaluate(
                            """(form) => {
                                if (form.requestSubmit) { form.requestSubmit(); }
                                else { form.submit(); }
                            }"""
                        )
                except PlaywrightTimeoutError:
                    dom_events.append(
                        DOMEventRecord(
                            timestamp=time.time(),
                            event_type="timeout",
                            details={"context": "form-submit", "url": url, "index": idx},
                        )
                    )
                except Exception as exc:
                    dom_events.append(
                        DOMEventRecord(
                            timestamp=time.time(),
                            event_type="form-error",
                            details={"error": repr(exc), "url": url, "index": idx},
                        )
                    )
                finally:
                    page.wait_for_timeout(self.config.mutation_observer_timeout * 1000)
                    detection = self._detect_reflection(page, payload)
                    if detection:
                        findings.append(
                            self._build_reflection_finding(url, payload, detection)
                        )
                        return

    def _fill_form_with_payload(self, form: Any, payload: str) -> None:
        fields = form.query_selector_all("input, textarea")
        for field in fields:
            input_type = (field.get_attribute("type") or "").lower()
            if input_type in {"password", "hidden", "submit", "button", "image"}:
                continue
            try:
                field.fill(payload)
            except Exception:
                field.evaluate("(el, value) => el.value = value", payload)

    @staticmethod
    def _init_script() -> str:
        return """
            window.__heimdallEvents = [];
            const record = (type, detail) => {
                try {
                    console.log(`[HEIMDALL:${type}]` + JSON.stringify(detail));
                } catch (_) {}
            };
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    record('mutation', {
                        type: mutation.type,
                        added: mutation.addedNodes.length,
                        removed: mutation.removedNodes.length
                    });
                });
            });
            observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
            window.alert = ((orig) => (...args) => {
                record('alert', { args });
                return orig(...args);
            })(window.alert);
        """

    def _detect_reflection(self, page: Page, payload: str) -> Optional[Dict[str, Any]]:
        content = page.content()
        if payload in content:
            return {"kind": "raw", "context": "html"}

        escaped = html_escape(payload)
        if escaped in content:
            return {"kind": "escaped", "context": "html"}

        return None

    def _build_reflection_finding(
        self, url: str, payload: str, detection: Dict[str, Any]
    ) -> InjectionFinding:
        severity = Severity.MEDIUM
        description = "Payload reflected but HTML-encoded."
        if detection["kind"] == "raw":
            severity = Severity.HIGH
            description = (
                "Unescaped payload reflected in HTML, indicating a likely reflected XSS vector."
            )
        return InjectionFinding(
            id="injection_reflected_xss",
            severity=severity,
            title="Reflected payload detected",
            description=description,
            recommendation="Ensure user input is contextually encoded and consider server-side sanitization.",
            evidence={"url": url, "payload": payload, "detection": detection},
        )

    def _augment_url_with_payload(self, url: str, payload: str) -> str:
        parsed = urlparse(url)
        query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query_params["__heimdall_marker"] = payload
        new_query = urlencode(query_params, doseq=True)
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            )
        )


def _finding_location(finding: InjectionFinding) -> Optional[str]:
    evidence = finding.evidence or {}
    url = evidence.get("url")
    context = evidence.get("detection", {}).get("context") if evidence else None
    if url and context:
        return f"{context} {url}"
    if url:
        return str(url)
    return None


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Injection/XSS detection agent.")
    parser.add_argument("url", help="Target URL or domain to probe.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Additional path (relative to the target) to probe. Can be repeated.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="HTTP/Playwright timeout in seconds (default: 12).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS verification (not recommended).",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Playwright in headed mode for debugging.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )

    args = parser.parse_args()
    config = InjectionAgentConfig(
        timeout=args.timeout,
        verify_ssl=not args.insecure,
        headless=not args.headful,
    )
    agent = InjectionAgent(config)
    result = agent.scan(args.url, extra_paths=args.path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"[target] {result.target}")
    print(f"[resolved] {result.resolved_url}")
    print(f"[http responses] {len(result.http_records)}")
    print(f"[dom events] {len(result.dom_events)}")
    print(f"[attempts] {len(result.attempts)}")
    print(f"[summary] {result.summary_text} (source: {result.summary_source})")
    for finding in result.findings:
        location = _finding_location(finding)
        location_suffix = f" at {location}" if location else ""
        print(
            f"- ({finding.severity.value.upper()}) {finding.title}"
            f"{location_suffix}: {finding.description}"
        )
    if not result.findings:
        print("No injection vectors detected.")


if __name__ == "__main__":
    run_cli()
