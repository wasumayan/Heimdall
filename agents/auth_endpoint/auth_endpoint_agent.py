"""Auth & Endpoint Agent (Agent C) for Heimdall.

The agent ingests DOM/network artifacts, discovers callable endpoints, and
probes them without credentials to highlight access-control gaps or sensitive
responses that are reachable anonymously.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urljoin, urlparse, urlsplit

import httpx

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:  # pragma: no cover - optional dependency
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        async_playwright,
    )
except ImportError as exc:  # pragma: no cover
    Browser = BrowserContext = Page = Playwright = None  # type: ignore
    async_playwright = None  # type: ignore
    _PLAYWRIGHT_IMPORT_ERROR = exc
else:  # pragma: no cover
    _PLAYWRIGHT_IMPORT_ERROR = None

from agents.network.network_agent import Finding, Severity

_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


@dataclass
class EndpointCandidate:
    """Endpoint discovered from DOM/network artifacts."""

    method: str
    url: str
    sources: List[str]
    requires_auth: bool = False
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str]:
        return (self.method.upper(), self.url)

    def add_source(self, source: str) -> None:
        if source not in self.sources:
            self.sources.append(source)

    def merge_metadata(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if key not in self.metadata:
                self.metadata[key] = value
                continue
            if isinstance(self.metadata[key], dict) and isinstance(value, dict):
                self.metadata[key].update(value)
            elif isinstance(self.metadata[key], list) and isinstance(value, list):
                merged = list(self.metadata[key])
                for item in value:
                    if item not in merged:
                        merged.append(item)
                self.metadata[key] = merged
            else:
                self.metadata[key] = value

    def add_tags(self, tags: Iterable[str]) -> None:
        existing = set(self.tags)
        for tag in tags:
            existing.add(tag)
        self.tags = tuple(sorted(existing))


@dataclass
class EndpointProbeResult:
    """HTTP response metadata captured during probing."""

    candidate: EndpointCandidate
    status_code: Optional[int]
    reason: Optional[str]
    headers: Dict[str, str]
    body_preview: Optional[str]
    elapsed_ms: Optional[float]
    error: Optional[str] = None
    sensitivity_score: int = 0
    sensitivity_signals: List[str] = field(default_factory=list)
    sensitivity_confidence: float = 0.0
    sensitivity_tags: List[str] = field(default_factory=list)
    risk_level: str = "info"
    recommendation: Optional[str] = None
    probed_at: Optional[str] = None
    risk_severity: Severity = Severity.INFO

    def is_success(self) -> bool:
        return bool(self.status_code) and 200 <= self.status_code < 300

    def to_evidence(self) -> Dict[str, Any]:
        return {
            "url": self.candidate.url,
            "method": self.candidate.method,
            "status_code": self.status_code,
            "sample": self.body_preview,
            "sensitivity_tags": self.sensitivity_tags,
            "sensitivity_confidence": self.sensitivity_confidence,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "discovered_from": list(self.candidate.sources),
            "probed_at": self.probed_at,
        }


@dataclass
class SensitivityDetectionResult:
    score: int
    signals: List[str]
    confidence: float
    tags: List[str]


@dataclass
class AgentCConfig:
    """Runtime configuration for the Auth/Endpoint agent."""

    timeout: float = 8.0
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_endpoints: int = 30
    max_body_preview: int = 4096
    user_agent: str = "Heimdall-AuthEndpoint-Agent/0.1"
    safe_endpoints_whitelist: Tuple[str, ...] = ()
    rendered_dom_path: Optional[str] = None
    network_log_path: Optional[str] = None
    allow_external_hosts: bool = False
    concurrency: int = 3


@dataclass
class AuthEndpointScanResult:
    """Structured payload returned by Agent C."""

    target: str
    base_url: str
    endpoints: List[EndpointCandidate]
    probes: List[EndpointProbeResult]
    findings: List[Finding]
    generated_at: float = field(default_factory=time.time)

    @property
    def summary(self) -> Dict[str, Any]:
        counts = {sev.value: 0 for sev in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return {
            "target": self.target,
            "base_url": self.base_url,
            "discovered_endpoints": len(self.endpoints),
            "probed_endpoints": len(self.probes),
            "severity_counts": counts,
        }

    @property
    def summary_text(self) -> str:
        if not self.findings:
            return "No unauthenticated sensitive endpoints detected."
        top = min(
            self.findings,
            key=lambda finding: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(
                    finding.severity.value, 99
                ),
                finding.id,
            ),
        )
        return (
            f"Identified {len(self.findings)} endpoint issue(s); "
            f"highest severity {top.severity.value.upper()} on {top.evidence.get('url')}."
        )

    @property
    def evidence(self) -> List[Dict[str, Any]]:
        return [probe.to_evidence() for probe in self.probes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "base_url": self.base_url,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "summary_text": self.summary_text,
            "endpoints": [
                {
                    "method": endpoint.method,
                    "url": endpoint.url,
                    "requires_auth": endpoint.requires_auth,
                    "sources": endpoint.sources,
                    "tags": endpoint.tags,
                }
                for endpoint in self.endpoints
            ],
            "probes": self.evidence,
            "evidence": self.evidence,
            "findings": [finding.to_dict() for finding in self.findings],
        }


_SENSITIVE_PATH_HINTS = (
    "admin",
    "internal",
    "export",
    "backup",
    "download",
    "billing",
    "invoice",
    "token",
    "secret",
    "report",
    "debug",
)

_PAID_PATH_HINTS = ("premium", "pro", "enterprise", "plan", "tier")

_QUERY_FLAG_HINTS = ("plan", "tier", "role", "access", "entitlement")


def _infer_tags(url: str, metadata: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    path = urlparse(url).path.lower()
    for hint in _SENSITIVE_PATH_HINTS:
        if hint in path:
            tags.append("sensitive-path")
            break
    for hint in _PAID_PATH_HINTS:
        if hint in path:
            tags.append("paid-path")
            break

    query = metadata.get("query_params") or {}
    for key, value in query.items():
        lowered_key = str(key).lower()
        lowered_val = str(value).lower()
        if lowered_key in _QUERY_FLAG_HINTS:
            tags.append("access-flag")
        if any(hint in lowered_val for hint in _PAID_PATH_HINTS):
            tags.append("paid-flag")
    return tags


class _FormCollector(HTMLParser):
    """Extracts form/action metadata without external dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.forms: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "form":
            return
        attr_map = {key.lower(): value or "" for key, value in attrs}
        self.forms.append(attr_map)


class EndpointDiscovery:
    """Builds the endpoint candidate list from DOM and network artifacts."""

    _FETCH_RE = re.compile(
        r"""(?:fetch|axios\.(?:get|post|put|delete|patch)|axios)\s*\(\s*['"]([^'"`]+)['"]""",
        re.IGNORECASE,
    )
    _XHR_RE = re.compile(
        r"""(?:open|request)\s*\(\s*['"]([A-Z]+)['"]\s*,\s*['"]([^'"]+)['"]""",
        re.IGNORECASE,
    )

    def __init__(
        self,
        base_url: str,
        rendered_dom: Optional[str],
        network_log: Optional[Sequence[Dict[str, Any]]],
        max_endpoints: int,
    ) -> None:
        self.base_url = base_url
        self.rendered_dom = rendered_dom or ""
        self.network_log = network_log or []
        self.max_endpoints = max_endpoints
        self._candidates: Dict[Tuple[str, str], EndpointCandidate] = {}

    def discover(self) -> List[EndpointCandidate]:
        self._from_network_log()
        self._from_forms()
        self._from_scripts()
        return list(self._candidates.values())[: self.max_endpoints]

    def _from_network_log(self) -> None:
        for entry in self.network_log:
            resource_type = (entry.get("resourceType") or entry.get("type") or "").lower()
            if resource_type and resource_type not in {"xhr", "fetch"}:
                continue
            url = (
                entry.get("url")
                or entry.get("request", {}).get("url")
                or entry.get("documentURL")
            )
            if not url:
                continue
            method = (
                entry.get("method")
                or entry.get("request", {}).get("method")
                or "GET"
            )
            headers = entry.get("request", {}).get("headers") or entry.get("headers") or {}
            requires_auth = any(
                key.lower() == "authorization" for key in headers.keys()
            )
            metadata = {
                "headers": self._sanitize_headers(headers),
                "query_params": dict(parse_qsl(urlsplit(url).query)),
            }
            self._add_candidate(
                method=method,
                url=url,
                source="network_log",
                requires_auth=requires_auth,
                metadata=metadata,
            )

    def _from_forms(self) -> None:
        if not self.rendered_dom:
            return
        parser = _FormCollector()
        try:
            parser.feed(self.rendered_dom)
        except Exception:
            return

        for form in parser.forms:
            action = form.get("action")
            if not action:
                continue
            method = form.get("method", "GET")
            requires_auth = form.get("data-requires-auth") == "true"
            metadata = {"form_attributes": form}
            self._add_candidate(
                method=method,
                url=action,
                source="dom_form",
                requires_auth=requires_auth,
                metadata=metadata,
            )

    def _from_scripts(self) -> None:
        if not self.rendered_dom:
            return
        for match in self._FETCH_RE.finditer(self.rendered_dom):
            self._add_candidate("GET", match.group(1), "js_fetch", metadata={})
        for match in self._XHR_RE.finditer(self.rendered_dom):
            method = match.group(1)
            url = match.group(2)
            self._add_candidate(method, url, "js_xhr", metadata={})

    def _add_candidate(
        self,
        method: str,
        url: str,
        source: str,
        requires_auth: bool,
        metadata: Dict[str, Any],
    ) -> None:
        normalized_url = self._normalize_url(url)
        method = (method or "GET").upper()
        key = (method, normalized_url)
        tags = _infer_tags(normalized_url, metadata)

        if key in self._candidates:
            candidate = self._candidates[key]
            candidate.add_source(source)
            candidate.requires_auth = candidate.requires_auth or requires_auth
            candidate.merge_metadata(metadata)
            candidate.add_tags(tags)
            return

        candidate = EndpointCandidate(
            method=method,
            url=normalized_url,
            sources=[source],
            requires_auth=requires_auth,
            tags=tuple(sorted(tags)),
            metadata=metadata,
        )
        self._candidates[key] = candidate

    def _normalize_url(self, url: str) -> str:
        if not url:
            return self.base_url
        parsed = urlparse(url)
        if parsed.scheme:
            return url
        return urljoin(self.base_url, url)

    @staticmethod
    def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        sanitized = {}
        for key, value in headers.items():
            lowered = key.lower()
            if lowered in {"authorization", "cookie"}:
                continue
            sanitized[key] = value
        return sanitized


class LLMSnippetClassifier:
    """Optional LLM-based classification for ambiguous snippets."""

    _DEFAULT_MODEL = "grok-2-latest"
    _TIMEOUT = 20

    @classmethod
    def classify(cls, snippet: str) -> Optional[Dict[str, Any]]:
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key or not snippet or OpenAI is None:
            return None
        model = os.environ.get("XAI_CLASSIFIER_MODEL", cls._DEFAULT_MODEL)
        base_url = os.environ.get("XAI_API_BASE", "https://api.x.ai/v1")
        system_prompt = (
            "You are a precise vulnerability classifier. Return only JSON with keys "
            'sensitive (bool), tags (array of strings), confidence (0-1 float), explanation (string).'
        )
        user_prompt = (
            "Determine whether the following HTTP response snippet contains sensitive data "
            "(PII, tokens, secrets, paid content). Respond strictly in JSON.\n"
            f"Snippet:\n{snippet[:2000]}"
        )
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=model,
                timeout=cls._TIMEOUT,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None
        if not completion.choices:
            return None
        content = completion.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None


class SensitiveContentDetector:
    """Scores response bodies for sensitive data exposure."""

    _PII_REGEXES = {
        "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}", re.IGNORECASE),
        "ssn": re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b"),
        "cc": re.compile(r"\\b(?:\\d[ -]*){13,16}\\b"),
    }
    _TOKEN_REGEXES = {
        "jwt": re.compile(r"eyJ[a-zA-Z0-9_-]+\\.[a-zA-Z0-9_-]+\\.[a-zA-Z0-9_-]+"),
        "api_key": re.compile(r"(?i)(api[_-]?key|secret|token)\\s*[:=]\\s*['\\\"]?[A-Za-z0-9-_]{16,}"),
    }
    _KEYWORDS = (
        "confidential",
        "internal use",
        "password",
        "bearer ",
        "auth_token",
        "sessionid",
        "refresh_token",
        "plan_pro",
        "admin_email",
    )

    @classmethod
    def analyze(cls, text: Optional[str]) -> SensitivityDetectionResult:
        if not text:
            return SensitivityDetectionResult(score=0, signals=[], confidence=0.0, tags=[])
        snippet = text[:2000]
        signals: List[str] = []
        score = 0

        for label, pattern in cls._PII_REGEXES.items():
            if pattern.search(snippet):
                signals.append(f"pii:{label}")
                score += 3
        for label, pattern in cls._TOKEN_REGEXES.items():
            if pattern.search(snippet):
                signals.append(f"token:{label}")
                score += 4
        lower_snippet = snippet.lower()
        for keyword in cls._KEYWORDS:
            if keyword in lower_snippet:
                signals.append(f"keyword:{keyword}")
                score += 1

        confidence = min(1.0, 0.2 * len(signals))
        if score < 4:
            llm_result = LLMSnippetClassifier.classify(snippet)
            if llm_result and llm_result.get("sensitive"):
                llm_tags = llm_result.get("tags") or []
                for tag in llm_tags:
                    signals.append(f"llm:{tag}")
                confidence = max(confidence, float(llm_result.get("confidence", 0.0)))
                if confidence >= 0.7:
                    score = max(score, 6)
                else:
                    score = max(score, 4)

        score = min(score, 10)
        signals = sorted(set(signals))
        tags = cls._signals_to_tags(signals)
        return SensitivityDetectionResult(score=score, signals=signals, confidence=confidence, tags=tags)

    @staticmethod
    def _signals_to_tags(signals: Sequence[str]) -> List[str]:
        tags: List[str] = []
        for signal in signals:
            if ":" in signal:
                tags.append(signal.split(":", 1)[1])
            else:
                tags.append(signal)
        return sorted(set(tags))


class RiskAssessor:
    """Assigns human-readable risk levels and recommendations."""

    _BUCKETS: Tuple[Tuple[int, Severity, str], ...] = (
        (9, Severity.CRITICAL, "critical"),
        (7, Severity.HIGH, "high"),
        (5, Severity.MEDIUM, "medium"),
        (3, Severity.LOW, "low"),
        (0, Severity.INFO, "info"),
    )

    @classmethod
    def annotate(cls, probe: EndpointProbeResult) -> Severity:
        score = cls._compute_score(probe)
        severity, risk = cls._bucket(score)
        probe.risk_level = risk
        probe.risk_severity = severity
        if not probe.recommendation:
            probe.recommendation = cls._default_recommendation(probe, severity)
        return severity

    @staticmethod
    def _compute_score(probe: EndpointProbeResult) -> int:
        score = probe.sensitivity_score
        if probe.is_success() and probe.status_code == 200:
            if probe.sensitivity_confidence >= 0.7:
                score += 1
            if "paid-path" in probe.candidate.tags:
                score += 1
            if probe.candidate.requires_auth:
                score = max(score, 5)
        if not probe.is_success():
            score = max(score - 2, 0)
        return min(score, 10)

    @classmethod
    def _bucket(cls, score: int) -> Tuple[Severity, str]:
        for threshold, severity, label in cls._BUCKETS:
            if score >= threshold:
                return severity, label
        return Severity.INFO, "info"

    @staticmethod
    def _default_recommendation(probe: EndpointProbeResult, severity: Severity) -> str:
        actions: List[str] = []
        if probe.candidate.requires_auth or "paid-path" in probe.candidate.tags:
            actions.append("Require authentication and return 401/403 for anonymous requests")
        if probe.sensitivity_tags:
            actions.append("Redact sensitive fields (PII/tokens) from public responses")
        if not actions:
            actions.append("Validate that this endpoint is intended for public access")
        if severity in (Severity.CRITICAL, Severity.HIGH):
            actions.append("Log and monitor anonymous access attempts for abuse patterns")
        return "; ".join(actions)


class EndpointAnalyzer:
    """Turns probe responses into structured findings."""

    @classmethod
    def generate_findings(cls, probes: Sequence[EndpointProbeResult]) -> List[Finding]:
        findings: List[Finding] = []
        for probe in probes:
            if probe.error:
                continue
            severity_hint = probe.risk_severity
            sensitive = cls._sensitive_exposure(probe, severity_hint)
            if sensitive:
                findings.append(sensitive)
                continue

            bypass = cls._auth_bypass(probe)
            if bypass:
                findings.append(bypass)
        return findings

    @staticmethod
    def _sensitive_exposure(
        probe: EndpointProbeResult, severity_hint: Optional[Severity]
    ) -> Optional[Finding]:
        score = probe.sensitivity_score
        if score < 4:
            return None

        if score >= 8:
            severity = Severity.CRITICAL
        elif score >= 6:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM
        if severity_hint and _SEVERITY_RANK[severity_hint] < _SEVERITY_RANK[severity]:
            severity = severity_hint

        description = "Endpoint response included indicators of sensitive data."
        recommendation = (
            "Require authentication and scrub sensitive fields from unauthenticated responses."
        )
        evidence = {
            "url": probe.candidate.url,
            "method": probe.candidate.method,
            "status": probe.status_code,
            "signals": probe.sensitivity_signals,
            "body_preview": probe.body_preview[:300] if probe.body_preview else None,
        }
        return Finding(
            id="auth_endpoint_sensitive_response",
            severity=severity,
            title="Sensitive data exposed without authentication",
            description=description,
            recommendation=recommendation,
            evidence=evidence,
        )

    @staticmethod
    def _auth_bypass(probe: EndpointProbeResult) -> Optional[Finding]:
        candidate = probe.candidate
        if not candidate.requires_auth and "paid-path" not in candidate.tags:
            return None
        if not probe.is_success():
            return None

        severity = Severity.HIGH if candidate.requires_auth else Severity.MEDIUM
        description = (
            "Endpoint that was previously accessed with credentials also responds successfully without auth."
        )
        if "paid-path" in candidate.tags:
            description = (
                "Endpoint associated with paid or premium content is accessible anonymously."
            )
        recommendation = (
            "Enforce authentication/authorization middleware before returning endpoint data."
        )
        evidence = {
            "url": candidate.url,
            "method": candidate.method,
            "observed_status": probe.status_code,
            "tags": candidate.tags,
            "sources": candidate.sources,
        }
        probe.recommendation = recommendation
        probe.risk_level = severity.value
        probe.risk_severity = severity
        return Finding(
            id="auth_endpoint_missing_guard",
            severity=severity,
            title="Endpoint lacks expected authentication",
            description=description,
            recommendation=recommendation,
            evidence=evidence,
        )


class AuthEndpointAgent:
    """Discovers and probes endpoints to highlight auth weaknesses."""

    SAFE_METHODS = ("GET", "HEAD", "OPTIONS")

    def __init__(self, config: Optional[AgentCConfig] = None):
        self.config = config or AgentCConfig()

    def scan(
        self,
        url: str,
        rendered_dom: Optional[str] = None,
        network_log: Optional[Sequence[Dict[str, Any]]] = None,
        cookie_jar: Optional[Dict[str, str]] = None,
        safe_endpoints_whitelist: Optional[Sequence[str]] = None,
    ) -> AuthEndpointScanResult:
        base_url = self._normalize_url(url)
        dom_text = rendered_dom or self._load_text(self.config.rendered_dom_path)
        log_entries = network_log or self._load_json(self.config.network_log_path) or []

        discovery = EndpointDiscovery(
            base_url=base_url,
            rendered_dom=dom_text,
            network_log=log_entries,
            max_endpoints=self.config.max_endpoints,
        )
        endpoints = discovery.discover()
        whitelist = tuple(safe_endpoints_whitelist or self.config.safe_endpoints_whitelist)
        probes = self._probe(endpoints, base_url, cookie_jar or {}, whitelist)
        findings = EndpointAnalyzer.generate_findings(probes)
        return AuthEndpointScanResult(
            target=url,
            base_url=base_url,
            endpoints=endpoints,
            probes=probes,
            findings=findings,
        )

    def _probe(
        self,
        endpoints: Sequence[EndpointCandidate],
        base_url: str,
        cookie_jar: Dict[str, str],
        whitelist: Sequence[str],
    ) -> List[EndpointProbeResult]:
        limited = list(endpoints)[: self.config.max_endpoints]
        if not limited:
            return []
        coro = self._probe_async(limited, base_url, cookie_jar, whitelist)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return asyncio.run(coro)

    async def _probe_async(
        self,
        endpoints: Sequence[EndpointCandidate],
        base_url: str,
        cookie_jar: Dict[str, str],
        whitelist: Sequence[str],
    ) -> List[EndpointProbeResult]:
        allowed_hosts = self._build_allowed_hosts(base_url, whitelist)
        headers = {"User-Agent": self.config.user_agent}
        semaphore = asyncio.Semaphore(max(1, self.config.concurrency))
        records: List[EndpointProbeResult] = []

        async with httpx.AsyncClient(
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            follow_redirects=self.config.follow_redirects,
        ) as client:
            tasks = []
            for candidate in endpoints:
                if not self._host_allowed(candidate.url, allowed_hosts):
                    continue
                tasks.append(
                    asyncio.create_task(
                        self._probe_candidate_async(
                            client, candidate, headers, cookie_jar, semaphore
                        )
                    )
                )

            if not tasks:
                return []

            for task in asyncio.as_completed(tasks):
                try:
                    record = await task
                except Exception as exc:  # pragma: no cover - defensive
                    record = EndpointProbeResult(
                        candidate=EndpointCandidate(
                            method="GET", url=base_url, sources=["probe"], metadata={}
                        ),
                        status_code=None,
                        reason=None,
                        headers={},
                        body_preview=None,
                        elapsed_ms=None,
                        error=str(exc),
                    )
                if record:
                    records.append(record)
        return records

    async def _probe_candidate_async(
        self,
        client: httpx.AsyncClient,
        candidate: EndpointCandidate,
        headers: Dict[str, str],
        cookie_jar: Dict[str, str],
        semaphore: asyncio.Semaphore,
    ) -> EndpointProbeResult:
        method = (candidate.method or "GET").upper()
        if method not in self.SAFE_METHODS:
            method = "GET"
        async with semaphore:
            record = await self._probe_single_async(
                client, candidate, method, headers, cookie_jar
            )
        detection = SensitiveContentDetector.analyze(record.body_preview)
        record.sensitivity_score = detection.score
        record.sensitivity_signals = detection.signals
        record.sensitivity_confidence = detection.confidence
        record.sensitivity_tags = detection.tags
        RiskAssessor.annotate(record)
        return record

    async def _probe_single_async(
        self,
        client: httpx.AsyncClient,
        candidate: EndpointCandidate,
        method: str,
        headers: Dict[str, str],
        cookie_jar: Dict[str, str],
    ) -> EndpointProbeResult:
        sanitized_headers = dict(headers)
        candidate_headers = candidate.metadata.get("headers", {})
        for key, value in candidate_headers.items():
            lowered = key.lower()
            if lowered in {"authorization", "cookie"}:
                continue
            sanitized_headers.setdefault(key, value)

        start = time.perf_counter()
        try:
            response = await client.request(
                method,
                candidate.url,
                headers=sanitized_headers,
                cookies=cookie_jar,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            body_preview = self._extract_preview(response)
            return EndpointProbeResult(
                candidate=candidate,
                status_code=response.status_code,
                reason=response.reason_phrase,
                headers=dict(response.headers),
                body_preview=body_preview,
                elapsed_ms=round(elapsed_ms, 2),
                probed_at=datetime.now(timezone.utc).isoformat(),
            )
        except httpx.HTTPError as exc:
            return EndpointProbeResult(
                candidate=candidate,
                status_code=None,
                reason=None,
                headers={},
                body_preview=None,
                elapsed_ms=None,
                error=str(exc),
                probed_at=datetime.now(timezone.utc).isoformat(),
            )

    def _extract_preview(self, response: httpx.Response) -> Optional[str]:
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                data = response.json()
                snippet = json.dumps(data, indent=2)
            except Exception:
                snippet = response.text
        else:
            snippet = response.text
        return snippet[: self.config.max_body_preview]

    def _build_allowed_hosts(self, base_url: str, whitelist: Sequence[str]) -> Sequence[str]:
        base_host = urlparse(base_url).hostname or ""
        allowed = [base_host]
        for item in whitelist:
            if item not in allowed:
                allowed.append(item.lower())
        return allowed

    def _host_allowed(self, url: str, allowed_hosts: Sequence[str]) -> bool:
        host = urlparse(url).hostname
        if not host:
            return True
        host = host.lower()
        for allowed in allowed_hosts:
            if not allowed:
                continue
            if host == allowed or host.endswith(f".{allowed}"):
                return True
        return self.config.allow_external_hosts

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme:
            return url
        return f"https://{url}"

    @staticmethod
    def _load_text(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return None

    @staticmethod
    def _load_json(path: Optional[str]) -> Optional[Sequence[Dict[str, Any]]]:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
            return data.get("logs") if isinstance(data, dict) else None
        except OSError:
            return None


# ---------------------------------------------------------------------------
# Playwright capture wrapper (optional)
# ---------------------------------------------------------------------------


@dataclass
class BrowserCaptureConfig:
    """Controls how Playwright navigates the site to collect artifacts."""

    headless: bool = True
    viewport: Optional[Dict[str, int]] = field(
        default_factory=lambda: {"width": 1280, "height": 720}
    )
    navigation_timeout_ms: int = 15000
    post_load_wait_ms: int = 2500
    extra_paths: Sequence[str] = ()
    user_agent: Optional[str] = None
    block_media: bool = True
    include_console: bool = False


@dataclass
class BrowserArtifacts:
    rendered_dom: str
    network_log: List[Dict[str, Any]]
    console_messages: List[Dict[str, Any]]


class PlaywrightAuthEndpointAgent:
    """Drives a browser to capture DOM/network logs, then invokes AuthEndpointAgent."""

    def __init__(
        self,
        capture_config: Optional[BrowserCaptureConfig] = None,
        auth_agent_config: Optional[AgentCConfig] = None,
    ) -> None:
        if async_playwright is None:
            raise ImportError(
                "playwright is not installed. Install it via `pip install playwright` "
                "and run `playwright install chromium`."
            ) from _PLAYWRIGHT_IMPORT_ERROR

        self.capture_config = capture_config or BrowserCaptureConfig()
        self.auth_agent = AuthEndpointAgent(auth_agent_config)

    def scan(self, url: str) -> AuthEndpointScanResult:
        artifacts = self._capture(url)
        return self.auth_agent.scan(
            url,
            rendered_dom=artifacts.rendered_dom,
            network_log=artifacts.network_log,
        )

    def _capture(self, url: str) -> BrowserArtifacts:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self._capture_async(url))
            finally:
                new_loop.close()
        return asyncio.run(self._capture_async(url))

    async def _capture_async(self, url: str) -> BrowserArtifacts:
        assert async_playwright is not None  # for type checkers
        async with async_playwright() as p:
            browser = await self._launch_browser(p)
            try:
                context = await self._create_context(browser)
                page = await context.new_page()
                network_log: List[Dict[str, Any]] = []
                console_messages: List[Dict[str, Any]] = []

                self._wire_request_logging(page, network_log)
                if self.capture_config.include_console:
                    self._wire_console_logging(page, console_messages)

                await self._navigate(page, url)
                for extra in self.capture_config.extra_paths:
                    target = extra if "://" in extra else urljoin(url, extra)
                    await self._navigate(page, target)

                await page.wait_for_timeout(self.capture_config.post_load_wait_ms)
                rendered_dom = await page.content()
                await context.close()
                return BrowserArtifacts(rendered_dom, network_log, console_messages)
            finally:
                await browser.close()

    async def _launch_browser(self, playwright: Playwright) -> Browser:
        return await playwright.chromium.launch(headless=self.capture_config.headless)

    async def _create_context(self, browser: Browser) -> BrowserContext:
        context_kwargs: Dict[str, Any] = {}
        if self.capture_config.viewport:
            context_kwargs["viewport"] = self.capture_config.viewport
        if self.capture_config.user_agent:
            context_kwargs["user_agent"] = self.capture_config.user_agent

        context = await browser.new_context(**context_kwargs)
        if self.capture_config.block_media:

            async def block_media(route, request):
                if request.resource_type in {"image", "media", "font"}:
                    await route.abort()
                else:
                    await route.continue_()

            await context.route("**/*", block_media)
        return context

    def _wire_request_logging(self, page: "Page", network_log: List[Dict[str, Any]]) -> None:
        def handle_request(request):
            try:
                headers = request.headers
            except Exception:  # pragma: no cover - defensive
                headers = {}
            network_log.append(
                {
                    "url": request.url,
                    "method": request.method,
                    "headers": headers,
                    "resourceType": request.resource_type,
                    "postData": request.post_data,
                }
            )

        page.on("requestfinished", handle_request)
        page.on("requestfailed", handle_request)

    def _wire_console_logging(self, page: "Page", console_messages: List[Dict[str, Any]]) -> None:
        def handle_console(msg):
            console_messages.append(
                {
                    "type": msg.type,
                    "text": msg.text,
                    "location": msg.location,
                }
            )

        page.on("console", handle_console)

    async def _navigate(self, page: "Page", url: str) -> None:
        try:
            await page.goto(
                url,
                wait_until="networkidle",
                timeout=self.capture_config.navigation_timeout_ms,
            )
        except Exception:
            with contextlib.suppress(Exception):
                await page.goto(url, wait_until="load")


def run_cli() -> None:
    """Playwright-enabled CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture DOM/network data with Playwright and run Auth Endpoint Agent.",
    )
    parser.add_argument("url", help="Target URL to visit.")
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run the browser in headful mode (default: headless).",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Additional path to navigate after the main URL (relative or absolute).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable findings.",
    )
    args = parser.parse_args()

    capture_cfg = BrowserCaptureConfig(
        headless=not args.headful,
        extra_paths=tuple(args.extra_path or []),
    )
    agent = PlaywrightAuthEndpointAgent(capture_cfg)
    result = agent.scan(args.url)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"[target] {result.target}")
    print(f"[endpoints] {len(result.endpoints)} discovered / {len(result.probes)} probed")
    print(f"[summary] {result.summary_text}")
    for finding in result.findings:
        evidence = finding.evidence or {}
        location = evidence.get("url")
        location_suffix = f" at {location}" if location else ""
        print(
            f"- ({finding.severity.value.upper()}) {finding.title}"
            f"{location_suffix}: {finding.description}"
        )
    if not result.findings:
        print("No unauthenticated sensitive endpoints detected.")


if __name__ == "__main__":  # pragma: no cover
    run_cli()
