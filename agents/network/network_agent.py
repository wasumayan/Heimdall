"""Network/Request-Response layer agent for Heimdall.

This module focuses purely on network response inspection. It probes a target
URL, collects HTTP response metadata, and emits structured findings for
misconfigured security headers and permissive CORS policies. The agent is
deliberately standalone and does not depend on the BRAMA repository so it can
evolve independently alongside the Client/Browser surface agent.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import httpx

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


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
class Finding:
    """Represents a single security issue surfaced by the agent."""

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
class ResponseRecord:
    """Captured HTTP response metadata for later analysis."""

    method: str
    url: str
    status_code: Optional[int]
    headers: Dict[str, str]
    elapsed_ms: Optional[float]
    error: Optional[str] = None

    def simplified_headers(self) -> Dict[str, str]:
        return {k.lower(): v for k, v in self.headers.items()}


@dataclass
class AgentConfig:
    """Runtime configuration for the agent."""

    timeout: float = 8.0
    verify_ssl: bool = True
    follow_redirects: bool = True
    methods: Tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    user_agent: str = "Heimdall-Network-Agent/0.1"
    extra_paths: Tuple[str, ...] = ()


@dataclass
class NetworkScanResult:
    """Top-level payload returned by the agent."""

    target: str
    resolved_url: str
    responses: List[ResponseRecord]
    findings: List[Finding]
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
            "response_count": len(self.responses),
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
        _ = self.summary_text  # populate cache if needed
        return self._summary_source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "resolved_url": self.resolved_url,
            "responses": [asdict(resp) for resp in self.responses],
            "findings": [finding.to_dict() for finding in self.findings],
            "summary": self.summary,
            "summary_text": self.summary_text,
            "summary_source": self.summary_source,
            "generated_at": self.generated_at,
        }


def _finding_location(finding: Finding) -> Optional[str]:
    evidence = finding.evidence or {}
    method = evidence.get("method")
    url = evidence.get("url")
    path = evidence.get("path")

    if method and url:
        return f"{method} {url}"
    if url:
        return str(url)
    if method and path:
        return f"{method} {path}"
    if path:
        return str(path)
    if method:
        return str(method)
    return None


class LLMSummaryGenerator:
    """Produces natural-language summaries via xAI with a deterministic fallback."""

    _DEFAULT_MODEL = "grok-2-latest"
    _DEFAULT_TIMEOUT = 30

    @classmethod
    def generate(cls, scan: NetworkScanResult) -> Tuple[str, str]:
        llm_summary = cls._try_xai_summary(scan)
        if llm_summary:
            return llm_summary, "xai"
        return cls._heuristic_summary(scan), "heuristic"

    @classmethod
    def _try_xai_summary(cls, scan: NetworkScanResult) -> Optional[str]:
        api_key = os.environ.get("TRBUOEyXCAcQzE6b9c2XtK8zdh0TAOldIGFU0pKG1cOo2D6IrCwL6RtnydQ5WiPiulLqHFnwmLLgNVQ8")
        if not api_key or OpenAI is None:
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
            "You are a security engineer summarizing HTTP response policy issues. "
            "Write 2-4 sentences, grouping similar problems, naming severities, "
            "and pointing to the exact method+URL for each item. Finish with a remediation note."
        )
        user_prompt = (
            "Summarize these scan results for a technical stakeholder:\n"
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
    def _build_payload(scan: NetworkScanResult) -> Dict[str, Any]:
        responses = [
            {
                "method": resp.method,
                "url": resp.url,
                "status": resp.status_code,
                "error": resp.error,
            }
            for resp in scan.responses[:5]
        ]
        findings = [
            {
                "id": finding.id,
                "title": finding.title,
                "severity": finding.severity.value,
                "description": finding.description,
                "location": _finding_location(finding),
                "recommendation": finding.recommendation,
            }
            for finding in scan.findings
        ]
        return {
            "target": scan.target,
            "resolved_url": scan.resolved_url,
            "findings": findings,
            "responses": responses,
        }

    @staticmethod
    def _heuristic_summary(scan: NetworkScanResult) -> str:
        if not scan.findings:
            friendly_target = scan.resolved_url or scan.target
            return f"No network-layer misconfigurations detected for {friendly_target}."

        top_finding = min(
            scan.findings,
            key=lambda finding: (_SEVERITY_RANK.get(finding.severity, 99), finding.id),
        )
        location = _finding_location(top_finding)
        location_clause = f" at {location}" if location else ""
        return (
            f"Found {len(scan.findings)} network-layer issue(s). "
            f"Highest severity {top_finding.severity.value.upper()} - {top_finding.title}"
            f"{location_clause}: {top_finding.description}"
        )


class SecurityHeaderAnalyzer:
    """Encapsulates the heuristics for header and CORS analysis."""

    _HEADER_EXPECTATIONS: Tuple[Tuple[str, Severity, str], ...] = (
        ("strict-transport-security", Severity.HIGH, "Enforce HTTPS with HSTS."),
        (
            "content-security-policy",
            Severity.HIGH,
            "Define CSP to limit script/style execution sources.",
        ),
        (
            "x-content-type-options",
            Severity.MEDIUM,
            "Prevent MIME type sniffing with X-Content-Type-Options: nosniff.",
        ),
        ("x-frame-options", Severity.MEDIUM, "Use DENY or SAMEORIGIN to stop clickjacking."),
        (
            "referrer-policy",
            Severity.LOW,
            "Limit referer leakage via a restrictive Referrer-Policy header.",
        ),
        (
            "permissions-policy",
            Severity.LOW,
            "Explicitly declare Permissions-Policy to curb powerful APIs.",
        ),
    )

    @classmethod
    def analyze(cls, records: Sequence[ResponseRecord]) -> List[Finding]:
        findings: List[Finding] = []
        primary = cls._select_primary_record(records)
        if not primary:
            findings.append(
                Finding(
                    id="network_no_response",
                    severity=Severity.CRITICAL,
                    title="Unable to retrieve HTTP response",
                    description="All attempts to reach the target failed. Review network connectivity, DNS, or TLS configuration.",
                    recommendation="Verify the host is reachable and serving HTTP/S content before re-running the agent.",
                    evidence={"attempts": [asdict(r) for r in records]},
                )
            )
            return findings

        headers = primary.simplified_headers()
        findings.extend(cls._missing_header_findings(primary, headers))
        findings.extend(cls._cors_findings(records))
        findings.extend(cls._transport_findings(records))
        return findings

    @staticmethod
    def _select_primary_record(records: Sequence[ResponseRecord]) -> Optional[ResponseRecord]:
        for record in records:
            if record.status_code and record.headers and (400 > record.status_code >= 200):
                return record
        for record in records:
            if record.status_code and record.headers:
                return record
        return None

    @classmethod
    def _missing_header_findings(
        cls, record: ResponseRecord, headers: Dict[str, str]
    ) -> List[Finding]:
        missing: List[Dict[str, Any]] = []
        parsed = urlparse(record.url)
        for header_name, severity, remediation in cls._HEADER_EXPECTATIONS:
            if header_name == "strict-transport-security" and parsed.scheme != "https":
                continue  # Only meaningful on HTTPS responses
            if header_name not in headers:
                missing.append(
                    {
                        "header": header_name,
                        "severity": severity,
                        "remediation": remediation,
                    }
                )

        if not missing:
            return []

        highest = min(
            missing,
            key=lambda item: _SEVERITY_RANK.get(item["severity"], 99),
        )["severity"]
        return [
            Finding(
                id="network_missing_security_headers",
                severity=highest,
                title="Missing security headers",
                description=f"{len(missing)} high-value security headers are absent from the response.",
                recommendation="Add the missing headers at the edge (load balancer/CDN) or web server level.",
                evidence={
                    "url": record.url,
                    "method": record.method,
                    "missing_headers": missing,
                },
            )
        ]

    @staticmethod
    def _cors_findings(records: Sequence[ResponseRecord]) -> List[Finding]:
        findings: List[Finding] = []
        for record in records:
            headers = record.simplified_headers()
            if not headers:
                continue

            allow_origin = headers.get("access-control-allow-origin")
            allow_credentials = headers.get("access-control-allow-credentials", "").lower()
            allow_headers = headers.get("access-control-allow-headers", "")
            allow_methods = headers.get("access-control-allow-methods", "")

            if allow_origin == "*":
                severity = Severity.HIGH if allow_credentials == "true" else Severity.MEDIUM
                findings.append(
                    Finding(
                        id="network_permissive_cors",
                        severity=severity,
                        title="Permissive CORS policy",
                        description="Response allows any origin via Access-Control-Allow-Origin: *. This enables arbitrary websites to read responses via JavaScript.",
                        recommendation="Return a specific allowlist of origins (or echo Origin with proper validation) and avoid wildcards.",
                        evidence={
                            "url": record.url,
                            "method": record.method,
                            "headers": {
                                "access-control-allow-origin": allow_origin,
                                "access-control-allow-credentials": allow_credentials,
                                "access-control-allow-headers": allow_headers,
                                "access-control-allow-methods": allow_methods,
                            },
                        },
                    )
                )
                continue

            if allow_credentials == "true" and allow_origin in ("*", None):
                findings.append(
                    Finding(
                        id="network_credentials_without_origin",
                        severity=Severity.HIGH,
                        title="Credentials allowed without trusted origin",
                        description="Access-Control-Allow-Credentials is true but the Access-Control-Allow-Origin header is missing or wildcarded. Browsers will refuse wildcard + credentials, but the configuration signals a broader trust issue.",
                        recommendation="Restrict credentialed responses to vetted origins and ensure Access-Control-Allow-Origin mirrors the request origin only when it is on an allowlist.",
                        evidence={
                            "url": record.url,
                            "method": record.method,
                            "headers": {
                                "access-control-allow-origin": allow_origin,
                                "access-control-allow-credentials": allow_credentials,
                            },
                        },
                    )
                )

            if "*" in allow_headers.lower() or "*" in allow_methods.lower():
                findings.append(
                    Finding(
                        id="network_overly_broad_preflight",
                        severity=Severity.MEDIUM,
                        title="Overly broad CORS preflight allowances",
                        description="Wildcard Access-Control-Allow-Headers or Methods grants broad JavaScript access to custom headers or verbs.",
                        recommendation="Limit allowed headers/methods to the minimum needed for public APIs.",
                        evidence={
                            "url": record.url,
                            "method": record.method,
                            "headers": {
                                "access-control-allow-headers": allow_headers,
                                "access-control-allow-methods": allow_methods,
                            },
                        },
                    )
                )
        return findings

    @staticmethod
    def _transport_findings(records: Sequence[ResponseRecord]) -> List[Finding]:
        findings: List[Finding] = []
        for record in records:
            parsed = urlparse(record.url)
            if parsed.scheme == "http":
                findings.append(
                    Finding(
                        id="network_http_insecure_transport",
                        severity=Severity.HIGH,
                        title="Insecure HTTP endpoint",
                        description="Endpoint was served over plain HTTP, enabling interception and header manipulation.",
                        recommendation="Force HTTPS and redirect HTTP traffic to TLS with HSTS enabled.",
                        evidence={"url": record.url, "method": record.method},
                    )
                )
        return findings


class NetworkPolicyAgent:
    """Facade that orchestrates HTTP collection and header analysis."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def scan(self, url: str, extra_paths: Optional[Iterable[str]] = None) -> NetworkScanResult:
        normalized = self._normalize_url(url)
        paths = list(self.config.extra_paths)
        if extra_paths:
            paths.extend(extra_paths)

        records = self._collect_responses(normalized, paths)
        findings = SecurityHeaderAnalyzer.analyze(records)
        resolved_url = records[0].url if records else normalized
        return NetworkScanResult(
            target=url,
            resolved_url=resolved_url,
            responses=records,
            findings=findings,
        )

    def _collect_responses(self, base_url: str, paths: Sequence[str]) -> List[ResponseRecord]:
        records: List[ResponseRecord] = []
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
    ) -> ResponseRecord:
        start = time.perf_counter()
        try:
            response = client.request(method, url, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000
            return ResponseRecord(
                method=method,
                url=str(response.url),
                status_code=response.status_code,
                headers=dict(response.headers),
                elapsed_ms=round(elapsed, 2),
            )
        except httpx.HTTPError as exc:
            return ResponseRecord(
                method=method,
                url=url,
                status_code=None,
                headers={},
                elapsed_ms=None,
                error=str(exc),
            )

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Network/Request-Response layer agent for Heimdall."
    )
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
        default=8.0,
        help="HTTP client timeout in seconds (default: 8).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS verification (not recommended).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )

    args = parser.parse_args()
    config = AgentConfig(timeout=args.timeout, verify_ssl=not args.insecure)
    agent = NetworkPolicyAgent(config)
    result = agent.scan(args.url, extra_paths=args.path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"[target] {result.target}")
    print(f"[resolved] {result.resolved_url}")
    print(f"[responses] {len(result.responses)}")
    print(f"[summary] {result.summary_text} (source: {result.summary_source})")
    for finding in result.findings:
        location = _finding_location(finding)
        location_suffix = f" at {location}" if location else ""
        print(
            f"- ({finding.severity.value.upper()}) {finding.title}"
            f"{location_suffix}: {finding.description}"
        )
    if not result.findings:
        print("No network-layer issues detected.")


if __name__ == "__main__":
    run_cli()
