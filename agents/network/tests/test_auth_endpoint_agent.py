from agents.network.auth_endpoint_agent import (
    EndpointAnalyzer,
    EndpointCandidate,
    EndpointDiscovery,
    EndpointProbeResult,
    SensitiveContentDetector,
)
from agents.network.network_agent import Severity


def build_probe(candidate, status=200, score=0, signals=None, body="sample", risk=Severity.INFO):
    tags = []
    for signal in signals or []:
        if ":" in signal:
            tags.append(signal.split(":", 1)[1])
        else:
            tags.append(signal)
    return EndpointProbeResult(
        candidate=candidate,
        status_code=status,
        reason="OK",
        headers={},
        body_preview=body,
        elapsed_ms=10.0,
        sensitivity_score=score,
        sensitivity_signals=signals or [],
        sensitivity_confidence=0.8 if score else 0.0,
        sensitivity_tags=tags,
        risk_level=risk.value,
        recommendation=None,
        probed_at="2025-01-01T00:00:00Z",
        risk_severity=risk,
    )


def test_discovery_merges_sources_and_tags():
    dom = '<form action="/admin/export" method="POST"></form>'
    network_log = [
        {
            "url": "/admin/export",
            "method": "GET",
            "request": {"headers": {"Authorization": "Bearer token"}},
        }
    ]
    discovery = EndpointDiscovery(
        base_url="https://example.com",
        rendered_dom=dom,
        network_log=network_log,
        max_endpoints=5,
    )
    endpoints = discovery.discover()

    assert len(endpoints) == 1
    candidate = endpoints[0]
    assert candidate.requires_auth is True
    assert set(candidate.sources) == {"network_log", "dom_form"}
    assert "sensitive-path" in candidate.tags


def test_sensitive_content_detector_flags_pii_and_tokens():
    body = "Contact admin@example.com with token: secret=abcdEFGH12345678"
    result = SensitiveContentDetector.analyze(body)

    assert result.score >= 4
    assert "email" in result.tags
    assert any(sig.startswith("token:") for sig in result.signals)


def test_endpoint_analyzer_reports_sensitive_exposure():
    candidate = EndpointCandidate(
        method="GET",
        url="https://example.com/api/export",
        sources=["network_log"],
        requires_auth=False,
    )
    probe = build_probe(
        candidate,
        score=8,
        signals=["pii:email"],
        body="email=a@example.com",
        risk=Severity.HIGH,
    )

    findings = EndpointAnalyzer.generate_findings([probe])
    assert findings
    assert findings[0].id == "auth_endpoint_sensitive_response"
    assert findings[0].severity in {Severity.CRITICAL, Severity.HIGH}


def test_endpoint_analyzer_reports_auth_bypass():
    candidate = EndpointCandidate(
        method="GET",
        url="https://example.com/api/premium",
        sources=["network_log"],
        requires_auth=True,
    )
    probe = build_probe(candidate, status=200, score=0, signals=[], risk=Severity.HIGH)

    findings = EndpointAnalyzer.generate_findings([probe])
    assert findings
    assert findings[0].id == "auth_endpoint_missing_guard"
    assert findings[0].severity == Severity.HIGH
