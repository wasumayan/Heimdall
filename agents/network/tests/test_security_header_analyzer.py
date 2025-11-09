import pytest

from agents.network.network_agent import (
    NetworkScanResult,
    ResponseRecord,
    SecurityHeaderAnalyzer,
    Severity,
)


def build_record(headers, url="https://example.com", method="GET", status=200):
    return ResponseRecord(
        method=method,
        url=url,
        status_code=status,
        headers=headers,
        elapsed_ms=12.3,
    )


def test_missing_security_headers_detection():
    record = build_record(headers={"Server": "nginx"})
    findings = SecurityHeaderAnalyzer.analyze([record])

    header_finding = next(f for f in findings if f.id == "network_missing_security_headers")
    assert header_finding.severity == Severity.HIGH
    assert len(header_finding.evidence["missing_headers"]) >= 4


def test_permissive_cors_flagged():
    record = build_record(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "*,Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, *",
        }
    )
    findings = SecurityHeaderAnalyzer.analyze([record])

    cors_finding = next(f for f in findings if f.id == "network_permissive_cors")
    assert cors_finding.severity == Severity.HIGH


def test_insecure_http_transport():
    record = build_record(headers={"Server": "nginx"}, url="http://example.com")
    findings = SecurityHeaderAnalyzer.analyze([record])

    http_finding = next(f for f in findings if f.id == "network_http_insecure_transport")
    assert http_finding.severity == Severity.HIGH


def test_summary_text_without_findings():
    record = build_record(headers={"Strict-Transport-Security": "max-age=100"})
    result = NetworkScanResult(
        target="https://example.com",
        resolved_url="https://example.com",
        responses=[record],
        findings=[],
    )

    assert "No network-layer misconfigurations detected" in result.summary_text


def test_summary_text_highlights_top_finding():
    record = build_record(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )
    findings = SecurityHeaderAnalyzer.analyze([record])
    result = NetworkScanResult(
        target="https://example.com",
        resolved_url="https://example.com",
        responses=[record],
        findings=findings,
    )

    assert "Found" in result.summary_text
    assert "network-layer issue" in result.summary_text
    assert "https://example.com" in result.summary_text
    assert result.summary_source == "heuristic"
