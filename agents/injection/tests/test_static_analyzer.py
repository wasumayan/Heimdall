from agents.injection.injection_agent import (
    HTTPRequestRecord,
    StaticAnalyzer,
    InjectionFinding,
    Severity,
)


def _make_record(body: str, headers: dict | None = None) -> HTTPRequestRecord:
    headers = headers or {"Content-Type": "text/html"}
    return HTTPRequestRecord(
        method="GET",
        url="https://example.com",
        status_code=200,
        headers=headers,
        elapsed_ms=5.0,
        body=body,
    )


def test_static_analyzer_detects_inline_handlers() -> None:
    record = _make_record(
        """
        <html>
            <body>
                <form action="/submit" method="post" onsubmit="doThing()">
                    <input type="text" name="q" onclick="alert(1)">
                </form>
            </body>
        </html>
        """
    )

    findings, forms = StaticAnalyzer.analyze_records([record])

    assert forms, "Expected the analyzer to capture at least one form."
    assert any(
        finding.id == "injection_inline_event_handlers" for finding in findings
    ), "Inline event handler finding not produced."


def test_static_analyzer_flags_missing_csp() -> None:
    record = _make_record("<html><body>Hello</body></html>")

    findings, _ = StaticAnalyzer.analyze_records([record])

    csp_findings = [finding for finding in findings if finding.id == "injection_missing_csp"]
    assert csp_findings, "Missing CSP header should trigger a finding."
    assert csp_findings[0].severity == Severity.MEDIUM

