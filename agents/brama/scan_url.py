#!/usr/bin/env python3
"""
BRAMA URL Scanner - Programmatic interface for Heimdall integration
This script allows BRAMA to be called from command line with a URL and returns JSON results.
Includes comprehensive red-teaming capabilities.
"""
import sys
import json
import os
import re
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from agentBrama import CybersecurityAgent
    from red_team_scanner import RedTeamScanner
except ImportError as e:
    print(json.dumps({
        "error": f"Failed to import BRAMA agent: {str(e)}",
        "findings": [],
        "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0}
    }), file=sys.stderr)
    sys.exit(1)


def scan_url(url: str) -> dict:
    """
    Scan a URL using BRAMA with comprehensive red-teaming capabilities.
    Returns structured results including domain analysis and red-team findings.
    """
    try:
        # Normalize URL
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url
        
        findings = []
        finding_id = 1
        
        # Extract domain from URL
        domain_match = re.search(r'https?://([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', url)
        if not domain_match:
            domain_match = re.search(r'([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', url)
        
        if not domain_match:
            return {
                "findings": [],
                "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0},
                "error": "Could not extract domain from URL"
            }
        
        domain = domain_match.group(1)
        
        # 1. Run domain threat intelligence analysis (if agent available)
        try:
            agent = CybersecurityAgent()
            domain_analysis = agent.analyze_domain(domain)
            
            # Parse domain analysis
            analysis_text = str(domain_analysis).lower()
            
            if any(keyword in analysis_text for keyword in ['malicious', 'phishing', 'malware', 'threat', 'dangerous']):
                severity = "high"
            elif any(keyword in analysis_text for keyword in ['suspicious', 'warning', 'risk', 'concern']):
                severity = "medium"
            else:
                severity = "low"
            
            findings.append({
                "id": f"brama_{finding_id:03d}",
                "severity": severity,
                "title": f"Domain Threat Intelligence: {domain}",
                "description": str(domain_analysis)[:500],
                "location": url,
                "recommendation": "Review the domain analysis results and take appropriate security measures",
                "fix_suggestion": "Implement security best practices based on the analysis findings",
            })
            finding_id += 1
        except Exception as e:
            # Continue even if domain analysis fails
            pass
        
        # 2. Run comprehensive red-team security scan
        try:
            scanner = RedTeamScanner(url)
            red_team_results = scanner.scan_all()
            
            # Convert red-team findings to Heimdall format
            for finding in scanner.findings:
                findings.append({
                    "id": f"brama_{finding_id:03d}",
                    "severity": finding['severity'],
                    "title": finding['title'],
                    "description": finding['description'],
                    "location": url,
                    "recommendation": finding['recommendation'],
                    "fix_suggestion": finding.get('fix', finding['recommendation']),
                })
                finding_id += 1
            
            # Add summary findings for each scan category
            if red_team_results.get('security_headers', {}).get('missing'):
                missing = red_team_results['security_headers']['missing']
                findings.append({
                    "id": f"brama_{finding_id:03d}",
                    "severity": red_team_results['security_headers'].get('severity', 'medium'),
                    "title": "Security Headers Analysis",
                    "description": f"Missing security headers: {', '.join(missing)}. {len(red_team_results['security_headers'].get('issues', []))} issues found.",
                    "location": url,
                    "recommendation": "Implement missing security headers",
                    "fix_suggestion": "Add X-Frame-Options, X-Content-Type-Options, HSTS, and CSP headers",
                })
                finding_id += 1
            
            if red_team_results.get('endpoints', {}).get('discovered'):
                discovered = red_team_results['endpoints']['discovered']
                findings.append({
                    "id": f"brama_{finding_id:03d}",
                    "severity": "medium",
                    "title": "Endpoint Discovery",
                    "description": f"Discovered {len(discovered)} accessible endpoints: {', '.join(discovered[:5])}",
                    "location": url,
                    "recommendation": "Review and secure discovered endpoints",
                    "fix_suggestion": "Implement proper access controls and authentication for sensitive endpoints",
                })
                finding_id += 1
            
            if red_team_results.get('cors', {}).get('vulnerable'):
                findings.append({
                    "id": f"brama_{finding_id:03d}",
                    "severity": "high",
                    "title": "CORS Misconfiguration",
                    "description": f"CORS vulnerabilities detected: {', '.join(red_team_results['cors'].get('issues', []))}",
                    "location": url,
                    "recommendation": "Fix CORS configuration",
                    "fix_suggestion": "Restrict CORS to specific trusted origins and avoid wildcard with credentials",
                })
                finding_id += 1
            
            if red_team_results.get('technology_stack', {}).get('technologies'):
                techs = red_team_results['technology_stack']['technologies']
                findings.append({
                    "id": f"brama_{finding_id:03d}",
                    "severity": "low",
                    "title": "Technology Stack Fingerprint",
                    "description": f"Detected technologies: {', '.join(techs)}",
                    "location": url,
                    "recommendation": "Review technology stack disclosure",
                    "fix_suggestion": "Consider obfuscating technology stack information",
                })
                finding_id += 1
                
        except Exception as e:
            # Continue even if red-team scan fails
            pass
        
        # Calculate summary
        summary = {
            "total_findings": len(findings),
            "high": sum(1 for f in findings if f["severity"] == "high"),
            "medium": sum(1 for f in findings if f["severity"] == "medium"),
            "low": sum(1 for f in findings if f["severity"] == "low"),
        }
        
        return {
            "findings": findings,
            "summary": summary
        }
            
    except ValueError as e:
        # API keys missing
        return {
            "findings": [],
            "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0},
            "error": f"API keys not configured: {str(e)}"
        }
    except Exception as e:
        return {
            "findings": [],
            "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0},
            "error": f"Scan failed: {str(e)}"
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: python scan_url.py <url>",
            "findings": [],
            "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0}
        }), file=sys.stderr)
        sys.exit(1)
    
    url = sys.argv[1]
    result = scan_url(url)
    print(json.dumps(result, indent=2))

