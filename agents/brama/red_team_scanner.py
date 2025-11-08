#!/usr/bin/env python3
"""
Red-Team Scanner Module for BRAMA
Performs comprehensive security scanning including:
- Security headers analysis
- SSL/TLS certificate checks
- Endpoint discovery
- Technology stack fingerprinting
- CORS misconfiguration
- HTTP methods testing
- Information disclosure checks
"""
import requests
import ssl
import socket
import re
import json
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Common paths for directory/file enumeration
COMMON_PATHS = [
    '/admin', '/administrator', '/login', '/wp-admin', '/wp-login.php',
    '/.git', '/.env', '/.git/config', '/config.php', '/backup',
    '/api', '/api/v1', '/api/v2', '/graphql', '/swagger', '/swagger.json',
    '/.well-known', '/robots.txt', '/sitemap.xml', '/.htaccess',
    '/phpinfo.php', '/test.php', '/info.php', '/server-status',
    '/.aws', '/.docker', '/docker-compose.yml', '/.k8s',
    '/v1', '/v2', '/v3', '/docs', '/documentation'
]

# Common technology indicators
TECH_INDICATORS = {
    'WordPress': ['wp-content', 'wp-includes', 'wp-admin', 'wordpress'],
    'Drupal': ['drupal.js', 'sites/all', 'modules/system'],
    'Joomla': ['joomla', 'administrator', 'components/com_'],
    'React': ['react', 'react-dom', '__REACT_DEVTOOLS'],
    'Vue.js': ['vue.js', '__vue__', 'vue-router'],
    'Angular': ['angular.js', 'ng-app', '@angular'],
    'Django': ['csrfmiddlewaretoken', 'django', 'admin/login'],
    'Flask': ['flask', 'werkzeug'],
    'Laravel': ['laravel_session', 'laravel', 'artisan'],
    'Express': ['express', 'x-powered-by'],
    'Apache': ['server: apache', 'apache'],
    'Nginx': ['server: nginx', 'nginx'],
    'PHP': ['.php', 'phpinfo', 'x-powered-by: php'],
    'Node.js': ['x-powered-by: express', 'node.js'],
    'ASP.NET': ['.aspx', 'x-powered-by: asp.net', 'viewstate']
}


class RedTeamScanner:
    """Comprehensive red-team security scanner."""
    
    def __init__(self, url: str, timeout: int = 10):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = False  # Allow self-signed certs for testing
        self.findings = []
        
    def scan_all(self) -> Dict[str, Any]:
        """Run all red-team scans."""
        results = {
            'security_headers': self.check_security_headers(),
            'ssl_tls': self.check_ssl_tls(),
            'endpoints': self.discover_endpoints(),
            'technology_stack': self.fingerprint_technology(),
            'cors': self.check_cors(),
            'http_methods': self.test_http_methods(),
            'information_disclosure': self.check_information_disclosure(),
            'findings': self.findings
        }
        return results
    
    def check_security_headers(self) -> Dict[str, Any]:
        """Check for missing or misconfigured security headers."""
        try:
            response = self.session.get(self.url, timeout=self.timeout, allow_redirects=True)
            headers = response.headers
            
            security_headers = {
                'X-Frame-Options': headers.get('X-Frame-Options'),
                'X-Content-Type-Options': headers.get('X-Content-Type-Options'),
                'X-XSS-Protection': headers.get('X-XSS-Protection'),
                'Strict-Transport-Security': headers.get('Strict-Transport-Security'),
                'Content-Security-Policy': headers.get('Content-Security-Policy'),
                'Referrer-Policy': headers.get('Referrer-Policy'),
                'Permissions-Policy': headers.get('Permissions-Policy'),
                'Server': headers.get('Server'),
                'X-Powered-By': headers.get('X-Powered-By')
            }
            
            missing_headers = []
            issues = []
            
            if not security_headers['X-Frame-Options']:
                missing_headers.append('X-Frame-Options')
                issues.append('Missing X-Frame-Options - vulnerable to clickjacking')
            
            if not security_headers['X-Content-Type-Options']:
                missing_headers.append('X-Content-Type-Options')
                issues.append('Missing X-Content-Type-Options - vulnerable to MIME sniffing')
            
            if not security_headers['Strict-Transport-Security']:
                missing_headers.append('Strict-Transport-Security')
                issues.append('Missing HSTS - vulnerable to protocol downgrade attacks')
            
            if not security_headers['Content-Security-Policy']:
                missing_headers.append('Content-Security-Policy')
                issues.append('Missing CSP - vulnerable to XSS attacks')
            
            if security_headers['Server']:
                issues.append(f'Server header disclosure: {security_headers["Server"]}')
                self.add_finding('medium', 'Server Header Disclosure', 
                               f'Server header reveals: {security_headers["Server"]}',
                               'Remove or obfuscate Server header')
            
            if security_headers['X-Powered-By']:
                issues.append(f'X-Powered-By header disclosure: {security_headers["X-Powered-By"]}')
                self.add_finding('low', 'X-Powered-By Header Disclosure',
                               f'X-Powered-By reveals: {security_headers["X-Powered-By"]}',
                               'Remove X-Powered-By header')
            
            # Map severity: critical -> high for Heimdall compatibility
            severity = 'high' if len(missing_headers) >= 3 else 'medium' if missing_headers else 'low'
            
            if missing_headers:
                self.add_finding(severity, 'Missing Security Headers',
                               f'Missing headers: {", ".join(missing_headers)}',
                               'Implement missing security headers')
            
            return {
                'headers': security_headers,
                'missing': missing_headers,
                'issues': issues,
                'severity': severity
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_ssl_tls(self) -> Dict[str, Any]:
        """Check SSL/TLS certificate and configuration."""
        try:
            parsed = urlparse(self.url)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            
            if port == 80:
                return {'error': 'Not an HTTPS endpoint'}
            
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
            
            issues = []
            
            # Check certificate expiration
            import datetime
            not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            days_until_expiry = (not_after - datetime.datetime.now()).days
            
            if days_until_expiry < 30:
                issues.append(f'Certificate expires in {days_until_expiry} days')
                self.add_finding('high', 'Certificate Expiring Soon',
                               f'SSL certificate expires in {days_until_expiry} days',
                               'Renew SSL certificate immediately')
            elif days_until_expiry < 90:
                issues.append(f'Certificate expires in {days_until_expiry} days')
                self.add_finding('medium', 'Certificate Expiring Soon',
                               f'SSL certificate expires in {days_until_expiry} days',
                               'Plan certificate renewal')
            
            # Check TLS version
            if version in ['TLSv1', 'TLSv1.1']:
                issues.append(f'Weak TLS version: {version}')
                self.add_finding('high', 'Weak TLS Version',
                               f'Using deprecated TLS version: {version}',
                               'Upgrade to TLS 1.2 or higher')
            
            # Check cipher
            if cipher:
                cipher_name = cipher[0]
                if 'RC4' in cipher_name or 'DES' in cipher_name or 'MD5' in cipher_name:
                    issues.append(f'Weak cipher: {cipher_name}')
                    self.add_finding('high', 'Weak Cipher Suite',
                                   f'Using weak cipher: {cipher_name}',
                                   'Disable weak cipher suites')
            
            return {
                'version': version,
                'cipher': cipher[0] if cipher else None,
                'certificate': {
                    'subject': dict(x[0] for x in cert['subject']),
                    'issuer': dict(x[0] for x in cert['issuer']),
                    'not_after': cert['notAfter']
                },
                'issues': issues
            }
        except Exception as e:
            return {'error': str(e)}
    
    def discover_endpoints(self) -> Dict[str, Any]:
        """Discover common endpoints and directories."""
        discovered = []
        accessible = []
        
        for path in COMMON_PATHS[:20]:  # Limit to 20 for speed
            try:
                test_url = urljoin(self.url, path)
                response = self.session.get(test_url, timeout=self.timeout, allow_redirects=False)
                
                if response.status_code in [200, 301, 302, 403]:
                    accessible.append({
                        'path': path,
                        'status': response.status_code,
                        'size': len(response.content)
                    })
                    
                    if response.status_code == 200:
                        discovered.append(path)
                        
                        # Check for sensitive files
                        if any(sensitive in path.lower() for sensitive in ['.env', '.git', 'config', 'backup']):
                            self.add_finding('high', 'Sensitive File Exposed',
                                           f'Accessible sensitive path: {path}',
                                           'Restrict access to sensitive files')
                        elif 'admin' in path.lower() or 'login' in path.lower():
                            self.add_finding('medium', 'Admin/Login Endpoint Found',
                                           f'Admin/login endpoint accessible: {path}',
                                           'Ensure proper authentication and rate limiting')
            except:
                continue
        
        if discovered:
            self.add_finding('medium', 'Endpoint Discovery',
                           f'Found {len(discovered)} accessible endpoints',
                           'Review and secure discovered endpoints')
        
        return {
            'discovered': discovered,
            'accessible': accessible,
            'total_tested': len(COMMON_PATHS[:20])
        }
    
    def fingerprint_technology(self) -> Dict[str, Any]:
        """Fingerprint technology stack."""
        try:
            response = self.session.get(self.url, timeout=self.timeout)
            content = response.text.lower()
            headers = {k.lower(): v.lower() for k, v in response.headers.items()}
            
            detected = []
            indicators_found = {}
            
            for tech, indicators in TECH_INDICATORS.items():
                for indicator in indicators:
                    if indicator.lower() in content or indicator.lower() in str(headers):
                        if tech not in detected:
                            detected.append(tech)
                        if tech not in indicators_found:
                            indicators_found[tech] = []
                        indicators_found[tech].append(indicator)
            
            # Check for version disclosure
            version_patterns = [
                r'version["\']?\s*[:=]\s*["\']?([0-9.]+)',
                r'v(\d+\.\d+\.\d+)',
                r'(\d+\.\d+\.\d+)'
            ]
            
            versions = []
            for pattern in version_patterns:
                matches = re.findall(pattern, content[:5000])  # Limit search
                versions.extend(matches[:3])  # Limit results
            
            if versions:
                self.add_finding('low', 'Version Information Disclosure',
                               f'Possible version info found: {versions[:3]}',
                               'Remove version information from responses')
            
            return {
                'technologies': detected,
                'indicators': indicators_found,
                'versions': versions[:5] if versions else []
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_cors(self) -> Dict[str, Any]:
        """Check for CORS misconfiguration."""
        try:
            # Test CORS with origin header
            test_origin = 'https://evil.com'
            response = self.session.options(
                self.url,
                headers={'Origin': test_origin, 'Access-Control-Request-Method': 'GET'},
                timeout=self.timeout
            )
            
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Credentials': response.headers.get('Access-Control-Allow-Credentials'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
            }
            
            issues = []
            
            if cors_headers['Access-Control-Allow-Origin'] == '*':
                if cors_headers['Access-Control-Allow-Credentials'] == 'true':
                    issues.append('CORS misconfiguration: Allows credentials with wildcard origin')
                    self.add_finding('critical', 'CORS Misconfiguration',
                                   'CORS allows credentials with wildcard origin - critical security issue',
                                   'Remove wildcard origin or disable credentials')
                else:
                    issues.append('CORS allows all origins')
                    self.add_finding('medium', 'Permissive CORS Policy',
                                   'CORS allows all origins (*)',
                                   'Restrict CORS to specific trusted origins')
            
            elif cors_headers['Access-Control-Allow-Origin'] == test_origin:
                issues.append('CORS reflects arbitrary origin')
                self.add_finding('high', 'CORS Origin Reflection',
                               'CORS reflects arbitrary origin - vulnerable to CSRF',
                               'Implement proper origin validation')
            
            return {
                'headers': cors_headers,
                'issues': issues,
                'vulnerable': len(issues) > 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_http_methods(self) -> Dict[str, Any]:
        """Test allowed HTTP methods."""
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD', 'TRACE']
        allowed = []
        
        for method in methods:
            try:
                response = self.session.request(method, self.url, timeout=self.timeout, allow_redirects=False)
                if response.status_code not in [405, 501]:
                    allowed.append(method)
            except:
                continue
        
        issues = []
        if 'TRACE' in allowed:
            issues.append('TRACE method enabled - vulnerable to XST attacks')
            self.add_finding('medium', 'TRACE Method Enabled',
                           'TRACE method is enabled - can be used for XST attacks',
                           'Disable TRACE method')
        
        if 'PUT' in allowed or 'DELETE' in allowed:
            issues.append(f'Dangerous methods enabled: {", ".join([m for m in ["PUT", "DELETE"] if m in allowed])}')
            self.add_finding('high', 'Dangerous HTTP Methods Enabled',
                           f'PUT/DELETE methods enabled: {", ".join([m for m in ["PUT", "DELETE"] if m in allowed])}',
                           'Disable PUT/DELETE methods if not needed')
        
        return {
            'allowed': allowed,
            'issues': issues
        }
    
    def check_information_disclosure(self) -> Dict[str, Any]:
        """Check for information disclosure vulnerabilities."""
        try:
            response = self.session.get(self.url, timeout=self.timeout)
            content = response.text
            headers = response.headers
            
            disclosures = []
            
            # Check for error messages
            error_patterns = [
                r'stack trace',
                r'traceback',
                r'fatal error',
                r'warning:',
                r'notice:',
                r'error in',
                r'exception',
                r'at \w+\.\w+\([^)]+\)'
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    disclosures.append('Error messages in response')
                    self.add_finding('medium', 'Error Message Disclosure',
                                   'Error messages or stack traces found in response',
                                   'Configure proper error handling')
                    break
            
            # Check for debug information
            if 'debug' in content.lower() or 'DEBUG' in headers:
                disclosures.append('Debug information present')
                self.add_finding('medium', 'Debug Information Disclosure',
                               'Debug information found in response',
                               'Disable debug mode in production')
            
            # Check for comments with sensitive info
            comment_patterns = [
                r'<!--.*(password|api[_-]?key|secret|token).*-->',
                r'//.*(password|api[_-]?key|secret|token)',
                r'/\*.*(password|api[_-]?key|secret|token).*\*/'
            ]
            
            for pattern in comment_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    disclosures.append('Sensitive information in comments')
                    self.add_finding('high', 'Sensitive Info in Comments',
                                   'Potential sensitive information found in code comments',
                                   'Remove sensitive information from source code')
                    break
            
            return {
                'disclosures': disclosures,
                'found': len(disclosures) > 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def add_finding(self, severity: str, title: str, description: str, recommendation: str):
        """Add a security finding."""
        # Map critical to high for Heimdall compatibility
        mapped_severity = 'high' if severity == 'critical' else severity
        self.findings.append({
            'severity': mapped_severity,
            'title': title,
            'description': description,
            'recommendation': recommendation
        })

