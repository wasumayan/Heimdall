'use client'

import { useState } from 'react'
import { Shield, Code, Globe, AlertCircle, CheckCircle, Download, ExternalLink } from 'lucide-react'
import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Finding {
  id: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  title: string
  description: string
  location?: string
  code_snippet?: string
  recommendation: string
  fix_suggestion?: string
}

interface ScanResult {
  scan_id: string
  status: string
  findings: Finding[]
  summary: {
    total_findings: number
    critical?: number
    high: number
    medium: number
    low: number
  }
  timestamp: string
}

export default function Home() {
  const [mode, setMode] = useState<'select' | 'scan' | 'audit'>('select')
  const [url, setUrl] = useState('')
  const [githubRepo, setGithubRepo] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleScanWebsite = async () => {
    if (!url) {
      setError('Please enter a URL')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/scan-url`, {
        url: url,
      })
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Scan failed')
    } finally {
      setLoading(false)
    }
  }

  const handleAuditCodebase = async () => {
    if (!githubRepo) {
      setError('Please enter a GitHub repository URL')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/audit-codebase`, {
        github_repo: githubRepo,
      })
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Audit failed')
    } finally {
      setLoading(false)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 border-red-500 text-red-900'
      case 'high':
        return 'bg-orange-100 border-orange-500 text-orange-900'
      case 'medium':
        return 'bg-yellow-100 border-yellow-500 text-yellow-900'
      case 'low':
        return 'bg-green-100 border-green-500 text-green-900'
      default:
        return 'bg-gray-100 border-gray-500 text-gray-900'
    }
  }

  const downloadReport = async (format: 'html' | 'pdf' = 'html') => {
    if (!result) return

    try {
      const response = await axios.get(
        `${API_BASE_URL}/result/${result.scan_id}/report?format=${format}`,
        { responseType: 'blob' }
      )
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `heimdall-report-${result.scan_id}.${format}`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (err) {
      console.error('Failed to download report:', err)
    }
  }

  if (result) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">Scan Results</h1>
                <p className="text-gray-600">Scan ID: {result.scan_id}</p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={downloadReport}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                >
                  <Download size={18} />
                  Download Report
                </button>
                <button
                  onClick={() => {
                    setResult(null)
                    setMode('select')
                    setUrl('')
                    setGithubRepo('')
                  }}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
                >
                  New Scan
                </button>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-4 mb-8">
              <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-red-600">{result.summary.critical || 0}</div>
                <div className="text-sm text-red-700">Critical</div>
              </div>
              <div className="bg-orange-50 border-2 border-orange-200 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-orange-600">{result.summary.high}</div>
                <div className="text-sm text-orange-700">High</div>
              </div>
              <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-yellow-600">{result.summary.medium}</div>
                <div className="text-sm text-yellow-700">Medium</div>
              </div>
              <div className="bg-green-50 border-2 border-green-200 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-green-600">{result.summary.low}</div>
                <div className="text-sm text-green-700">Low</div>
              </div>
            </div>

            <div className="space-y-4">
              {result.findings.map((finding) => (
                <div
                  key={finding.id}
                  className={`border-l-4 rounded-lg p-6 ${getSeverityColor(finding.severity)}`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-xl font-semibold mb-1">{finding.title}</h3>
                      <span className="inline-block px-3 py-1 bg-white rounded-full text-sm font-medium capitalize">
                        {finding.severity}
                      </span>
                    </div>
                  </div>
                  <p className="text-gray-800 mb-3">{finding.description}</p>
                  {finding.location && (
                    <p className="text-sm mb-2">
                      <strong>Location:</strong> <code className="bg-white px-2 py-1 rounded">{finding.location}</code>
                    </p>
                  )}
                  {finding.code_snippet && (
                    <div className="bg-white rounded p-3 mb-3 font-mono text-sm overflow-x-auto">
                      <pre>{finding.code_snippet}</pre>
                    </div>
                  )}
                  <div className="bg-white rounded p-4 mb-3">
                    <p className="font-semibold mb-1">💡 Recommendation:</p>
                    <p className="text-gray-800">{finding.recommendation}</p>
                  </div>
                  {finding.fix_suggestion && (
                    <div className="bg-white rounded p-4">
                      <p className="font-semibold mb-1">🔧 Fix Suggestion:</p>
                      <p className="text-gray-800 font-mono text-sm">{finding.fix_suggestion}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-8">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield size={48} className="text-blue-600" />
            <h1 className="text-5xl font-bold text-gray-900">Heimdall</h1>
          </div>
          <p className="text-xl text-gray-600 mb-2">Cybersecurity Co-Pilot & Vulnerability Scanner</p>
          <p className="text-gray-500">Idiot-proof security scanning for small businesses, startups, and solo devs</p>
        </div>

        {mode === 'select' && (
          <div className="grid md:grid-cols-2 gap-6">
            <button
              onClick={() => setMode('scan')}
              className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-xl transition-shadow group"
            >
              <div className="flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4 group-hover:bg-blue-200 transition-colors">
                <Globe size={32} className="text-blue-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Scan Website</h2>
              <p className="text-gray-600 mb-4">
                External scanning mode. Analyze surface vulnerabilities, endpoints, and web risks.
              </p>
              <div className="flex items-center text-blue-600 font-medium">
                Start Scan <ExternalLink size={18} className="ml-2" />
              </div>
            </button>

            <button
              onClick={() => setMode('audit')}
              className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-xl transition-shadow group"
            >
              <div className="flex items-center justify-center w-16 h-16 bg-indigo-100 rounded-full mb-4 group-hover:bg-indigo-200 transition-colors">
                <Code size={32} className="text-indigo-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Audit Codebase</h2>
              <p className="text-gray-600 mb-4">
                Deep analysis mode. Knowledge graph-based codebase audit with actionable findings.
              </p>
              <div className="flex items-center text-indigo-600 font-medium">
                Start Audit <ExternalLink size={18} className="ml-2" />
              </div>
            </button>
          </div>
        )}

        {mode === 'scan' && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <button
              onClick={() => setMode('select')}
              className="text-gray-600 hover:text-gray-900 mb-6 flex items-center gap-2"
            >
              ← Back
            </button>
            <div className="flex items-center gap-3 mb-6">
              <Globe size={32} className="text-blue-600" />
              <h2 className="text-3xl font-bold text-gray-900">Scan Website</h2>
            </div>
            <p className="text-gray-600 mb-6">
              Enter a URL to scan for external vulnerabilities, endpoint exposures, and web risks.
            </p>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Website URL</label>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-2 text-red-800">
                  <AlertCircle size={20} />
                  <span>{error}</span>
                </div>
              )}
              <button
                onClick={handleScanWebsite}
                disabled={loading}
                className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Scanning...
                  </>
                ) : (
                  <>
                    <Shield size={20} />
                    Scan Now
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {mode === 'audit' && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <button
              onClick={() => setMode('select')}
              className="text-gray-600 hover:text-gray-900 mb-6 flex items-center gap-2"
            >
              ← Back
            </button>
            <div className="flex items-center gap-3 mb-6">
              <Code size={32} className="text-indigo-600" />
              <h2 className="text-3xl font-bold text-gray-900">Audit Codebase</h2>
            </div>
            <p className="text-gray-600 mb-6">
              Connect a GitHub repository for deep codebase analysis using knowledge graphs and evidence tracking.
            </p>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">GitHub Repository URL</label>
                <input
                  type="url"
                  value={githubRepo}
                  onChange={(e) => setGithubRepo(e.target.value)}
                  placeholder="https://github.com/username/repo"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                />
                <p className="text-sm text-gray-500 mt-2">
                  💡 Tip: Make sure the repository is public or you have access permissions
                </p>
              </div>
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-2 text-red-800">
                  <AlertCircle size={20} />
                  <span>{error}</span>
                </div>
              )}
              <button
                onClick={handleAuditCodebase}
                disabled={loading}
                className="w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Auditing...
                  </>
                ) : (
                  <>
                    <Shield size={20} />
                    Start Audit
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}

