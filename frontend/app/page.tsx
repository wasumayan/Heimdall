'use client'

import React, { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { Shield, Code, Globe, AlertCircle, CheckCircle, Download, ExternalLink, Filter, TrendingUp, Clock, Layers, Target, BarChart3, FileCode, Zap, Settings, Network, X, Activity, Radio, Send, PlayCircle, Square } from 'lucide-react'
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
  // Hound-specific fields - preserving ALL fields from Hound's output
  confidence?: number
  status?: 'proposed' | 'investigating' | 'confirmed' | 'rejected' | 'supported' | 'refuted'
  vulnerability_type?: string
  evidence?: any[]  // Can be list of Evidence objects or strings
  node_refs?: string[]
  reasoning?: string
  created_at?: string
  junior_model?: string
  senior_model?: string
  created_by?: string
  session_id?: string
  visibility?: string
  properties?: Record<string, any>
  root_cause?: string
  attack_vector?: string
  [key: string]: any  // Allow any additional fields from Hound
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
  // Hound-specific fields
  coverage?: {
    nodes?: { visited: number; total: number; percent: number }
    cards?: { visited: number; total: number; percent: number }
  }
  session?: string
  project_name?: string
  diagnostic?: {
    message?: string
    hypotheses_found?: number
    suggestion?: string
  }
  telemetry?: {
    sse_url?: string
    control_url?: string
    token?: string
    port?: number
  }
}

export default function Home() {
  const [mode, setMode] = useState<'select' | 'scan' | 'audit'>('select')
  const [url, setUrl] = useState('')
  const [githubRepo, setGithubRepo] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [showSignIn, setShowSignIn] = useState(false)
  
  // Hound audit options - All CLI options exposed
  const [auditMode, setAuditMode] = useState<'sweep' | 'intuition'>('sweep')
  const [buildGraphs, setBuildGraphs] = useState(true)
  const [timeLimit, setTimeLimit] = useState<string>('')
  const [iterations, setIterations] = useState<string>('20')
  const [planN, setPlanN] = useState<string>('5')
  const [debugMode, setDebugMode] = useState(false)
  const [mission, setMission] = useState<string>('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // Graph building options
  const [graphMaxIterations, setGraphMaxIterations] = useState<string>('3')
  const [graphMaxGraphs, setGraphMaxGraphs] = useState<string>('5')
  const [autoGenerateWhitelist, setAutoGenerateWhitelist] = useState(true)  // Default: auto-generate
  const [whitelistLocBudget, setWhitelistLocBudget] = useState<string>('50000')
  const [graphFileFilter, setGraphFileFilter] = useState<string>('')  // Manual override
  const [graphFocusAreas, setGraphFocusAreas] = useState<string>('')
  const [graphAuto, setGraphAuto] = useState(true)  // Default to true to ensure SystemArchitecture is created
  const [graphInitOnly, setGraphInitOnly] = useState(false)
  const [graphRefineExisting, setGraphRefineExisting] = useState(true)
  const [graphVisualize, setGraphVisualize] = useState(true)
  
  // Model overrides
  const [scoutPlatform, setScoutPlatform] = useState<string>('')
  const [scoutModel, setScoutModel] = useState<string>('')
  const [strategistPlatform, setStrategistPlatform] = useState<string>('')
  const [strategistModel, setStrategistModel] = useState<string>('')
  
  // Session options
  const [sessionId, setSessionId] = useState<string>('')
  const [newSession, setNewSession] = useState(true)
  const [sessionPrivateHypotheses, setSessionPrivateHypotheses] = useState(false)
  
  // Advanced options
  const [strategistTwoPass, setStrategistTwoPass] = useState(false)
  const [enableTelemetry, setEnableTelemetry] = useState(true)
  
  // Finalize options
  const [finalizeThreshold, setFinalizeThreshold] = useState<string>('0.5')
  const [finalizeIncludeBelowThreshold, setFinalizeIncludeBelowThreshold] = useState(false)
  const [finalizePlatform, setFinalizePlatform] = useState<string>('')
  const [finalizeModel, setFinalizeModel] = useState<string>('')
  
  // Filtering and display options
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'severity' | 'confidence' | 'status'>('severity')
  const [expandedFinding, setExpandedFinding] = useState<string | null>(null)
  
  // Graph visualization
  const [showGraphView, setShowGraphView] = useState(false)
  const [availableGraphs, setAvailableGraphs] = useState<any[]>([])
  const [selectedGraph, setSelectedGraph] = useState<string | null>(null)
  const [graphData, setGraphData] = useState<any>(null)
  const [loadingGraph, setLoadingGraph] = useState(false)
  
  // Telemetry
  const [showTelemetry, setShowTelemetry] = useState(false)
  const [telemetryEvents, setTelemetryEvents] = useState<any[]>([])
  const [telemetryConnected, setTelemetryConnected] = useState(false)
  const [currentActivity, setCurrentActivity] = useState<string | null>(null)
  const [steeringText, setSteeringText] = useState('')
  const [sendingSteering, setSendingSteering] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  
  // Whitelist configuration modal
  const [showWhitelistModal, setShowWhitelistModal] = useState(false)

  // Whitelist Configuration Modal Component
  const WhitelistModal = () => {
    if (!showWhitelistModal) return null
    
    return (
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" 
        onClick={() => setShowWhitelistModal(false)}
      >
        <div 
          className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto" 
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <FileCode size={24} />
              File Whitelist Configuration
            </h2>
            <button
              onClick={() => setShowWhitelistModal(false)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X size={24} className="text-gray-600" />
            </button>
          </div>
          
          <div className="p-6 space-y-6">
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-1">
                    Auto-generate File Whitelist
                  </label>
                  <p className="text-xs text-gray-600">
                    Automatically select files within LOC budget (recommended for best results)
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={autoGenerateWhitelist}
                  onChange={(e) => setAutoGenerateWhitelist(e.target.checked)}
                  className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                />
              </div>
              
              {autoGenerateWhitelist && (
                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    LOC Budget (lines of code)
                  </label>
                  <input
                    type="number"
                    value={whitelistLocBudget}
                    onChange={(e) => setWhitelistLocBudget(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                    min="10000"
                    max="200000"
                    step="10000"
                  />
                  <p className="text-xs text-gray-500 mt-2">
                    Maximum lines of code to include. Larger budgets = more files analyzed. 
                    Default: 50,000 LOC (good for most projects)
                  </p>
                  <div className="mt-3 p-3 bg-white rounded border border-gray-200">
                    <p className="text-xs font-medium text-gray-700 mb-1">Recommended budgets:</p>
                    <ul className="text-xs text-gray-600 space-y-1">
                      <li>• Small projects (&lt;20k LOC): 20,000</li>
                      <li>• Medium projects (20-80k LOC): 50,000</li>
                      <li>• Large projects (&gt;80k LOC): 100,000+</li>
                    </ul>
                  </div>
                </div>
              )}
              
              {!autoGenerateWhitelist && (
                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Manual File Whitelist
                  </label>
                  <textarea
                    value={graphFileFilter}
                    onChange={(e) => setGraphFileFilter(e.target.value)}
                    placeholder="Enter comma-separated file paths relative to repo root, e.g.:&#10;src/main.py,src/utils.py,lib/helpers.js,config/settings.json"
                    rows={8}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-2">
                    Specify exact files to analyze. One file per line or comma-separated. 
                    Paths should be relative to the repository root.
                  </p>
                </div>
              )}
            </div>
            
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowWhitelistModal(false)}
                className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 font-medium"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowWhitelistModal(false)}
                className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium"
              >
                Save Configuration
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

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

  // Telemetry connection management
  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
  }, [])

  const connectTelemetry = async (projectName: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    // Check if telemetry server is available
    try {
      const statusResponse = await axios.get(`${API_BASE_URL}/telemetry/${projectName}/status`)
      if (!statusResponse.data.available) {
        setTelemetryConnected(false)
        setError('Telemetry server not available. The audit may have completed.')
        return
      }
    } catch (err) {
      console.error('Failed to check telemetry status:', err)
      setTelemetryConnected(false)
      setError('Telemetry server not available. The audit may have completed.')
      return
    }

    const eventSource = new EventSource(`${API_BASE_URL}/telemetry/${projectName}/events`)
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      setTelemetryConnected(true)
      setError(null)
      console.log('Telemetry connected')
    }

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        // Add to events list (keep last 100)
        setTelemetryEvents((prev) => {
          const newEvents = [...prev, { ...data, timestamp: new Date().toISOString() }]
          return newEvents.slice(-100)
        })

        // Update current activity based on event type
        if (data.type === 'decision' || data.type === 'executing') {
          const action = data.action || data.type
          const file = data.parameters?.file_path || ''
          setCurrentActivity(`${action}${file ? `: ${file}` : ''}`)
        } else if (data.type === 'status') {
          setCurrentActivity(data.message || 'Processing...')
        } else if (data.type === 'result') {
          setCurrentActivity(`Completed: ${data.action || 'investigation'}`)
        }
      } catch (e) {
        console.error('Failed to parse telemetry event:', e)
      }
    }

    eventSource.onerror = (error) => {
      console.error('Telemetry connection error:', error)
      setTelemetryConnected(false)
      // Only try to reconnect if we're still showing telemetry
      // and the error isn't a 404 (server not found)
      if (showTelemetry && projectName && eventSource.readyState === EventSource.CONNECTING) {
        setTimeout(() => {
          if (showTelemetry && projectName) {
            connectTelemetry(projectName)
          }
        }, 3000)
      }
    }
  }

  const disconnectTelemetry = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
      setTelemetryConnected(false)
      setTelemetryEvents([])
      setCurrentActivity(null)
    }
  }

  const sendSteering = async (projectName: string, text: string) => {
    if (!text.trim()) return

    setSendingSteering(true)
    try {
      await axios.post(`${API_BASE_URL}/telemetry/${projectName}/steer`, {
        text: text.trim(),
        ts: Date.now() / 1000
      })
      setSteeringText('')
      // Add steering event to feed
      setTelemetryEvents((prev) => [
        ...prev,
        {
          type: 'steer',
          text: text.trim(),
          timestamp: new Date().toISOString(),
          actor: 'user'
        }
      ])
    } catch (err: any) {
      console.error('Failed to send steering:', err)
      setError(err.response?.data?.error || 'Failed to send steering command')
    } finally {
      setSendingSteering(false)
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
      // Build complete request with all CLI options
      const auditRequest: any = {
        github_repo: githubRepo,
        // Graph options
        build_graphs: buildGraphs,
        graph_max_iterations: parseInt(graphMaxIterations) || 3,
        graph_max_graphs: parseInt(graphMaxGraphs) || 5,
        auto_generate_whitelist: autoGenerateWhitelist,
        whitelist_loc_budget: parseInt(whitelistLocBudget) || 50000,
        graph_file_filter: graphFileFilter || undefined,  // Manual override (only used if auto_generate_whitelist is false)
        graph_focus_areas: graphFocusAreas || undefined,
        graph_auto: graphAuto,
        graph_init_only: graphInitOnly,
        graph_refine_existing: graphRefineExisting,
        graph_visualize: graphVisualize,
        // Audit options
        audit_mode: auditMode,
        iterations: parseInt(iterations) || 20,
        plan_n: parseInt(planN) || 5,
        time_limit_minutes: timeLimit ? parseInt(timeLimit) : undefined,
        debug: debugMode,
        mission: mission || undefined,
        // Model overrides
        scout_platform: scoutPlatform || undefined,
        scout_model: scoutModel || undefined,
        strategist_platform: strategistPlatform || undefined,
        strategist_model: strategistModel || undefined,
        // Session options
        session_id: sessionId || undefined,
        new_session: newSession,
        session_private_hypotheses: sessionPrivateHypotheses,
        // Advanced
        strategist_two_pass: strategistTwoPass,
        telemetry: enableTelemetry,
        // Finalize options
        finalize_threshold: parseFloat(finalizeThreshold) || 0.5,
        finalize_include_below_threshold: finalizeIncludeBelowThreshold,
        finalize_platform: finalizePlatform || undefined,
        finalize_model: finalizeModel || undefined,
      }

      const response = await axios.post(`${API_BASE_URL}/audit-codebase`, auditRequest)
      setResult(response.data)
      
      // If telemetry is enabled and available, automatically connect
      if (enableTelemetry && response.data.telemetry && response.data.project_name) {
        // Small delay to ensure telemetry server is ready
        setTimeout(() => {
          connectTelemetry(response.data.project_name)
          setShowTelemetry(true)
        }, 1000)
      }
    } catch (err: any) {
      const errorDetail = err.response?.data?.detail || err.message || 'Audit failed'
      console.error('Audit error:', err)
      setError(errorDetail)
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'confirmed':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'investigating':
        return 'bg-blue-100 text-blue-800 border-blue-300'
      case 'proposed':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'rejected':
        return 'bg-gray-100 text-gray-800 border-gray-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 font-bold'
    if (confidence >= 0.6) return 'text-yellow-600 font-semibold'
    return 'text-red-600'
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

  const loadGraphData = async (projectName: string, graphName: string) => {
    try {
      setLoadingGraph(true)
      const response = await axios.get(`${API_BASE_URL}/project/${projectName}/graph/${graphName}`)
      setGraphData(response.data)
    } catch (err) {
      console.error('Failed to load graph data:', err)
    } finally {
      setLoadingGraph(false)
    }
  }

  const handleSignIn = (email: string, password: string) => {
    if (email && password) {
      setIsAuthenticated(true)
      setShowSignIn(false)
    }
  }

  // Filter and sort findings
  const filteredFindings = result?.findings
    ?.filter((f) => {
      if (statusFilter !== 'all' && f.status !== statusFilter) return false
      if (severityFilter !== 'all' && f.severity !== severityFilter) return false
      return true
    })
    .sort((a, b) => {
      if (sortBy === 'severity') {
        const order = { critical: 4, high: 3, medium: 2, low: 1 }
        return (order[b.severity] || 0) - (order[a.severity] || 0)
      }
      if (sortBy === 'confidence') {
        return (b.confidence || 0) - (a.confidence || 0)
      }
      if (sortBy === 'status') {
        const order = { confirmed: 3, investigating: 2, proposed: 1, rejected: 0 }
        return (order[b.status as keyof typeof order] || 0) - (order[a.status as keyof typeof order] || 0)
      }
      return 0
    }) || []

  // Graph Visualization Component
  const GraphVisualization = () => {
    if (!showGraphView || !result?.project_name) return null

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-7xl h-[90vh] flex flex-col">
          <div className="flex items-center justify-between p-6 border-b">
            <div className="flex items-center gap-3">
              <Network size={24} className="text-indigo-600" />
              <h2 className="text-2xl font-bold text-gray-900">Knowledge Graph Visualization</h2>
            </div>
            <button
              onClick={() => {
                setShowGraphView(false)
                setGraphData(null)
                setSelectedGraph(null)
              }}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X size={24} className="text-gray-600" />
            </button>
          </div>

          <div className="flex-1 flex overflow-hidden">
            {/* Sidebar */}
            <div className="w-64 bg-gray-50 border-r p-4 overflow-y-auto">
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">Select Graph</label>
                <select
                  value={selectedGraph || ''}
                  onChange={async (e) => {
                    setSelectedGraph(e.target.value)
                    await loadGraphData(result.project_name!, e.target.value)
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                >
                  {availableGraphs.map((g) => (
                    <option key={g.name} value={g.name}>
                      {g.name} ({g.nodes} nodes, {g.edges} edges)
                    </option>
                  ))}
                </select>
              </div>

              {graphData && (
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-3 border">
                    <h3 className="font-semibold text-sm mb-2">Graph Statistics</h3>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Nodes:</span>
                        <span className="font-semibold">{graphData.graph?.nodes?.length || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Edges:</span>
                        <span className="font-semibold">{graphData.graph?.edges?.length || 0}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Graph Canvas */}
            <div className="flex-1 relative bg-gray-900">
              {loadingGraph ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-white">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                    <p>Loading graph...</p>
                  </div>
                </div>
              ) : graphData ? (
                <GraphCanvas graphData={graphData} />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                  <div className="text-center">
                    <Network size={48} className="mx-auto mb-4 opacity-50" />
                    <p>Select a graph to visualize</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Graph Canvas Component using D3.js
  const GraphCanvas = ({ graphData }: { graphData: any }) => {
    const canvasRef = React.useRef<HTMLDivElement>(null)
    const [selectedNode, setSelectedNode] = useState<any>(null)

    React.useEffect(() => {
      if (!canvasRef.current || !graphData?.graph) return

      // Clear previous graph
      const svg = d3.select(canvasRef.current).select('svg')
      svg.remove()

      const width = canvasRef.current.clientWidth
      const height = canvasRef.current.clientHeight

      const svgElement = d3
        .select(canvasRef.current)
        .append('svg')
        .attr('width', width)
        .attr('height', height)

      const g = svgElement.append('g')

      // Zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.1, 4]).on('zoom', (event) => {
        g.attr('transform', event.transform)
      })
      svgElement.call(zoom)

      const nodes = graphData.graph.nodes || []
      const links = (graphData.graph.edges || []).map((e: any) => ({
        source: e.source_id,
        target: e.target_id,
        type: e.type,
        label: e.label,
      }))

      // Create node ID map
      const nodeMap = new Map(nodes.map((n: any) => [n.id, n]))

      // Filter valid links
      const validLinks = links.filter((l: any) => {
        const source = typeof l.source === 'string' ? nodeMap.get(l.source) : l.source
        const target = typeof l.target === 'string' ? nodeMap.get(l.target) : l.target
        return source && target
      })

      // Color scheme
      const typeColors: Record<string, string> = {
        contract: '#30a14e',
        interface: '#58a6ff',
        library: '#a371f7',
        function: '#56d364',
        storage: '#f9826c',
        event: '#79c0ff',
        modifier: '#bc8cff',
        role: '#ffa657',
      }

      const edgeColors: Record<string, string> = {
        calls: '#30a14e',
        contains: '#58a6ff',
        depends_on: '#f9826c',
        references: '#a371f7',
        uses: '#79c0ff',
        implements: '#56d364',
      }

      // Force simulation
      const simulation = d3
        .forceSimulation(nodes as any)
        .force(
          'link',
          d3
            .forceLink(validLinks)
            .id((d: any) => d.id)
            .distance(150)
        )
        .force('charge', d3.forceManyBody().strength(-800))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(50))

      // Create links
      const link = g
        .append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(validLinks)
        .enter()
        .append('line')
        .attr('stroke', (d: any) => edgeColors[d.type] || '#8b949e')
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.6)

      // Create nodes
      const node = g
        .append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', 8)
        .attr('fill', (d: any) => typeColors[d.type?.toLowerCase()] || '#8b949e')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('click', (event, d: any) => {
          setSelectedNode(d)
        })
        .call(
          d3
            .drag<SVGCircleElement, any>()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended)
        )

      // Node labels
      const labels = g
        .append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(nodes)
        .enter()
        .append('text')
        .text((d: any) => d.label || d.id)
        .attr('font-size', '12px')
        .attr('fill', '#e6edf3')
        .attr('dx', 12)
        .attr('dy', 4)
        .style('pointer-events', 'none')

      // Tooltip
      const tooltip = d3
        .select('body')
        .append('div')
        .attr('class', 'graph-tooltip')
        .style('opacity', 0)
        .style('position', 'absolute')
        .style('background', 'rgba(22, 27, 34, 0.95)')
        .style('color', '#e6edf3')
        .style('padding', '12px')
        .style('border-radius', '6px')
        .style('pointer-events', 'none')
        .style('font-size', '12px')
        .style('max-width', '300px')
        .style('border', '1px solid #30363d')
        .style('box-shadow', '0 4px 12px rgba(0,0,0,0.3)')
        .style('z-index', '10000')

      node
        .on('mouseover', function (event, d: any) {
          tooltip.transition().duration(200).style('opacity', 0.9)
          tooltip
            .html(
              `<strong>${d.label || d.id}</strong><br/>Type: ${d.type}<br/>${d.description ? `Description: ${d.description.substring(0, 100)}...` : ''}`
            )
            .style('left', event.pageX + 10 + 'px')
            .style('top', event.pageY - 28 + 'px')
        })
        .on('mouseout', function () {
          tooltip.transition().duration(500).style('opacity', 0)
        })

      // Update positions
      simulation.on('tick', () => {
        link
          .attr('x1', (d: any) => {
            const source = typeof d.source === 'string' ? nodeMap.get(d.source) : d.source
            return source?.x || 0
          })
          .attr('y1', (d: any) => {
            const source = typeof d.source === 'string' ? nodeMap.get(d.source) : d.source
            return source?.y || 0
          })
          .attr('x2', (d: any) => {
            const target = typeof d.target === 'string' ? nodeMap.get(d.target) : d.target
            return target?.x || 0
          })
          .attr('y2', (d: any) => {
            const target = typeof d.target === 'string' ? nodeMap.get(d.target) : d.target
            return target?.y || 0
          })

        node.attr('cx', (d: any) => d.x || 0).attr('cy', (d: any) => d.y || 0)

        labels.attr('x', (d: any) => d.x || 0).attr('y', (d: any) => d.y || 0)
      })

      function dragstarted(event: any, d: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      }

      function dragged(event: any, d: any) {
        d.fx = event.x
        d.fy = event.y
      }

      function dragended(event: any, d: any) {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null
        d.fy = null
      }

      // Cleanup
      return () => {
        svgElement.remove()
        tooltip.remove()
      }
    }, [graphData])

    return (
      <>
        <div ref={canvasRef} className="w-full h-full"></div>
        {selectedNode && (
          <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-xl p-4 max-w-md border">
            <div className="flex items-start justify-between mb-2">
              <h3 className="font-bold text-lg">{selectedNode.label || selectedNode.id}</h3>
              <button onClick={() => setSelectedNode(null)} className="text-gray-400 hover:text-gray-600">
                <X size={18} />
              </button>
            </div>
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-semibold">Type:</span> {selectedNode.type}
              </div>
              {selectedNode.description && (
                <div>
                  <span className="font-semibold">Description:</span>
                  <p className="text-gray-600 mt-1">{selectedNode.description}</p>
                </div>
              )}
              {selectedNode.observations && selectedNode.observations.length > 0 && (
                <div>
                  <span className="font-semibold">Observations:</span>
                  <ul className="list-disc list-inside text-gray-600 mt-1">
                    {selectedNode.observations.slice(0, 3).map((obs: any, i: number) => (
                      <li key={i}>{typeof obs === 'object' ? obs.description || obs.content : String(obs)}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </>
    )
  }

  // Telemetry Dashboard Component
  const TelemetryDashboard = () => {
    if (!showTelemetry || !result?.project_name) return null

    return (
      <div className="fixed bottom-0 right-0 w-96 h-[600px] bg-white rounded-t-2xl shadow-2xl border-t-2 border-purple-500 z-40 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-purple-50 to-indigo-50">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${telemetryConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <h3 className="font-bold text-gray-900">Live Telemetry</h3>
            {currentActivity && (
              <span className="text-xs text-gray-600 bg-white px-2 py-1 rounded">Active</span>
            )}
          </div>
          <button
            onClick={() => {
              setShowTelemetry(false)
              disconnectTelemetry()
            }}
            className="p-1 hover:bg-gray-200 rounded"
          >
            <X size={18} className="text-gray-600" />
          </button>
        </div>

        {/* Current Activity */}
        {currentActivity && (
          <div className="p-3 bg-blue-50 border-b">
            <div className="flex items-center gap-2 text-sm">
              <Radio size={14} className="text-blue-600 animate-pulse" />
              <span className="font-semibold text-blue-900">Now:</span>
              <span className="text-blue-700">{currentActivity}</span>
            </div>
          </div>
        )}

        {/* Events Feed */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {telemetryEvents.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <Activity size={48} className="mx-auto mb-2 opacity-50" />
              <p>Waiting for events...</p>
              {!telemetryConnected && (
                <p className="text-xs text-red-600 mt-2">Not connected. Check if audit is running.</p>
              )}
            </div>
          ) : (
            telemetryEvents.slice().reverse().map((event, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-lg border text-sm ${
                  event.type === 'decision' || event.type === 'executing'
                    ? 'bg-blue-50 border-blue-200'
                    : event.type === 'result'
                    ? 'bg-green-50 border-green-200'
                    : event.type === 'steer'
                    ? 'bg-purple-50 border-purple-200'
                    : event.type === 'status'
                    ? 'bg-gray-50 border-gray-200'
                    : 'bg-white border-gray-200'
                }`}
              >
                <div className="flex items-start justify-between mb-1">
                  <span className="font-semibold text-gray-800 capitalize">{event.type || 'event'}</span>
                  <span className="text-xs text-gray-500">
                    {new Date(event.timestamp || event.ts * 1000).toLocaleTimeString()}
                  </span>
                </div>
                {event.action && (
                  <div className="text-gray-700">
                    <strong>Action:</strong> {event.action}
                  </div>
                )}
                {event.message && (
                  <div className="text-gray-700">{event.message}</div>
                )}
                {event.text && (
                  <div className="text-gray-700">
                    <strong>{event.actor === 'user' ? 'You:' : 'Steering:'}</strong> {event.text}
                  </div>
                )}
                {event.parameters?.file_path && (
                  <div className="text-xs text-gray-600 mt-1">
                    📄 {event.parameters.file_path}
                  </div>
                )}
                {event.reasoning && (
                  <div className="text-xs text-gray-600 mt-1 italic">
                    💭 {event.reasoning.substring(0, 100)}...
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Steering Controls */}
        <div className="p-4 border-t bg-gray-50">
          <form
            onSubmit={(e) => {
              e.preventDefault()
              if (result.project_name && steeringText.trim()) {
                sendSteering(result.project_name, steeringText)
              }
            }}
            className="flex gap-2"
          >
            <input
              type="text"
              value={steeringText}
              onChange={(e) => setSteeringText(e.target.value)}
              placeholder="Steer audit (e.g., 'Investigate reentrancy')"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500"
              disabled={!telemetryConnected || sendingSteering}
            />
            <button
              type="submit"
              disabled={!telemetryConnected || sendingSteering || !steeringText.trim()}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Send size={16} />
            </button>
          </form>
          <p className="text-xs text-gray-500 mt-2">
            💡 Send guidance to steer the audit in real-time
          </p>
        </div>
      </div>
    )
  }

  if (result) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
        <GraphVisualization />
        <TelemetryDashboard />
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">Audit Results</h1>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span>Scan ID: {result.scan_id}</span>
                  {result.project_name && (
                    <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded">Project: {result.project_name}</span>
                  )}
                  {result.session && (
                    <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded">Session: {result.session.substring(0, 12)}...</span>
                  )}
                </div>
              </div>
              <div className="flex gap-2">
                {result.project_name && (
                  <>
                    <button
                      onClick={async () => {
                        setShowGraphView(true)
                        setLoadingGraph(true)
                        try {
                          const graphsResponse = await axios.get(`${API_BASE_URL}/project/${result.project_name}/graphs`)
                          setAvailableGraphs(graphsResponse.data.graphs || [])
                          if (graphsResponse.data.graphs && graphsResponse.data.graphs.length > 0) {
                            setSelectedGraph(graphsResponse.data.graphs[0].name)
                            await loadGraphData(result.project_name!, graphsResponse.data.graphs[0].name)
                          }
                        } catch (err) {
                          console.error('Failed to load graphs:', err)
                        } finally {
                          setLoadingGraph(false)
                        }
                      }}
                      className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2"
                    >
                      <Layers size={18} />
                      View Knowledge Graphs
                    </button>
                    {result.telemetry && (
                      <button
                        onClick={() => {
                          setShowTelemetry(!showTelemetry)
                          if (!showTelemetry && result.project_name) {
                            connectTelemetry(result.project_name)
                          } else {
                            disconnectTelemetry()
                          }
                        }}
                        className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                          showTelemetry
                            ? 'bg-green-600 text-white hover:bg-green-700'
                            : 'bg-purple-600 text-white hover:bg-purple-700'
                        }`}
                      >
                        <Activity size={18} />
                        {showTelemetry ? 'Hide' : 'Show'} Live Telemetry
                      </button>
                    )}
                  </>
                )}
                <button
                  onClick={() => downloadReport('html')}
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
                    setShowGraphView(false)
                  }}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
                >
                  New Scan
                </button>
              </div>
            </div>

            {/* Coverage Statistics */}
            {result.coverage && (result.coverage.nodes || result.coverage.cards) && (
              <div className="mb-6 p-4 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg border border-indigo-200">
                <div className="flex items-center gap-2 mb-3">
                  <BarChart3 size={20} className="text-indigo-600" />
                  <h3 className="font-semibold text-gray-900">Coverage Statistics</h3>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  {result.coverage.nodes && (
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">Graph Nodes</span>
                        <span className="font-semibold">{result.coverage.nodes.visited}/{result.coverage.nodes.total}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-indigo-600 h-2 rounded-full transition-all"
                          style={{ width: `${result.coverage.nodes.percent || 0}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">{result.coverage.nodes.percent?.toFixed(1)}% coverage</span>
                    </div>
                  )}
                  {result.coverage.cards && (
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">Code Cards</span>
                        <span className="font-semibold">{result.coverage.cards.visited}/{result.coverage.cards.total}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all"
                          style={{ width: `${result.coverage.cards.percent || 0}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">{result.coverage.cards.percent?.toFixed(1)}% coverage</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Summary Cards */}
            <div className="grid grid-cols-4 gap-4 mb-6">
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

            {/* Filters and Sorting */}
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 mb-3">
                <Filter size={18} className="text-gray-600" />
                <span className="font-semibold text-gray-700">Filters & Sorting</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Status</label>
                  <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  >
                    <option value="all">All Status</option>
                    <option value="confirmed">Confirmed</option>
                    <option value="investigating">Investigating</option>
                    <option value="proposed">Proposed</option>
                    <option value="rejected">Rejected</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Severity</label>
                  <select
                    value={severityFilter}
                    onChange={(e) => setSeverityFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  >
                    <option value="all">All Severities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Sort By</label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  >
                    <option value="severity">Severity</option>
                    <option value="confidence">Confidence</option>
                    <option value="status">Status</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Findings List */}
            <div className="space-y-4">
              {filteredFindings.length === 0 ? (
                <div className="bg-white rounded-lg p-8 text-center">
                  <AlertCircle size={48} className="mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-700 font-semibold mb-2">
                    {result.findings?.length === 0 
                      ? "No findings were detected in this audit." 
                      : "No findings match your filters."}
                  </p>
                  {result.findings?.length === 0 && (
                    <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-left max-w-2xl mx-auto">
                      <p className="text-sm text-yellow-800 font-semibold mb-2">💡 Troubleshooting Tips:</p>
                      <ul className="text-sm text-yellow-700 space-y-1 list-disc list-inside">
                        <li>Make sure <strong>"Build Knowledge Graphs"</strong> is enabled (recommended for better results)</li>
                        <li>Try increasing <strong>iterations</strong> (default: 20, try 30-50 for large codebases)</li>
                        <li>For deeper analysis, try <strong>Intuition Mode</strong> instead of Sweep Mode</li>
                        <li>Check backend logs for detailed Hound output and any errors</li>
                        <li>Large repositories like Juice Shop may need more time - consider setting a time limit</li>
                      </ul>
                      {result.diagnostic && (
                        <div className="mt-3 pt-3 border-t border-yellow-300">
                          <p className="text-xs text-yellow-600">
                            <strong>Debug Info:</strong> {result.diagnostic.hypotheses_found} hypotheses found in Hound output
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                filteredFindings.map((finding) => (
                  <div
                    key={finding.id}
                    className={`border-l-4 rounded-lg p-6 bg-white shadow-md hover:shadow-lg transition-shadow ${getSeverityColor(finding.severity)}`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-xl font-semibold">{finding.title}</h3>
                          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(finding.status || 'proposed')}`}>
                            {finding.status || 'proposed'}
                          </span>
                          <span className="inline-block px-3 py-1 bg-white rounded-full text-sm font-medium capitalize">
                            {finding.severity}
                          </span>
                          {finding.vulnerability_type && (
                            <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">
                              {finding.vulnerability_type}
                            </span>
                          )}
                        </div>
                        {finding.confidence !== undefined && (
                          <div className="flex items-center gap-2 mb-2">
                            <TrendingUp size={14} className="text-gray-500" />
                            <span className="text-sm text-gray-600">Confidence: </span>
                            <span className={`text-sm ${getConfidenceColor(finding.confidence)}`}>
                              {(finding.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                      </div>
                      <button
                        onClick={() => setExpandedFinding(expandedFinding === finding.id ? null : finding.id)}
                        className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-white rounded"
                      >
                        {expandedFinding === finding.id ? 'Less' : 'More'}
                      </button>
                    </div>
                    
                    <p className="text-gray-800 mb-3">{finding.description}</p>
                    
                    {finding.location && (
                      <p className="text-sm mb-2">
                        <strong>Location:</strong> <code className="bg-white px-2 py-1 rounded">{finding.location}</code>
                      </p>
                    )}

                    {/* Expanded Details - All Hound Fields */}
                    {expandedFinding === finding.id && (
                      <div className="mt-4 space-y-4 pt-4 border-t border-gray-200">
                        {/* Root Cause & Attack Vector */}
                        {(finding.root_cause || finding.attack_vector) && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {finding.root_cause && (
                              <div className="bg-red-50 rounded p-3 border border-red-200">
                                <p className="text-xs font-semibold text-red-900 mb-1 flex items-center gap-1">
                                  <Target size={12} />
                                  Root Cause
                                </p>
                                <p className="text-sm text-red-800">{finding.root_cause}</p>
                              </div>
                            )}
                            {finding.attack_vector && (
                              <div className="bg-orange-50 rounded p-3 border border-orange-200">
                                <p className="text-xs font-semibold text-orange-900 mb-1 flex items-center gap-1">
                                  <AlertCircle size={12} />
                                  Attack Vector
                                </p>
                                <p className="text-sm text-orange-800">{finding.attack_vector}</p>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Reasoning */}
                        {finding.reasoning && (
                          <div className="bg-blue-50 rounded p-3 border border-blue-200">
                            <p className="text-xs font-semibold text-blue-900 mb-1 flex items-center gap-1">
                              <Zap size={12} />
                              Reasoning
                            </p>
                            <p className="text-sm text-blue-800">{finding.reasoning}</p>
                          </div>
                        )}
                        
                        {/* Evidence - Full Structure */}
                        {finding.evidence && finding.evidence.length > 0 && (
                          <div className="bg-purple-50 rounded p-3 border border-purple-200">
                            <p className="text-xs font-semibold text-purple-900 mb-2 flex items-center gap-1">
                              <FileCode size={12} />
                              Evidence ({finding.evidence.length})
                            </p>
                            <div className="space-y-2">
                              {finding.evidence.map((ev: any, idx: number) => {
                                if (typeof ev === 'string') {
                                  return (
                                    <div key={idx} className="bg-white p-2 rounded text-xs text-purple-800 border border-purple-100">
                                      • {ev.substring(0, 300)}{ev.length > 300 ? '...' : ''}
                                    </div>
                                  )
                                } else if (typeof ev === 'object' && ev !== null) {
                                  return (
                                    <div key={idx} className="bg-white p-3 rounded border border-purple-100">
                                      {ev.description && (
                                        <p className="text-xs text-purple-800 mb-1"><strong>Description:</strong> {ev.description}</p>
                                      )}
                                      <div className="flex gap-3 text-xs text-purple-600 mt-1">
                                        {ev.type && <span>Type: <span className="font-mono">{ev.type}</span></span>}
                                        {ev.confidence !== undefined && <span>Confidence: {(ev.confidence * 100).toFixed(0)}%</span>}
                                        {ev.created_at && <span>Created: {new Date(ev.created_at).toLocaleDateString()}</span>}
                                      </div>
                                      {ev.node_refs && ev.node_refs.length > 0 && (
                                        <div className="mt-2 flex flex-wrap gap-1">
                                          {ev.node_refs.map((ref: string, i: number) => (
                                            <span key={i} className="px-1.5 py-0.5 bg-purple-100 rounded text-xs font-mono">{ref}</span>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  )
                                }
                                return null
                              })}
                            </div>
                          </div>
                        )}

                        {/* Graph Node References */}
                        {finding.node_refs && finding.node_refs.length > 0 && (
                          <div className="bg-gray-50 rounded p-3 border border-gray-200">
                            <p className="text-xs font-semibold text-gray-700 mb-2 flex items-center gap-1">
                              <Network size={12} />
                              Graph Node References ({finding.node_refs.length})
                            </p>
                            <div className="flex flex-wrap gap-1.5">
                              {finding.node_refs.map((ref, idx) => (
                                <span key={idx} className="px-2 py-1 bg-white rounded text-xs font-mono text-gray-700 border border-gray-300">
                                  {ref}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Model Information */}
                        {(finding.junior_model || finding.senior_model || finding.created_by) && (
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 bg-gray-50 rounded p-3 border border-gray-200">
                            {finding.junior_model && (
                              <div>
                                <p className="text-xs font-semibold text-gray-600 mb-1">Junior Agent (Scout)</p>
                                <p className="text-xs text-gray-700 font-mono">{finding.junior_model}</p>
                              </div>
                            )}
                            {finding.senior_model && (
                              <div>
                                <p className="text-xs font-semibold text-gray-600 mb-1">Senior Agent (Strategist)</p>
                                <p className="text-xs text-gray-700 font-mono">{finding.senior_model}</p>
                              </div>
                            )}
                            {finding.created_by && (
                              <div>
                                <p className="text-xs font-semibold text-gray-600 mb-1">Created By</p>
                                <p className="text-xs text-gray-700">{finding.created_by}</p>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Additional Metadata */}
                        <div className="bg-gray-50 rounded p-3 border border-gray-200">
                          <p className="text-xs font-semibold text-gray-700 mb-2">Metadata</p>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                            {finding.created_at && (
                              <div>
                                <span className="text-gray-600">Created:</span>
                                <p className="text-gray-700">{new Date(finding.created_at).toLocaleString()}</p>
                              </div>
                            )}
                            {finding.session_id && (
                              <div>
                                <span className="text-gray-600">Session:</span>
                                <p className="text-gray-700 font-mono text-xs">{finding.session_id.substring(0, 12)}...</p>
                              </div>
                            )}
                            {finding.visibility && (
                              <div>
                                <span className="text-gray-600">Visibility:</span>
                                <p className="text-gray-700 capitalize">{finding.visibility}</p>
                              </div>
                            )}
                            {finding.properties && Object.keys(finding.properties).length > 0 && (
                              <div className="md:col-span-2">
                                <span className="text-gray-600">Properties:</span>
                                <p className="text-gray-700 font-mono text-xs truncate">
                                  {JSON.stringify(finding.properties).substring(0, 60)}...
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Any other Hound fields */}
                        {Object.keys(finding).filter(key => 
                          !['id', 'severity', 'title', 'description', 'location', 'code_snippet', 
                            'recommendation', 'fix_suggestion', 'confidence', 'status', 'vulnerability_type',
                            'evidence', 'node_refs', 'reasoning', 'created_at', 'junior_model', 'senior_model',
                            'created_by', 'session_id', 'visibility', 'properties', 'root_cause', 'attack_vector'].includes(key)
                        ).length > 0 && (
                          <div className="bg-gray-100 rounded p-3 border border-gray-300">
                            <p className="text-xs font-semibold text-gray-700 mb-2">Additional Fields</p>
                            <pre className="text-xs font-mono text-gray-600 overflow-x-auto">
                              {JSON.stringify(
                                Object.fromEntries(
                                  Object.entries(finding).filter(([key]) => 
                                    !['id', 'severity', 'title', 'description', 'location', 'code_snippet', 
                                      'recommendation', 'fix_suggestion', 'confidence', 'status', 'vulnerability_type',
                                      'evidence', 'node_refs', 'reasoning', 'created_at', 'junior_model', 'senior_model',
                                      'created_by', 'session_id', 'visibility', 'properties', 'root_cause', 'attack_vector'].includes(key)
                                  )
                                ), null, 2
                              )}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}

                    {finding.code_snippet && (
                      <div className="bg-white rounded p-3 mb-3 font-mono text-sm overflow-x-auto mt-3">
                        <pre className="whitespace-pre-wrap">{finding.code_snippet}</pre>
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
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-8">
      <div className="max-w-4xl w-full">
        {/* Sign In Button */}
        {!isAuthenticated && !showSignIn && (
          <div className="flex justify-end mb-4">
            <button
              onClick={() => setShowSignIn(true)}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Sign In
            </button>
          </div>
        )}

        {/* Sign In Modal */}
        {showSignIn && !isAuthenticated && (
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Sign In to Heimdall</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                <input
                  type="email"
                  placeholder="your@email.com"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const email = (e.target as HTMLInputElement).value
                      const passwordInput = (e.target as HTMLInputElement).nextElementSibling?.querySelector('input') as HTMLInputElement
                      if (passwordInput?.value) {
                        handleSignIn(email, passwordInput.value)
                      }
                    }
                  }}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                <input
                  type="password"
                  placeholder="Enter your password"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const password = (e.target as HTMLInputElement).value
                      const emailInput = (e.target as HTMLInputElement).previousElementSibling?.querySelector('input') as HTMLInputElement
                      if (emailInput?.value) {
                        handleSignIn(emailInput.value, password)
                      }
                    }
                  }}
                />
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    const emailInput = document.querySelector('input[type="email"]') as HTMLInputElement
                    const passwordInput = document.querySelector('input[type="password"]') as HTMLInputElement
                    if (emailInput && passwordInput) {
                      handleSignIn(emailInput.value, passwordInput.value)
                    }
                  }}
                  className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
                >
                  Sign In
                </button>
                <button
                  onClick={() => setShowSignIn(false)}
                  className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Cancel
                </button>
              </div>
              <p className="text-sm text-gray-500 text-center mt-4">
                For MVP: Any email/password combination will work. Full authentication coming soon.
              </p>
            </div>
          </div>
        )}

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
          <>
            <WhitelistModal />
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
              
              {/* Whitelist Configuration Card - Always Visible & Prominent */}
              <div className="p-5 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-300 shadow-sm">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <FileCode size={20} className="text-indigo-600" />
                      <h3 className="text-base font-semibold text-gray-900">File Whitelist Configuration</h3>
                    </div>
                    <p className="text-sm text-gray-700 mb-2">
                      {autoGenerateWhitelist 
                        ? `✅ Auto-generating whitelist with ${parseInt(whitelistLocBudget || '50000').toLocaleString()} LOC budget (recommended)`
                        : graphFileFilter 
                          ? `✅ Manual whitelist: ${graphFileFilter.split(',').filter(f => f.trim()).length} files specified`
                          : '⚠️ No whitelist configured - all files will be analyzed (may be slow for large repos)'}
                    </p>
                    <p className="text-xs text-gray-600">
                      {autoGenerateWhitelist 
                        ? 'Files will be automatically selected based on importance and LOC budget'
                        : graphFileFilter 
                          ? 'Using your manually specified file list'
                          : 'For large repositories, configure a whitelist to improve performance and focus'}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowWhitelistModal(true)}
                    className="px-5 py-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 text-sm font-semibold flex items-center gap-2 shadow-md hover:shadow-lg transition-all whitespace-nowrap"
                  >
                    <Settings size={18} />
                    Configure
                  </button>
                </div>
              </div>

              {/* Advanced Options */}
              <div className="border-t pt-4">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm font-medium text-gray-700 hover:text-gray-900 mb-4"
                >
                  <Settings size={16} />
                  {showAdvanced ? 'Hide' : 'Show'} Advanced Options
                </button>

                {showAdvanced && (
                  <div className="space-y-6 p-4 bg-gray-50 rounded-lg">
                    {/* Basic Audit Options */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <Zap size={16} className="text-indigo-600" />
                        Basic Audit Options
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Audit Mode</label>
                          <select
                            value={auditMode}
                            onChange={(e) => setAuditMode(e.target.value as 'sweep' | 'intuition')}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                          >
                            <option value="sweep">Sweep Mode (Broad, Systematic)</option>
                            <option value="intuition">Intuition Mode (Deep, Targeted)</option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Iterations per Investigation</label>
                          <input
                            type="number"
                            value={iterations}
                            onChange={(e) => setIterations(e.target.value)}
                            min="1"
                            max="100"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Plan N (Investigations per Batch)</label>
                          <input
                            type="number"
                            value={planN}
                            onChange={(e) => setPlanN(e.target.value)}
                            min="1"
                            max="20"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Time Limit (minutes)</label>
                          <input
                            type="number"
                            value={timeLimit}
                            onChange={(e) => setTimeLimit(e.target.value)}
                            placeholder="No limit"
                            min="1"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Mission Statement</label>
                          <input
                            type="text"
                            value={mission}
                            onChange={(e) => setMission(e.target.value)}
                            placeholder="Overarching mission for the audit"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
                          />
                        </div>
                        <div className="flex items-center gap-2 pt-6">
                          <input
                            type="checkbox"
                            checked={debugMode}
                            onChange={(e) => setDebugMode(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Enable debug logging</label>
                        </div>
                      </div>
                    </div>

                    {/* Graph Building Options */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <Layers size={16} className="text-indigo-600" />
                        Knowledge Graph Options
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={buildGraphs}
                            onChange={(e) => setBuildGraphs(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Build knowledge graphs</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={graphAuto}
                            onChange={(e) => setGraphAuto(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Auto-generate default graphs</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={graphInitOnly}
                            onChange={(e) => setGraphInitOnly(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Initialize SystemArchitecture only</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={graphRefineExisting}
                            onChange={(e) => setGraphRefineExisting(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Refine existing graphs</label>
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Max Graph Iterations</label>
                          <input
                            type="number"
                            value={graphMaxIterations}
                            onChange={(e) => setGraphMaxIterations(e.target.value)}
                            min="1"
                            max="10"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Max Graphs</label>
                          <input
                            type="number"
                            value={graphMaxGraphs}
                            onChange={(e) => setGraphMaxGraphs(e.target.value)}
                            min="1"
                            max="20"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        {/* Whitelist info - full config in modal */}
                        <div className="md:col-span-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <p className="text-xs font-medium text-gray-900 mb-1">
                                File Whitelist: {autoGenerateWhitelist 
                                  ? `Auto (${parseInt(whitelistLocBudget).toLocaleString()} LOC)`
                                  : graphFileFilter 
                                    ? `Manual (${graphFileFilter.split(',').filter(f => f.trim()).length} files)`
                                    : 'Not configured'}
                              </p>
                              <p className="text-xs text-gray-600">
                                {autoGenerateWhitelist 
                                  ? 'Files will be automatically selected within LOC budget'
                                  : graphFileFilter 
                                    ? 'Using manually specified files'
                                    : 'No whitelist - all files will be analyzed (may be slow for large repos)'}
                              </p>
                            </div>
                            <button
                              type="button"
                              onClick={() => setShowWhitelistModal(true)}
                              className="ml-3 px-3 py-1.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 text-xs font-medium"
                            >
                              Configure
                            </button>
                          </div>
                        </div>
                        <div className="md:col-span-2">
                          <label className="block text-xs font-medium text-gray-600 mb-1">Focus Areas</label>
                          <input
                            type="text"
                            value={graphFocusAreas}
                            onChange={(e) => setGraphFocusAreas(e.target.value)}
                            placeholder="access control, reentrancy, math"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Model Overrides */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <Settings size={16} className="text-indigo-600" />
                        Model Overrides (Optional)
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Scout Platform</label>
                          <input
                            type="text"
                            value={scoutPlatform}
                            onChange={(e) => setScoutPlatform(e.target.value)}
                            placeholder="xai, openai, anthropic"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Scout Model</label>
                          <input
                            type="text"
                            value={scoutModel}
                            onChange={(e) => setScoutModel(e.target.value)}
                            placeholder="grok-code-fast-1"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Strategist Platform</label>
                          <input
                            type="text"
                            value={strategistPlatform}
                            onChange={(e) => setStrategistPlatform(e.target.value)}
                            placeholder="xai, openai, anthropic"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Strategist Model</label>
                          <input
                            type="text"
                            value={strategistModel}
                            onChange={(e) => setStrategistModel(e.target.value)}
                            placeholder="grok-4-fast-reasoning"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Session Options */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <Clock size={16} className="text-indigo-600" />
                        Session Options
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Session ID (attach to existing)</label>
                          <input
                            type="text"
                            value={sessionId}
                            onChange={(e) => setSessionId(e.target.value)}
                            placeholder="Leave empty for new session"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        <div className="flex items-center gap-2 pt-6">
                          <input
                            type="checkbox"
                            checked={newSession}
                            onChange={(e) => setNewSession(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Create new session</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={sessionPrivateHypotheses}
                            onChange={(e) => setSessionPrivateHypotheses(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Keep hypotheses private to session</label>
                        </div>
                      </div>
                    </div>

                    {/* Advanced Options */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <Zap size={16} className="text-indigo-600" />
                        Advanced Options
                      </h3>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={strategistTwoPass}
                            onChange={(e) => setStrategistTwoPass(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Enable strategist two-pass self-critique</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={enableTelemetry}
                            onChange={(e) => setEnableTelemetry(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Enable telemetry (real-time monitoring)</label>
                        </div>
                      </div>
                    </div>

                    {/* Finalize Options */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <CheckCircle size={16} className="text-indigo-600" />
                        Finalize Options
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Confidence Threshold (0.0-1.0)</label>
                          <input
                            type="number"
                            value={finalizeThreshold}
                            onChange={(e) => setFinalizeThreshold(e.target.value)}
                            min="0"
                            max="1"
                            step="0.1"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        <div className="flex items-center gap-2 pt-6">
                          <input
                            type="checkbox"
                            checked={finalizeIncludeBelowThreshold}
                            onChange={(e) => setFinalizeIncludeBelowThreshold(e.target.checked)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <label className="text-sm text-gray-700">Include below threshold</label>
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Finalize Platform</label>
                          <input
                            type="text"
                            value={finalizePlatform}
                            onChange={(e) => setFinalizePlatform(e.target.value)}
                            placeholder="xai, openai, anthropic"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Finalize Model</label>
                          <input
                            type="text"
                            value={finalizeModel}
                            onChange={(e) => setFinalizeModel(e.target.value)}
                            placeholder="grok-4-fast-reasoning"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-start gap-2 text-red-800">
                    <AlertCircle size={20} className="mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="font-semibold mb-1">Error</p>
                      <p className="text-sm">{error}</p>
                      {error.includes('Failed to create Hound project') && (
                        <p className="text-xs mt-2 text-red-600">
                          Tip: Make sure Hound is properly installed and XAI_API_KEY is set in backend/.env
                        </p>
                      )}
                      {error.includes('Hound script not found') && (
                        <p className="text-xs mt-2 text-red-600">
                          Tip: Ensure Hound is installed in the hound/ directory
                        </p>
                      )}
                    </div>
                  </div>
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
          </>
        )}
      </div>
    </main>
  )
}
