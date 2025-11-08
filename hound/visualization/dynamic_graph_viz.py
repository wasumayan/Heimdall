"""Dynamic graph visualization for agent-built knowledge graphs."""

import json
from pathlib import Path


def generate_dynamic_visualization(
    graphs_dir: Path,
    output_path: Path | None = None,
    include_card_viewer: bool = True
) -> Path:
    """
    Generate HTML visualization for dynamic multi-graph knowledge base.
    
    Creates an interactive view with:
    - Graph selector to switch between different graphs
    - Node type filtering
    - Timeline view showing when nodes were created
    - Cross-graph connections view
    """
    
    # Load all graphs
    graphs = {}
    for graph_file in graphs_dir.glob("graph_*.json"):
        with open(graph_file) as f:
            graph_data = json.load(f)
            graphs[graph_data["name"]] = graph_data
    
    # Load card store if available
    card_store = {}
    card_store_file = graphs_dir / "card_store.json"
    if card_store_file.exists() and include_card_viewer:
        with open(card_store_file) as f:
            card_store = json.load(f)
    
    # Load combined results if available
    results_file = graphs_dir / "knowledge_graphs.json"
    observations = []
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
            observations = results.get("observations", [])
    
    if not output_path:
        output_path = graphs_dir / "visualization.html"
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dynamic Knowledge Graphs Visualization</title>
    <meta charset="utf-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            color: #e6edf3;
        }}
        
        #header {{
            background: linear-gradient(135deg, #1c2128 0%, #2d333b 100%);
            color: #e6edf3;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            border-bottom: 1px solid #30a14e33;
        }}
        
        #header h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #30a14e;
        }}
        
        #controls {{
            background: #161b22;
            padding: 16px;
            border-bottom: 1px solid #30363d;
            display: flex;
            gap: 24px;
            align-items: center;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        
        .control-group label {{
            font-size: 11px;
            color: #7d8590;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }}
        
        select, button {{
            background: #21262d;
            color: #e6edf3;
            border: 1px solid #30363d;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            transition: all 0.2s;
        }}
        
        select:hover, button:hover {{
            background: #30363d;
            border-color: #30a14e;
        }}
        
        select:focus, button:focus {{
            outline: none;
            border-color: #30a14e;
            box-shadow: 0 0 0 2px #30a14e22;
        }}
        
        #main-container {{
            display: flex;
            height: calc(100vh - 140px);
        }}
        
        #graph-container {{
            flex: 1;
            background: #0d1117;
            position: relative;
        }}
        
        #sidebar {{
            width: 380px;
            background: #161b22;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #30363d;
        }}
        
        .sidebar-section {{
            margin-bottom: 24px;
        }}
        
        .sidebar-section h3 {{
            color: #e6edf3;
            margin-bottom: 12px;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }}
        
        #node-info {{
            background: #21262d;
            padding: 14px;
            border-radius: 6px;
            margin-bottom: 12px;
            border: 1px solid #30363d;
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .observation {{
            background: #21262d;
            padding: 10px;
            border-left: 3px solid #30a14e;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 12px;
            color: #8b949e;
        }}
        
        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #30363d;
            font-size: 13px;
        }}
        
        .stat-value {{
            color: #30a14e;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        #tooltip {{
            position: absolute;
            padding: 12px;
            background: rgba(22, 27, 34, 0.95);
            color: #e6edf3;
            border-radius: 6px;
            pointer-events: none;
            opacity: 0;
            font-size: 12px;
            max-width: 300px;
            border: 1px solid #30363d;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .node {{
            cursor: pointer;
        }}
        
        .node circle {{
            stroke-width: 2px;
            transition: all 0.3s;
        }}
        
        .node:hover circle {{
            stroke-width: 3px;
            filter: drop-shadow(0 0 10px currentColor);
        }}
        
        .link {{
            stroke-opacity: 0.4;
            transition: stroke-opacity 0.3s;
        }}
        
        .link:hover {{
            stroke-opacity: 0.8;
        }}
        
        #legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(22, 27, 34, 0.95);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #30363d;
            max-width: 220px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
        }}
        
        #legend h4 {{
            margin: 0 0 12px 0;
            color: #e6edf3;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            border-bottom: 1px solid #30363d;
            padding-bottom: 8px;
        }}
        
        .legend-section {{
            margin-bottom: 15px;
        }}
        
        .legend-section-title {{
            color: #7d8590;
            font-size: 11px;
            margin-bottom: 8px;
            font-weight: 500;
            text-transform: uppercase;
        }}
        
        .legend-item {{
            margin: 6px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: #8b949e;
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .legend-line {{
            width: 18px;
            height: 2px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🔍 Hound3 Knowledge Graph</h1>
        <div style="margin-top: 8px; opacity: 0.8; font-size: 13px; color: #8b949e;">
            Dynamic Security Analysis Visualization
        </div>
    </div>
    
    <div id="controls">
        <div class="control-group">
            <label>Graph</label>
            <select id="graph-selector">
                {' '.join(f'<option value="{name}">{name.split("_")[0] if "_" in name else name}</option>' for name, g in graphs.items())}
            </select>
        </div>
        
        <div class="control-group">
            <label>Node Type Filter</label>
            <select id="type-filter">
                <option value="all">All Types</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>Layout</label>
            <select id="layout-selector">
                <option value="force">Force Directed</option>
                <option value="hierarchical">Hierarchical</option>
                <option value="circular">Circular</option>
            </select>
        </div>
        
        <button id="reset-view">Reset View</button>
    </div>
    
    <div id="main-container">
        <div id="graph-container">
            <div id="legend"></div>
        </div>
        
        <div id="sidebar">
            <div class="sidebar-section">
                <h3>Graph Statistics</h3>
                <div id="stats"></div>
            </div>
            
            <div class="sidebar-section">
                <h3>Selected Node</h3>
                <div id="node-info">
                    <em style="color: #888;">Click a node to see details</em>
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Source Cards</h3>
                <div id="card-viewer" style="display: none;">
                    <div style="margin-bottom: 10px;">
                        <select id="card-selector" style="width: 100%;">
                            <option value="">Select a card to view code</option>
                        </select>
                    </div>
                    <div id="card-content" style="background: #0f3460; padding: 10px; border-radius: 6px; max-height: 300px; overflow-y: auto; font-family: 'Monaco', 'Menlo', monospace; font-size: 12px; white-space: pre-wrap;">
                        <em style="color: #888;">Select a card above to view its content</em>
                    </div>
                </div>
                <div id="no-cards" style="color: #888; font-style: italic;">
                    No source cards for this node
                </div>
            </div>
            
            <div class="sidebar-section">
                <h3>Agent Observations</h3>
                <div id="observations"></div>
            </div>
        </div>
    </div>
    
    <div id="tooltip"></div>
    
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        // All graph data
        const allGraphs = {json.dumps(graphs)};
        const observations = {json.dumps(observations)};
        const cardStore = {json.dumps(card_store)};
        
        let currentGraph = null;
        let currentGraphName = Object.keys(allGraphs)[0];
        let simulation = null;
        
        // Professional color schemes with tech accent
        const typeColors = {{
            // Common node types from graph generation
            'contract': '#30a14e',         // GitHub green - main contracts
            'interface': '#58a6ff',        // GitHub blue - interfaces
            'library': '#a371f7',          // Purple - library contracts
            'function': '#56d364',         // Light green - functions
            'storage': '#f9826c',          // Warm orange - storage/state
            'event': '#79c0ff',            // Light blue - events
            'modifier': '#bc8cff',         // Light purple - modifiers
            'external_actor': '#d29922',   // Gold - external entities
            'role': '#ffa657',             // Orange - roles/permissions
            
            // Legacy/alternative types
            'code': '#30a14e',             // GitHub green
            'concept': '#58a6ff',          // GitHub blue
            'invariant': '#f9826c',        // Warm orange
            'observation': '#a371f7',      // Purple
            'hypothesis': '#d29922',       // Gold
            'issue': '#f85149',            // GitHub red
            'pattern': '#56d364',          // Light green
            'dataflow': '#79c0ff',         // Light blue
            'control': '#ff7b72',          // Coral
            'vulnerability': '#da3633',   // Danger red
            'assumption': '#ffa657',       // Orange
            'constraint': '#bc8cff',      // Light purple
            'custom': '#8b949e'            // Gray - fallback
        }};
        
        // Edge type colors - expanded palette with more distinct colors
        const edgeColors = {{
            // Primary relationships (high contrast)
            'calls': '#30a14e',           // Green
            'contains': '#58a6ff',         // Blue  
            'depends_on': '#f9826c',       // Light red/orange
            'references': '#a371f7',       // Purple
            'uses': '#79c0ff',             // Light blue
            'implements': '#56d364',       // Bright green
            'extends': '#bc8cff',          // Light purple
            'imports': '#d29922',          // Gold
            
            // Data flow (warm colors)
            'dataflow': '#ffa657',         // Orange
            'reads': '#39d353',            // Green
            'writes': '#ff7b72',           // Coral
            'modifies': '#f85149',         // Red
            
            // Control flow (cool colors)
            'control': '#ffea7f',          // Yellow
            'triggers': '#db61a2',         // Pink
            'validates': '#2ea043',        // Dark green
            'checks': '#539bf5',           // Sky blue
            
            // Security/Analysis (strong colors)
            'violates': '#da3633',         // Danger red
            'protects': '#1f6feb',         // Royal blue
            'trusts': '#8957e5',           // Violet
            'restricts': '#f778ba',        // Light pink
            
            // Structural (purple spectrum)
            'inherits': '#986ee2',         // Lavender
            'overrides': '#ff9492',        // Salmon
            'delegates': '#56d4dd',        // Cyan
            'aggregates': '#a475f9',       // Medium purple
            
            // Associations (varied)
            'relates_to': '#8b949e',       // Gray (neutral)
            'links_to': '#7ee787',         // Mint green
            'connects': '#ffc680',         // Peach
            'maps_to': '#c297ff',          // Light violet
            
            // State/Lifecycle (traffic light colors)
            'initializes': '#3fb950',      // Green
            'creates': '#58a6ff',          // Blue
            'destroys': '#f85149',         // Red
            'updates': '#d29922',          // Gold
            
            // Additional common relationships
            'emits': '#ff6b6b',            // Light red
            'listens': '#4ecdc4',          // Teal
            'owns': '#95e1d3',             // Mint
            'manages': '#f38181',          // Rose
            'requires': '#aa96da',         // Soft purple
            'provides': '#8fcaca',         // Soft cyan
            'processes': '#ffd93d',        // Bright yellow
            'handles': '#6bcf7f',          // Fresh green
            
            // Default fallback - will cycle through palette for unknowns
            'custom': '#8b949e'            // Gray
        }};
        
        // Color palette for unknown edge types (will cycle through)
        const fallbackColors = [
            '#ff6b6b', '#4ecdc4', '#95e1d3', '#f38181', '#aa96da',
            '#8fcaca', '#ffd93d', '#6bcf7f', '#ffbe76', '#ff7979',
            '#badc58', '#dfe4ea', '#5f27cd', '#00d2d3', '#48dbfb',
            '#0abde3', '#ee5a24', '#f368e0', '#feca57', '#ff9ff3'
        ];
        
        // Function to get edge color (with fallback for unknown types)
        function getEdgeColor(type) {{
            if (edgeColors[type]) {{
                return edgeColors[type];
            }}
            // Generate consistent color for unknown types
            let hash = 0;
            for (let i = 0; i < type.length; i++) {{
                hash = ((hash << 5) - hash) + type.charCodeAt(i);
                hash = hash & hash;
            }}
            return fallbackColors[Math.abs(hash) % fallbackColors.length];
        }}
        
        // Initialize
        function init() {{
            loadGraph(currentGraphName);
            setupEventListeners();
            updateObservations();
        }}
        
        function loadGraph(graphName) {{
            currentGraphName = graphName;
            currentGraph = allGraphs[graphName];
            
            // Clear existing graph
            d3.select("#graph-container svg").remove();
            
            // Create new graph
            createGraph(currentGraph);
            updateStats();
            updateTypeFilter();
            updateLegend();
        }}
        
        function createGraph(graphData) {{
            const width = document.getElementById('graph-container').clientWidth;
            const height = document.getElementById('graph-container').clientHeight;
            
            const svg = d3.select("#graph-container")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            const g = svg.append("g");
            
            // Zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => {{
                    g.attr("transform", event.transform);
                }});
            
            svg.call(zoom);
            
            // Process nodes and edges
            const nodes = graphData.nodes || [];
            const edges = graphData.edges || [];
            
            // Create node ID set for validation
            const nodeIds = new Set(nodes.map(n => n.id));
            
            // Create links data, filtering out invalid edges
            const links = edges
                .filter(e => {{
                    if (!nodeIds.has(e.source_id)) {{
                        console.warn(`Edge references non-existent source node: ${{e.source_id}}`);
                        return false;
                    }}
                    if (!nodeIds.has(e.target_id)) {{
                        console.warn(`Edge references non-existent target node: ${{e.target_id}}`);
                        return false;
                    }}
                    return true;
                }})
                .map(e => ({{
                    source: e.source_id,
                    target: e.target_id,
                    type: e.type,
                    label: e.label,
                    confidence: e.confidence || 1
                }}));
            
            // Create force simulation with better spacing
            simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links)
                    .id(d => d.id)
                    .distance(150))  // Increased distance
                .force("charge", d3.forceManyBody().strength(-800))  // Stronger repulsion
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(50))  // Larger collision radius
                .force("x", d3.forceX(width / 2).strength(0.05))  // Gentle centering
                .force("y", d3.forceY(height / 2).strength(0.05));
            
            // Create arrow markers for edge types
            const allEdgeTypes = [...new Set(edges.map(e => e.type))];
            svg.append("defs").selectAll("marker")
                .data(allEdgeTypes)
                .enter().append("marker")
                .attr("id", d => `arrow-${{d}}`)
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 25)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", d => getEdgeColor(d));
            
            // Create link group for lines and labels
            const linkGroup = g.append("g").attr("class", "links");
            
            // Create links with edge-specific colors
            const link = linkGroup.selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("class", "link")
                .attr("stroke", d => getEdgeColor(d.type))
                .attr("stroke-width", d => 1 + Math.sqrt(d.confidence * 2))
                .attr("marker-end", d => `url(#arrow-${{d.type}})`);
            
            // Create edge labels
            const linkLabel = linkGroup.selectAll("text")
                .data(links)
                .enter().append("text")
                .attr("class", "link-label")
                .attr("font-size", "9px")
                .attr("fill", d => getEdgeColor(d.type))
                .attr("text-anchor", "middle")
                .attr("opacity", 0.7)
                .text(d => d.label || d.type);
            
            // Create nodes
            const node = g.append("g")
                .selectAll(".node")
                .data(nodes)
                .enter().append("g")
                .attr("class", "node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            node.append("circle")
                .attr("r", d => {{
                    const confidence = d.confidence || 1;
                    return 10 + confidence * 8;
                }})
                .attr("fill", d => typeColors[d.type?.toLowerCase()] || typeColors['custom'])
                .attr("stroke", d => typeColors[d.type?.toLowerCase()] || typeColors['custom'])
                .attr("fill-opacity", 0.8);
            
            // Node label
            node.append("text")
                .attr("dx", 18)
                .attr("dy", 4)
                .text(d => d.label ? (d.label.length > 30 ? d.label.substring(0, 30) + "..." : d.label) : "")
                .style("fill", "#e6edf3")
                .style("font-size", "12px")
                .style("font-weight", "500")
                .style("text-shadow", "1px 1px 3px rgba(0,0,0,0.5)");
            
            // Add observations/assumptions box if they exist
            node.each(function(d) {{
                const g = d3.select(this);
                const hasObs = d.observations && d.observations.length > 0;
                const hasAssump = d.assumptions && d.assumptions.length > 0;
                
                if (hasObs || hasAssump) {{
                    // Create background box
                    const bbox = g.append("rect")
                        .attr("x", 16)
                        .attr("y", 10)
                        .attr("rx", 3)
                        .attr("ry", 3)
                        .style("fill", "rgba(22, 27, 34, 0.8)")
                        .style("stroke", "rgba(48, 54, 61, 0.8)")
                        .style("stroke-width", "1px");
                    
                    let yOffset = 22;
                    let maxWidth = 0;
                    const textElements = [];
                    
                    // Add observations (up to 3)
                    if (hasObs) {{
                        const obsToShow = d.observations
                            .sort((a, b) => {{
                                const confA = (typeof a === 'object' ? a.confidence : 1) || 1;
                                const confB = (typeof b === 'object' ? b.confidence : 1) || 1;
                                return confB - confA;
                            }})
                            .slice(0, 3);
                        
                        obsToShow.forEach(obs => {{
                            const obsText = typeof obs === 'object' ? 
                                (obs.description || obs.content || '') : String(obs);
                            if (obsText) {{
                                const text = g.append("text")
                                    .attr("x", 20)
                                    .attr("y", yOffset)
                                    .text("✓ " + (obsText.length > 25 ? obsText.substring(0, 25) + "..." : obsText))
                                    .style("fill", "#30a14e")
                                    .style("font-size", "10px")
                                    .style("font-family", "'JetBrains Mono', monospace");
                                textElements.push(text);
                                yOffset += 12;
                            }}
                        }});
                    }}
                    
                    // Add assumptions (up to 2)
                    if (hasAssump) {{
                        const assumToShow = d.assumptions
                            .sort((a, b) => {{
                                const confA = (typeof a === 'object' ? a.confidence : 0.5) || 0.5;
                                const confB = (typeof b === 'object' ? b.confidence : 0.5) || 0.5;
                                return confB - confA;
                            }})
                            .slice(0, 2);
                        
                        assumToShow.forEach(assum => {{
                            const assumText = typeof assum === 'object' ? 
                                (assum.description || assum.content || '') : String(assum);
                            if (assumText) {{
                                const text = g.append("text")
                                    .attr("x", 20)
                                    .attr("y", yOffset)
                                    .text("? " + (assumText.length > 25 ? assumText.substring(0, 25) + "..." : assumText))
                                    .style("fill", "#d29922")
                                    .style("font-size", "10px")
                                    .style("font-family", "'JetBrains Mono', monospace");
                                textElements.push(text);
                                yOffset += 12;
                            }}
                        }});
                    }}
                    
                    // Calculate max width from text elements
                    textElements.forEach(text => {{
                        const bbox = text.node().getBBox();
                        maxWidth = Math.max(maxWidth, bbox.width);
                    }});
                    
                    // Update background box size
                    bbox.attr("width", maxWidth + 8)
                        .attr("height", yOffset - 14);
                }}
            }});
            
            // Tooltip
            const tooltip = d3.select("#tooltip");
            
            node.on("mouseover", function(event, d) {{
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                
                let content = `<strong>${{d.label}}</strong><br/>`;
                content += `Type: ${{d.type}}<br/>`;
                if (d.confidence !== undefined) {{
                    content += `Confidence: ${{(d.confidence * 100).toFixed(0)}}%<br/>`;
                }}
                if (d.description) {{
                    content += `<br/>${{d.description}}`;
                }}
                if (d.observations && d.observations.length > 0) {{
                    content += `<br/><br/><strong style="color: #30a14e;">Observations:</strong><br/>`;
                    d.observations.slice(0, 5).forEach(obs => {{
                        const obsText = typeof obs === 'object' ? 
                            (obs.description || obs.content || JSON.stringify(obs)) : String(obs);
                        content += `<span style="color: #30a14e;">✓</span> ${{obsText}}<br/>`;
                    }});
                }}
                if (d.assumptions && d.assumptions.length > 0) {{
                    content += `<br/><strong style="color: #d29922;">Assumptions:</strong><br/>`;
                    d.assumptions.slice(0, 3).forEach(assum => {{
                        const assumText = typeof assum === 'object' ? 
                            (assum.description || assum.content || JSON.stringify(assum)) : String(assum);
                        const conf = typeof assum === 'object' ? assum.confidence : 0.5;
                        content += `<span style="color: #d29922;">?</span> ${{assumText}} <span style="opacity: 0.7;">(conf: ${{conf}})</span><br/>`;
                    }});
                }}
                if (d.properties && Object.keys(d.properties).length > 0) {{
                    content += `<br/><br/>Properties:<br/>`;
                    for (const [key, value] of Object.entries(d.properties)) {{
                        content += `${{key}}: ${{value}}<br/>`;
                    }}
                }}
                
                tooltip.html(content)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }})
            .on("click", function(event, d) {{
                showNodeDetails(d);
            }});
            
            // Update positions
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                linkLabel
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2);
                
                node
                    .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
            }});
        }}
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        function updateStats() {{
            const stats = currentGraph.stats || {{}};
            let html = '';
            
            html += `<div class="stat"><span>Nodes</span><span class="stat-value">${{stats.num_nodes || 0}}</span></div>`;
            html += `<div class="stat"><span>Edges</span><span class="stat-value">${{stats.num_edges || 0}}</span></div>`;
            html += `<div class="stat"><span>Node Types</span><span class="stat-value">${{(stats.node_types || []).length}}</span></div>`;
            html += `<div class="stat"><span>Edge Types</span><span class="stat-value">${{(stats.edge_types || []).length}}</span></div>`;
            
            document.getElementById('stats').innerHTML = html;
        }}
        
        function updateTypeFilter() {{
            const types = new Set();
            (currentGraph.nodes || []).forEach(n => types.add(n.type));
            
            const select = document.getElementById('type-filter');
            select.innerHTML = '<option value="all">All Types</option>';
            
            Array.from(types).sort().forEach(type => {{
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type;
                select.appendChild(option);
            }});
        }}
        
        function updateLegend() {{
            // Get unique node types
            const nodeTypes = new Set();
            (currentGraph.nodes || []).forEach(n => nodeTypes.add(n.type));
            
            // Get unique edge types
            const edgeTypes = new Set();
            (currentGraph.edges || []).forEach(e => edgeTypes.add(e.type));
            
            let html = '<h4>Legend</h4>';
            
            // Node types section
            if (nodeTypes.size > 0) {{
                html += '<div class="legend-section">';
                html += '<div class="legend-section-title">Node Types</div>';
                Array.from(nodeTypes).sort().forEach(type => {{
                    const color = typeColors[type?.toLowerCase()] || typeColors['custom'];
                    html += `<div class="legend-item">
                        <div class="legend-color" style="background: ${{color}};"></div>
                        <span>${{type}}</span>
                    </div>`;
                }});
                html += '</div>';
            }}
            
            // Edge types section
            if (edgeTypes.size > 0) {{
                html += '<div class="legend-section">';
                html += '<div class="legend-section-title">Edge Types</div>';
                Array.from(edgeTypes).sort().forEach(type => {{
                    const color = getEdgeColor(type);
                    html += `<div class="legend-item">
                        <div class="legend-line" style="background: ${{color}};"></div>
                        <span>${{type}}</span>
                    </div>`;
                }});
                html += '</div>';
            }}
            
            // Add observations/assumptions legend
            html += '<div class="legend-section">';
            html += '<div class="legend-section-title">Annotations</div>';
            html += '<div class="legend-item"><span style="color: #30a14e; font-weight: bold;">✓</span> <span>Verified observation</span></div>';
            html += '<div class="legend-item"><span style="color: #d29922; font-weight: bold;">?</span> <span>Unverified assumption</span></div>';
            html += '</div>';
            
            document.getElementById('legend').innerHTML = html;
        }}
        
        function showNodeDetails(node) {{
            const nodeColor = typeColors[node.type?.toLowerCase()] || typeColors['custom'];
            let html = `<h4 style="margin: 0 0 10px 0; color: ${{nodeColor}};">${{node.label}}</h4>`;
            html += `<div style="color: #8b949e;"><strong>Type:</strong> <span style="color: ${{nodeColor}};">${{node.type}}</span></div>`;
            html += `<div style="color: #8b949e;"><strong>ID:</strong> <span style="color: #7d8590; font-family: 'JetBrains Mono', monospace; font-size: 11px;">${{node.id}}</span></div>`;
            
            if (node.confidence !== undefined) {{
                const confColor = node.confidence > 0.7 ? '#30a14e' : node.confidence > 0.4 ? '#d29922' : '#f9826c';
                html += `<div style="color: #8b949e;"><strong>Confidence:</strong> <span style="color: ${{confColor}};">${{(node.confidence * 100).toFixed(0)}}%</span></div>`;
            }}
            
            if (node.created_by) {{
                html += `<div style="color: #8b949e;"><strong>Created:</strong> <span style="color: #7d8590;">${{node.created_by}}</span></div>`;
            }}
            
            if (node.description) {{
                html += `<div style="margin-top: 10px; color: #8b949e;"><strong>Description:</strong><br/><span style="color: #e6edf3;">${{node.description}}</span></div>`;
            }}
            
            // Observations (verified facts about the system)
            if (node.observations && node.observations.length > 0) {{
                html += '<div style="margin-top: 12px; padding: 10px; background: rgba(48, 164, 78, 0.1); border-radius: 6px; border: 1px solid rgba(48, 164, 78, 0.3);">';
                html += '<strong style="color: #30a14e; font-size: 11px; text-transform: uppercase;">✓ Verified Observations</strong>';
                node.observations.forEach(obs => {{
                    let obsText, obsType = 'general', obsConf = 1.0;
                    if (typeof obs === 'object') {{
                        obsText = obs.description || obs.content || JSON.stringify(obs);
                        obsType = obs.type || 'general';
                        obsConf = obs.confidence !== undefined ? obs.confidence : 1.0;
                    }} else {{
                        obsText = String(obs);
                    }}
                    const typeLabel = obsType !== 'general' ? ` [${{obsType}}]` : '';
                    const confLabel = obsConf < 1.0 ? ` (conf: ${{obsConf.toFixed(1)}})` : '';
                    html += `<div style="margin-left: 10px; margin-top: 4px; color: #30a14e; font-size: 11px; font-family: 'JetBrains Mono', monospace;">• ${{obsText}}${{typeLabel}}${{confLabel}}</div>`;
                }});
                html += '</div>';
            }}
            
            // Assumptions (unverified, needs validation)
            if (node.assumptions && node.assumptions.length > 0) {{
                html += '<div style="margin-top: 12px; padding: 10px; background: rgba(210, 153, 34, 0.1); border-radius: 6px; border: 1px solid rgba(210, 153, 34, 0.3);">';
                html += '<strong style="color: #d29922; font-size: 11px; text-transform: uppercase;">? Unverified Assumptions</strong>';
                node.assumptions.forEach(assum => {{
                    let assumText, assumType = 'general', assumConf = 0.5;
                    if (typeof assum === 'object') {{
                        assumText = assum.description || assum.content || JSON.stringify(assum);
                        assumType = assum.type || 'general';
                        assumConf = assum.confidence !== undefined ? assum.confidence : 0.5;
                    }} else {{
                        assumText = String(assum);
                    }}
                    const typeLabel = assumType !== 'general' ? ` [${{assumType}}]` : '';
                    html += `<div style="margin-left: 10px; margin-top: 4px; color: #d29922; font-size: 11px; font-family: 'JetBrains Mono', monospace;">• ${{assumText}}${{typeLabel}} (conf: ${{assumConf.toFixed(1)}})</div>`;
                }});
                html += '</div>';
            }}
            
            if (node.properties && Object.keys(node.properties).length > 0) {{
                html += '<div style="margin-top: 12px;"><strong style="color: #58a6ff; font-size: 11px; text-transform: uppercase;">Properties</strong></div>';
                for (const [key, value] of Object.entries(node.properties)) {{
                    html += `<div style="margin-left: 10px; color: #8b949e; font-size: 12px;">• ${{key}}: <span style="color: #7d8590;">${{JSON.stringify(value)}}</span></div>`;
                }}
            }}
            
            if (node.source_refs && node.source_refs.length > 0) {{
                html += '<div style="margin-top: 12px;"><strong style="color: #56d364; font-size: 11px; text-transform: uppercase;">Source References</strong></div>';
                node.source_refs.forEach(ref => {{
                    html += `<div style="margin-left: 10px; font-size: 10px; color: #7d8590; font-family: 'JetBrains Mono', monospace;">• ${{ref}}</div>`;
                }});
            }}
            
            document.getElementById('node-info').innerHTML = html;
            
            // Update card viewer
            updateCardViewer(node);
        }}
        
        function updateCardViewer(node) {{
            const cardViewer = document.getElementById('card-viewer');
            const noCards = document.getElementById('no-cards');
            const cardSelector = document.getElementById('card-selector');
            
            if (node.source_refs && node.source_refs.length > 0) {{
                // Show card viewer
                cardViewer.style.display = 'block';
                noCards.style.display = 'none';
                
                // Clear and populate selector
                cardSelector.innerHTML = '<option value="">Select a card to view code</option>';
                
                node.source_refs.forEach(cardId => {{
                    if (cardStore[cardId]) {{
                        const card = cardStore[cardId];
                        const option = document.createElement('option');
                        option.value = cardId;
                        option.textContent = `${{cardId}} (${{card.relpath || 'unknown file'}})`;
                        cardSelector.appendChild(option);
                    }}
                }});
                
                // Handle card selection
                cardSelector.onchange = function() {{
                    const cardId = this.value;
                    const cardContent = document.getElementById('card-content');
                    
                    if (cardId && cardStore[cardId]) {{
                        const card = cardStore[cardId];
                        let content = '';
                        
                        // Show file path
                        if (card.relpath) {{
                            content += `// File: ${{card.relpath}}\\n`;
                            if (card.char_start !== undefined && card.char_end !== undefined) {{
                                content += `// Characters: ${{card.char_start}}-${{card.char_end}}\\n`;
                            }}
                            content += '\\n';
                        }}
                        
                        // Show actual code content
                        if (card.content) {{
                            content += card.content;
                        }} else if (card.peek_head || card.peek_tail) {{
                            content += card.peek_head || '';
                            if (card.peek_head && card.peek_tail) {{
                                content += '\\n...\\n';
                            }}
                            content += card.peek_tail || '';
                        }} else {{
                            content = 'No content available for this card';
                        }}
                        
                        cardContent.textContent = content;
                    }} else {{
                        cardContent.innerHTML = '<em style="color: #888;">Select a card above to view its content</em>';
                    }}
                }};
            }} else {{
                // Hide card viewer
                cardViewer.style.display = 'none';
                noCards.style.display = 'block';
            }}
        }}
        
        function updateObservations() {{
            let html = '';
            const recentObs = observations.slice(-10).reverse();
            
            recentObs.forEach(obs => {{
                html += `<div class="observation">${{obs}}</div>`;
            }});
            
            if (html === '') {{
                html = '<em style="color: #888;">No observations yet</em>';
            }}
            
            document.getElementById('observations').innerHTML = html;
        }}
        
        function setupEventListeners() {{
            document.getElementById('graph-selector').addEventListener('change', (e) => {{
                loadGraph(e.target.value);
            }});
            
            document.getElementById('type-filter').addEventListener('change', (e) => {{
                filterNodesByType(e.target.value);
            }});
            
            document.getElementById('layout-selector').addEventListener('change', (e) => {{
                changeLayout(e.target.value);
            }});
            
            document.getElementById('reset-view').addEventListener('click', () => {{
                resetView();
            }});
        }}
        
        function filterNodesByType(type) {{
            d3.selectAll('.node').style('opacity', d => {{
                return type === 'all' || d.type === type ? 1 : 0.1;
            }});
        }}
        
        function changeLayout(layout) {{
            // Simplified - would implement different layout algorithms
            if (simulation) {{
                simulation.alpha(1).restart();
            }}
        }}
        
        function resetView() {{
            const svg = d3.select("svg");
            const width = svg.attr("width");
            const height = svg.attr("height");
            
            svg.transition()
                .duration(750)
                .call(
                    d3.zoom().transform,
                    d3.zoomIdentity.translate(width / 2, height / 2).scale(1)
                );
        }}
        
        // Initialize on load
        init();
    </script>
</body>
</html>"""
    
    # Write HTML
    with open(output_path, "w") as f:
        f.write(html_content)
    
    return output_path