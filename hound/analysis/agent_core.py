"""
Clean autonomous agent implementation - works like Claude Code.
No fuzzy parsing, no prescriptive flow, just autonomous decision-making.
"""

import json
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm.token_tracker import get_token_tracker

try:
    # Pydantic v2 style config
    from pydantic import ConfigDict  # type: ignore
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore

from .concurrent_knowledge import GraphStore


class AgentParameters(BaseModel):
    """Strict parameters schema for agent actions (no extra keys)."""
    # load_graph
    graph_name: str | None = None
    # load_nodes
    node_ids: list[str] | None = None
    # update_node
    node_id: str | None = None
    observations: list[str] | None = None
    assumptions: list[str] | None = None
    # form_hypothesis
    description: str | None = None
    details: str | None = None  # Added to support Gemini's response format
    vulnerability_type: str | None = None
    confidence: float | None = None
    severity: str | None = None
    reasoning: str | None = None
    # update_hypothesis
    hypothesis_index: int | None = None
    hypothesis_id: str | None = None
    new_confidence: float | None = None
    evidence: str | None = None
    evidence_type: str | None = None

    # Ensure OpenAI JSON schema has additionalProperties: false
    if ConfigDict is not None:  # Pydantic v2
        model_config = ConfigDict(extra='forbid')  # type: ignore
    else:  # Fallback for safety
        class Config:  # type: ignore
            extra = 'forbid'


class AgentDecision(BaseModel):
    """Structured decision from the agent."""
    action: str = Field(..., description="Action to take: load_graph, load_nodes, update_node, form_hypothesis, update_hypothesis, complete")
    reasoning: str = Field(..., description="Reasoning for this action")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters as shown in examples")
    
    # Pydantic v2 config for strict validation
    if ConfigDict is not None:  # Pydantic v2
        model_config = ConfigDict(extra='forbid')  # type: ignore
    else:  # Pydantic v1 fallback
        class Config:  # type: ignore
            extra = 'forbid'


class AutonomousAgent:
    """
    Security analysis agent that works autonomously.
    """
    
    def __init__(self, 
                 graphs_metadata_path: Path,
                 manifest_path: Path,
                 agent_id: str,
                 config: dict | None = None,
                 debug: bool = False,
                 session_id: str | None = None):
        """Initialize the autonomous agent."""
        
        self.agent_id = agent_id
        self.manifest_path = manifest_path
        self.debug = debug
        self.session_id = session_id
        # Default hypothesis visibility; can be overridden by runner
        self.default_hypothesis_visibility = 'global'
        
        # Initialize debug logger if needed
        self.debug_logger = None
        if debug:
            from .debug_logger import DebugLogger
            self.debug_logger = DebugLogger(agent_id)
        
        # Initialize LLM client with proper config
        from llm.unified_client import UnifiedLLMClient
        
        # Use provided config or load defaults
        if config is None:
            from utils.config_loader import load_config
            config = load_config()
        
        # Save config for later utilities
        self.config = config

        # Check if there are platform/model overrides in the 'agent' profile
        # If so, use those instead of the 'scout' profile
        if (config and 'models' in config and 'agent' in config['models'] and 
            (config['models']['agent'].get('provider') or config['models']['agent'].get('model'))):
            # Use 'agent' profile when overrides are present
            profile_to_use = "agent"
        else:
            # Fall back to 'scout' profile (default behavior)
            profile_to_use = "scout"
        
        # Use the determined profile for agent operations
        self.llm = UnifiedLLMClient(
            cfg=config,
            profile=profile_to_use,
            debug_logger=self.debug_logger
        )
        # Remember which profile the agent uses for context limit calculations
        self.agent_profile = profile_to_use
        
        # Initialize strategist model for deep thinking
        try:
            self.guidance_client = UnifiedLLMClient(
                cfg=config,
                profile="strategist",
                debug_logger=self.debug_logger
            )
        except Exception:
            # If strategist model not configured, fall back to scout model
            self.guidance_client = self.llm

        # Use the agent model itself for context compression
        # This ensures consistency and leverages the same model's understanding
        self.summarizer = self.llm  # Just use the agent's own LLM
        
        # Remember where graphs live and load metadata
        self.graphs_metadata_path = graphs_metadata_path
        self.available_graphs = self._load_graphs_metadata(graphs_metadata_path)
        
        # Initialize persistent hypothesis store (separate from graphs)
        # Store in the project directory for persistence
        from .concurrent_knowledge import HypothesisStore
        project_dir = graphs_metadata_path.parent.parent  # Go up to project root from graphs/
        # Keep a reference for features like steering inbox, etc.
        try:
            self.project_dir: Path = Path(project_dir)
        except Exception:
            self.project_dir = Path.cwd()
        hypothesis_path = project_dir / "hypotheses.json"
        self.hypothesis_store = HypothesisStore(hypothesis_path, agent_id=agent_id)

        # Initialize per-project coverage index for persistent coverage tracking
        try:
            from .coverage_index import CoverageIndex
            self.coverage_index = CoverageIndex(project_dir / 'coverage_index.json', agent_id=agent_id)
        except Exception:
            # Keep attribute for conditional checks even if init fails
            self.coverage_index = None
        
        # Agent's memory - what it has loaded and discovered
        self.loaded_data = {
            'system_graph': None,  # The always-visible system architecture graph
            'nodes': {},       # Loaded node data by ID
            'code': {},        # Code content by node ID
            'hypotheses': [],  # Formed hypotheses (kept for backward compatibility)
            'graphs': {},      # Additional loaded graphs by name
        }
        # Lazy card index
        self._card_index: dict[str, dict[str, Any]] | None = None
        self._file_to_cards: dict[str, list[str]] = {}
        # Repo root to reconstruct card slices when needed
        self._repo_root: Path | None = None
        try:
            with open(self.manifest_path / 'manifest.json') as _mf:
                _manifest = json.load(_mf)
                rp = _manifest.get('repo_path')
                if rp:
                    self._repo_root = Path(rp)
        except Exception:
            self._repo_root = None
        
        # AUTO-LOAD the system architecture graph (first graph, usually SystemArchitecture or SystemOverview)
        self._auto_load_system_graph()
        
        # Load existing hypotheses from persistent store
        self._load_existing_hypotheses()
        
        # Conversation history for context
        self.conversation_history = []
        # Compressed memory notes
        self.memory_notes: list[str] = []
        # High-level action log
        self.action_log: list[dict[str, Any]] = []
        
        # Current investigation goal
        self.investigation_goal = ""

        # Steering cache (last seen lines to avoid duplication in memory notes)
        self._steering_seen: set[str] = set()

        # Abort flag (set by runner on steering replan)
        self._abort_requested: bool = False
        self._abort_reason: str | None = None

    def request_abort(self, reason: str | None = None):
        """Signal the agent loop to abort the current investigation ASAP.
        The runner uses this when a global steering request should preempt.
        """
        self._abort_requested = True
        self._abort_reason = (reason or "steering_replan").strip() or "steering_replan"

    def _read_steering_notes(self, limit: int = 12) -> list[str]:
        """Read recent steering notes from project .hound/steering.jsonl (if any)."""
        try:
            sfile = self.project_dir / '.hound' / 'steering.jsonl'
            if not sfile.exists():
                return []
            lines = []
            with sfile.open('r', encoding='utf-8', errors='ignore') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                        txt = obj.get('text') or obj.get('message') or obj.get('note')
                        if txt:
                            lines.append(str(txt).strip())
                    except Exception:
                        # fallback to raw line when not JSON
                        lines.append(ln)
            return lines[-limit:]
        except Exception:
            return []
    
    def _load_graphs_metadata(self, metadata_path: Path) -> dict:
        """Load metadata about available graphs."""
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        # Convert to expected format
        graphs = {}
        if 'graphs' in data:
            for name, path in data['graphs'].items():
                graphs[name] = {'path': path, 'name': name}
        
        return graphs
    
    def _auto_load_system_graph(self):
        """Automatically load the SystemOverview/SystemArchitecture graph at startup.
        Prefers file 'graph_SystemArchitecture.json' when present.
        """
        system_graph_name = None
        system_graph_path = None

        # 1) Exact key match 'SystemArchitecture'
        if 'SystemArchitecture' in self.available_graphs:
            system_graph_name = 'SystemArchitecture'
            system_graph_path = Path(self.available_graphs['SystemArchitecture']['path'])

        # 2) Heuristic match by name
        if not system_graph_path:
            for name in self.available_graphs.keys():
                lname = name.lower()
                if 'system' in lname and ('architecture' in lname or 'overview' in lname):
                    system_graph_name = name
                    system_graph_path = Path(self.available_graphs[name]['path'])
                    break

        # 3) Direct file presence: graph_SystemArchitecture.json next to metadata
        if not system_graph_path and hasattr(self, 'graphs_metadata_path') and self.graphs_metadata_path:
            candidate = Path(self.graphs_metadata_path).parent / 'graph_SystemArchitecture.json'
            if candidate.exists():
                system_graph_name = 'SystemArchitecture'
                system_graph_path = candidate

        # 4) Inspect graph files to find internal/display name of SystemArchitecture
        if not system_graph_path:
            for name, meta in self.available_graphs.items():
                try:
                    # Use concurrent-safe reload
                    gd = self._reload_graph(name)
                    if not gd:
                        continue
                    # Check common fields
                    internal = gd.get('internal_name') or gd.get('name') or ''
                    display = gd.get('name') or gd.get('metadata', {}).get('display_name') or ''
                    if str(internal) == 'SystemArchitecture' or str(display).strip().lower() in {'systemarchitecture', 'system architecture', 'systemoverview', 'system overview'}:
                        system_graph_name = name
                        system_graph_path = Path(meta['path'])
                        break
                except Exception:
                    continue

        # 5) Fallback to the first available
        if not system_graph_path and self.available_graphs:
            first_name = list(self.available_graphs.keys())[0]
            system_graph_name = first_name
            system_graph_path = Path(self.available_graphs[first_name]['path'])

        # Load selected graph
        if system_graph_name:
            try:
                # Use concurrent-safe reload
                graph_data = self._reload_graph(system_graph_name)
                if graph_data:
                    self.loaded_data['system_graph'] = {
                        'name': system_graph_name,
                        'data': graph_data
                    }

                    nodes = graph_data.get('nodes', [])
                    edges = graph_data.get('edges', [])
                    print(f"[*] Auto-loaded system graph: {system_graph_name} ({len(nodes)} nodes, {len(edges)} edges)")
            except Exception as e:
                print(f"[!] Failed to auto-load system graph: {e}")
    
    def _load_existing_hypotheses(self):
        """Load existing hypotheses from persistent store into memory."""
        try:
            # Clear existing loaded hypotheses to get fresh view
            self.loaded_data['hypotheses'] = []
            
            # Load from persistent store
            data = self.hypothesis_store._load_data()
            hypotheses = data.get("hypotheses", {})
            
            # Convert to memory format
            for hyp_id, hyp in hypotheses.items():
                self.loaded_data['hypotheses'].append({
                    'id': hyp_id,
                    'description': hyp.get('title', 'Unknown'),
                    'vulnerability_type': hyp.get('vulnerability_type', 'unknown'),
                    'confidence': hyp.get('confidence', 0.5),
                    'status': hyp.get('status', 'proposed'),
                    'node_ids': hyp.get('node_refs', []),
                    'evidence': [e.get('description') for e in hyp.get('evidence', [])]
                })
            
            # Only print on first load, not every refresh
            # if len(self.loaded_data['hypotheses']) > 0:
            #     print(f"[*] Loaded {len(self.loaded_data['hypotheses'])} existing hypotheses")
        except Exception as e:
            print(f"[!] Failed to load existing hypotheses: {e}")
    
    def _refresh_loaded_graphs(self):
        """Refresh loaded graphs from disk to see updates from other agents."""
        try:
            # Refresh system graph if loaded
            if self.loaded_data.get('system_graph'):
                graph_name = self.loaded_data['system_graph']['name']
                refreshed = self._reload_graph(graph_name)
                if refreshed:
                    self.loaded_data['system_graph']['data'] = refreshed
            
            # Refresh any additional loaded graphs
            for graph_name in list(self.loaded_data.get('graphs', {}).keys()):
                refreshed = self._reload_graph(graph_name)
                if refreshed:
                    self.loaded_data['graphs'][graph_name] = refreshed
        except Exception as e:
            print(f"[!] Failed to refresh graphs: {e}")
    
    def investigate(self, prompt: str, max_iterations: int = 20,
                   progress_callback: Callable[[dict], None] | None = None) -> dict:
        """
        Main investigation method - agent works autonomously until complete.
        """
        self.investigation_goal = prompt
        self.conversation_history = [
            {'role': 'user', 'content': prompt}
        ]
        
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Honor external abort signals early in the iteration
            if getattr(self, '_abort_requested', False):
                if progress_callback:
                    try:
                        progress_callback({
                            'status': 'complete',
                            'iteration': iterations,
                            'message': f"Aborted due to: {getattr(self, '_abort_reason', 'steering_replan')}"
                        })
                    except Exception:
                        pass
                break

            if progress_callback:
                progress_callback({
                    'status': 'analyzing',
                    'iteration': iterations,
                    'message': 'Analyzing context and deciding next action'
                })
            
            try:
                # Build full context for agent
                context = self._build_context()
                
                # Get agent's decision using structured output
                decision = self._get_agent_decision(context)
                # Surface the agent's decision and reasoning to UI
                if progress_callback:
                    try:
                        progress_callback({
                            'status': 'decision',
                            'iteration': iterations,
                            'action': decision.action,
                            'reasoning': decision.reasoning,
                            'parameters': decision.parameters,  # Already a dict
                            'message': f"Decided to {decision.action}"
                        })
                    except Exception:
                        pass
                
                # Emit context usage after each decision
                try:
                    if progress_callback:
                        usage_msg = self._format_context_usage()
                        progress_callback({
                            'status': 'usage',
                            'iteration': iterations,
                            'message': usage_msg
                        })
                except Exception:
                    pass
                
                # Log the decision
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': f"Action: {decision.action}\nReasoning: {decision.reasoning}"
                })
                
                if progress_callback:
                    progress_callback({
                        'status': 'executing',
                        'iteration': iterations,
                        'message': f"Executing: {decision.action}"
                    })
                
                # Execute the decision
                result = self._execute_action(decision)
                # Surface result to UI (generic) with defensive guards
                if progress_callback:
                    try:
                        safe = result if isinstance(result, dict) else {'status': 'error', 'error': 'No result'}
                        msg = None
                        try:
                            msg = safe.get('summary') or f"{decision.action} -> {safe.get('status', 'done')}"
                        except Exception:
                            msg = f"{decision.action} -> done"
                        progress_callback({
                            'status': 'result',
                            'iteration': iterations,
                            'action': decision.action,
                            'result': safe,
                            'message': msg
                        })
                    except Exception:
                        pass
                
                # Log the result - use formatted display for readability
                # Defensive: coerce result to dict for safe access
                safe_result = result if isinstance(result, dict) else {}
                # For successful graph/node loads, show the formatted display
                if safe_result.get('status') == 'success':
                    if 'graph_display' in result:
                        # load_graph action - show formatted graph
                        content = f"SUCCESS: {safe_result.get('summary', '')}\n{result['graph_display']}"
                    elif 'nodes_display' in result:
                        # load_nodes action - show formatted nodes with code
                        content = f"SUCCESS: {safe_result.get('summary', '')}\n{result['nodes_display']}"
                    else:
                        # Other successful actions - show as JSON but more readable
                        content = json.dumps(safe_result, indent=2)
                else:
                    # Errors and other statuses - show as JSON
                    content = json.dumps(safe_result or {'status': 'unknown'}, indent=2)
                
                self.conversation_history.append({
                    'role': 'system',
                    'content': content
                })

                # Record action in action log (compact, only non-null params)
                try:
                    # Parameters is now a dict, filter out None values
                    params_obj = {k: v for k, v in decision.parameters.items() if v is not None}
                except Exception:
                    params_obj = {}
                self.action_log.append({
                    'action': decision.action,
                    'params': params_obj,
                    'result': safe_result.get('summary') or safe_result.get('status') or 'ok'
                })

                # Maybe compress history if near budget
                self._maybe_compress_history()
                
                # Emit context usage after applying result and any compression
                try:
                    if progress_callback:
                        usage_msg = self._format_context_usage()
                        progress_callback({
                            'status': 'usage',
                            'iteration': iterations,
                            'message': usage_msg
                        })
                except Exception:
                    pass
                
                # Check if complete
                if decision.action == 'complete':
                    if progress_callback:
                        progress_callback({
                            'status': 'complete',
                            'iteration': iterations,
                            'message': 'Investigation complete'
                        })
                    break

                # Honor external abort signals after executing an action too
                if getattr(self, '_abort_requested', False):
                    if progress_callback:
                        try:
                            progress_callback({
                                'status': 'complete',
                                'iteration': iterations,
                                'message': f"Aborted due to: {getattr(self, '_abort_reason', 'steering_replan')}"
                            })
                        except Exception:
                            pass
                    break
                
                # Update progress based on action
                if decision.action == 'form_hypothesis' and safe_result.get('status') == 'success':
                    if progress_callback:
                        progress_callback({
                            'status': 'hypothesis_formed',
                            'iteration': iterations,
                            'message': f"Formed hypothesis: {decision.parameters.get('description', 'Unknown')}"
                        })
                elif decision.action == 'load_nodes' and safe_result.get('status') == 'success':
                    if progress_callback:
                        progress_callback({
                            'status': 'code_loaded',
                            'iteration': iterations,
                            'message': safe_result.get('summary', 'Loaded nodes')
                        })
                        
            except Exception as e:
                error_msg = f"Error in iteration {iterations}: {str(e)}"
                print(f"[!] {error_msg}")
                if self.debug:
                    traceback.print_exc()
                
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"ERROR: {error_msg}"
                })
        
        # Generate final report
        if progress_callback:
            progress_callback({
                'status': 'generating_report',
                'iteration': iterations,
                'message': 'Generating final report'
            })
        
        return self._generate_report(iterations)
    
    def _format_graph_for_display(self, graph_data: dict, graph_name: str) -> list[str]:
        """Compact representation of the FULL graph (nodes, edges, top annotations).

        - Shows ALL nodes and ALL edges
        - Node line: id|type|label|obs:a;b|asm:c;d (obs/asm optional, truncated)
        - Edge line: src->dst:type|obs:a;b|asm:c   (obs/asm optional, truncated)
        - Keeps lines short by limiting count and length of annotations
        """
        def _short_list(items, max_items=2, max_len=24):
            out = []
            for it in items[:max_items]:
                s = ''
                if isinstance(it, dict):
                    s = str(it.get('description') or it.get('content') or it)
                else:
                    s = str(it)
                s = s.replace('\n', ' ').strip()
                if len(s) > max_len:
                    s = s[:max_len-1] + '…'
                if s:
                    out.append(s)
            return out

        lines: list[str] = []
        lines.append(f"\n--- Graph: {graph_name} ---")
        nodes = graph_data.get('nodes', []) or []
        edges = graph_data.get('edges', []) or []
        lines.append(f"Total: {len(nodes)} nodes, {len(edges)} edges")
        if nodes:
            lines.append("NODES (id|type|label[|obs:…][|asm:…]):")
            for n in nodes:
                nid = n.get('id', 'unknown')
                ntype = n.get('type', '')
                nlabel = n.get('label', '')
                line = f"  {nid}|{ntype}|{nlabel}"
                obs = _short_list(n.get('observations', []) or [], max_items=2, max_len=28)
                if obs:
                    line += f"|obs:{';'.join(obs)}"
                asm = _short_list(n.get('assumptions', []) or [], max_items=2, max_len=28)
                if asm:
                    line += f"|asm:{';'.join(asm)}"
                lines.append(line)
        if edges:
            lines.append("EDGES (src->dst:type[|obs:…][|asm:…]):")
            for e in edges:
                src = e.get('source_id') or e.get('source') or e.get('src') or ''
                dst = e.get('target_id') or e.get('target') or e.get('dst') or ''
                et = e.get('type', '')
                line = f"  {src}->{dst}:{et}"
                eobs = _short_list(e.get('observations', []) or [], max_items=1, max_len=28)
                if eobs:
                    line += f"|obs:{';'.join(eobs)}"
                easm = _short_list(e.get('assumptions', []) or [], max_items=1, max_len=28)
                if easm:
                    line += f"|asm:{';'.join(easm)}"
                lines.append(line)
        return lines

    def _build_context(self) -> str:
        """Build complete context for the agent to see.
        
        Permanent context includes:
        - Investigation goal
        - Available graphs list
        - System architecture graph (always visible)
        - Memory notes (compressed history)
        - Recent actions
        
        Temporary data (appears only in action history):
        - Loaded graphs (other than system graph)
        - Node details and source code
        """
        # Refresh hypotheses from store to see updates from other agents
        self._load_existing_hypotheses()
        
        # Reload graphs to see updates from other agents
        self._refresh_loaded_graphs()
        
        context_parts = []
        
        # Investigation goal
        context_parts.append("=== INVESTIGATION GOAL ===")
        context_parts.append(self.investigation_goal)
        context_parts.append("")

        # User steering notes (if any)
        steering = self._read_steering_notes(limit=12)
        if steering:
            context_parts.append("=== USER STEERING (HONOR THESE) ===")
            for s in steering:
                # Add to memory notes lightly if marked as remember/note
                low = s.lower()
                if any(k in low for k in ("remember", "note:", "keep in mind")):
                    try:
                        if s not in self._steering_seen:
                            # Keep latest up to 5 steering memos
                            self.memory_notes.append(f"[STEER] {s[:160]}")
                            self._steering_seen.add(s)
                            # Bound memory notes size
                            if len(self.memory_notes) > 20:
                                self.memory_notes = self.memory_notes[-20:]
                    except Exception:
                        pass
                context_parts.append(f"• {s}")
            context_parts.append("")
        
        # Available graphs (show all)
        context_parts.append("=== AVAILABLE GRAPHS ===")
        context_parts.append("Use EXACT graph names as shown below:")
        for name in self.available_graphs.keys():
            if self.loaded_data['system_graph'] and name == self.loaded_data['system_graph']['name']:
                context_parts.append(f"• {name} [SYSTEM - AUTO-LOADED, see nodes below]")
            else:
                context_parts.append(f"• {name}")
        context_parts.append("")

        # Compressed memory notes (if any)
        if self.memory_notes:
            context_parts.append("=== MEMORY (COMPRESSED HISTORY) ===")
            for note in self.memory_notes[-5:]:
                context_parts.append(f"• {note}")
            context_parts.append("")
        
        # System graph - ALWAYS VISIBLE with ALL NODES/EDGES (compact)
        if self.loaded_data['system_graph']:
            context_parts.append("=== SYSTEM ARCHITECTURE (ALWAYS VISIBLE) ===")
            graph_name = self.loaded_data['system_graph']['name']
            graph_data = self.loaded_data['system_graph']['data']
            # Use unified formatting function
            context_parts.extend(self._format_graph_for_display(graph_data, graph_name))
        context_parts.append("")

        # Include any additionally loaded graphs in compact full form
        if self.loaded_data.get('graphs'):
            for gname, gdata in self.loaded_data['graphs'].items():
                context_parts.append(f"=== GRAPH LOADED: {gname} ===")
                context_parts.extend(self._format_graph_for_display(gdata, gname))
                context_parts.append("")

        # List currently loaded nodes to prevent reloading
        if self.loaded_data.get('nodes'):
            context_parts.append("=== LOADED NODES (CACHE — DO NOT RELOAD) ===")
            node_ids = list(self.loaded_data['nodes'].keys())
            # Print compact lists in lines of ~10
            line = []
            for i, nid in enumerate(node_ids, 1):
                line.append(nid)
                if (i % 10) == 0:
                    context_parts.append('  ' + ','.join(line))
                    line = []
            if line:
                context_parts.append('  ' + ','.join(line))
            context_parts.append("")
        
        # Actions performed (recent) - summary only since full data is in RECENT ACTIONS
        if self.action_log:
            context_parts.append("=== ACTIONS PERFORMED (SUMMARY) ===")
            for entry in self.action_log[-10:]:
                act = entry.get('action','-')
                r = entry.get('result','')
                # Just show action and brief result summary
                if isinstance(r, str):
                    rs = r[:100]
                else:
                    rs = str(r)[:100]
                context_parts.append(f"- {act}: {rs}")
            context_parts.append("")

        # Current hypotheses (ALWAYS show, display clearly to prevent duplicates)
        context_parts.append("=== EXISTING HYPOTHESES (DO NOT DUPLICATE!) ===")
        if self.loaded_data['hypotheses']:
            # Group by vulnerability type to make duplicates obvious
            by_type = {}
            for hyp in self.loaded_data['hypotheses']:
                vtype = hyp.get('vulnerability_type', 'unknown')
                if vtype not in by_type:
                    by_type[vtype] = []
                by_type[vtype].append(hyp)
            
            for vtype, hyps in by_type.items():
                context_parts.append(f"\n{vtype.upper()}:")
                for hyp in hyps:
                    status = hyp.get('status', 'proposed')
                    conf = hyp['confidence']
                    # Show full title and affected nodes to prevent duplicates
                    if status == 'confirmed':
                        icon = '✓'
                    elif status == 'rejected':
                        icon = '✗'
                    elif status == 'supported':
                        icon = '+'
                    elif status == 'refuted':
                        icon = '-'
                    else:
                        icon = '?'
                    
                    nodes = hyp.get('node_ids', [])
                    nodes_str = ','.join(nodes[:3]) if nodes else 'unknown'
                    title = hyp.get('title', hyp['description'][:60])
                    
                    # Single compact line per hypothesis with clear info
                    context_parts.append(f"  [{icon}] {conf:.0%} @{nodes_str}: {title}")
        else:
            context_parts.append("None")
        context_parts.append("")
        
        # Recent actions (for context awareness)
        # Show ALL conversation history (compression will handle size limits)
        if len(self.conversation_history) > 1:
            context_parts.append("=== RECENT ACTIONS ===")
            # Show all entries - compression handles size management
            for entry in self.conversation_history:
                if entry['role'] == 'assistant':
                    context_parts.append(f"Action: {entry['content']}")
                elif entry['role'] == 'system':
                    # Include full result for system responses (contains graph/node data)
                    content = entry['content']
                    # Mark compressed entries clearly
                    if content.startswith('[MEMORY]'):
                        context_parts.append(f"===== COMPRESSED HISTORY =====\n{content}")
                    else:
                        context_parts.append(f"Result: {content}")
            context_parts.append("")
        
        return '\n'.join(context_parts)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using accurate tokenization when available."""
        try:
            from llm.tokenization import count_tokens
            return count_tokens(text, self.llm.provider_name, self.llm.model)
        except Exception:
            try:
                return max(1, len(text) // 4)
            except Exception:
                return 0

    def _context_limit(self) -> int:
        """Return context token limit for the agent's model profile."""
        try:
            cfg = self.config or {}
            models = cfg.get('models', {}) if isinstance(cfg, dict) else {}
            prof = getattr(self, 'agent_profile', 'scout')
            mcfg = models.get(prof, {}) if isinstance(models, dict) else {}
            # Prefer model-specific max_context; fall back to global context.max_tokens
            return int(mcfg.get('max_context') or cfg.get('context', {}).get('max_tokens', 256000))
        except Exception:
            return 256000

    def _format_context_usage(self) -> str:
        """Build a concise usage string with context percent and last LLM call usage."""
        try:
            limit = self._context_limit()
            ctx = self._build_context()
            used = self._count_tokens(ctx)
            pct = min(100, int((used * 100) / max(1, limit)))
        except Exception:
            limit, used, pct = 0, 0, 0
        # Last LLM call usage (if any)
        try:
            last = get_token_tracker().get_last_usage() or {}
            prov = last.get('provider') or self.llm.provider_name
            model = last.get('model') or self.llm.model
            itok = int(last.get('input_tokens') or 0)
            otok = int(last.get('output_tokens') or 0)
            return f"Context {used}/{limit} tok ({pct}%) — Last call {prov}:{model} in={itok} out={otok}"
        except Exception:
            return f"Context {used}/{limit} tok ({pct}%)"

    def _maybe_compress_history(self):
        """Compress older conversation into memory notes when near context limit.
        
        Smart compression strategy:
        1. Compress when context reaches threshold (default 75% of max)
        2. Preserve recent actions (default last 5) in full detail
        3. Summarize older actions, focusing on:
           - Loaded graphs and their key findings
           - Formed hypotheses and confidence levels
           - Updated nodes with critical observations
           - Any errors or blockers encountered
        """
        # Get context management settings (global for both agent and guidance)
        context_cfg = (self.config or {}).get('context', {}) if isinstance(self.config, dict) else {}
        max_tokens = int(context_cfg.get('max_tokens', 128000))
        compression_threshold = float(context_cfg.get('compression_threshold', 0.75))
        keep_recent = int(context_cfg.get('keep_recent_actions', 5))
        
        # Calculate current context size using accurate tokenization when available
        try:
            current_context = self._build_context()
            current_tokens = self._count_tokens(current_context)
        except Exception:
            return
        
        # Check if compression is needed
        threshold_tokens = int(max_tokens * compression_threshold)
        if current_tokens < threshold_tokens:
            return  # Still have room
        
        # Log compression trigger
        print(f"\n[CONTEXT COMPRESSION] Triggered at {current_tokens}/{max_tokens} tokens ({current_tokens*100//max_tokens}% full)")
        print(f"[CONTEXT COMPRESSION] Compressing {len(self.conversation_history) - keep_recent} old entries, keeping {keep_recent} recent")
        
        if len(self.conversation_history) <= keep_recent + 1:
            print(f"[CONTEXT COMPRESSION] Not enough history to compress (only {len(self.conversation_history)} entries)")
            return  # Not enough history to compress
        
        # Split history into old (to compress) and recent (to keep)
        old_entries = self.conversation_history[:-keep_recent]
        recent_entries = self.conversation_history[-keep_recent:]
        
        # Intelligently extract key information from old entries
        graphs_loaded = set()
        nodes_analyzed = set()
        hypotheses_formed = []
        key_observations = []
        errors_encountered = []
        
        for entry in old_entries:
            role = entry.get('role', '')
            content = entry.get('content', '')
            
            if role == 'assistant':
                # Extract action type
                if 'load_graph' in content:
                    # Extract graph name from action
                    if 'Reasoning:' in content:
                        graphs_loaded.add(content.split('\n')[0].replace('Action: load_graph', '').strip())
            
            elif role == 'system':
                # Parse results
                if 'SUCCESS:' in content and 'Graph:' in content:
                    # Extract loaded graph info
                    for line in content.split('\n'):
                        if '--- Graph:' in line:
                            graph_name = line.split('Graph:')[1].split('---')[0].strip()
                            graphs_loaded.add(graph_name)
                        elif 'obs:' in line or 'assume:' in line:
                            # Capture important observations
                            key_observations.append(line.strip())
                            if len(key_observations) > 20:  # Keep only most important
                                key_observations = key_observations[-20:]
                
                elif 'nodes_display' in content or 'LOADED NODE DETAILS' in content:
                    # Extract analyzed nodes
                    for line in content.split('\n'):
                        if ' | ' in line and not line.startswith(' '):
                            node_id = line.split(' | ')[0].strip()
                            if node_id:
                                nodes_analyzed.add(node_id)
                
                elif '"status": "error"' in content or 'ERROR:' in content:
                    # Track errors
                    error_msg = content[:200]  # First 200 chars
                    if error_msg not in errors_encountered:
                        errors_encountered.append(error_msg)
                
                # Extract formed hypotheses
                if 'form_hypothesis' in content and 'success' in content:
                    hypotheses_formed.append("Hypothesis formed")
        
        # Build compressed summary
        summary_parts = []
        
        if graphs_loaded:
            summary_parts.append(f"Graphs analyzed: {', '.join(list(graphs_loaded)[:5])}")
        
        if nodes_analyzed:
            summary_parts.append(f"Nodes examined: {', '.join(list(nodes_analyzed)[:10])}")
        
        if hypotheses_formed:
            summary_parts.append(f"Hypotheses: {len(hypotheses_formed)} formed")
        
        if key_observations:
            # Include most recent key observations
            obs_summary = ' | '.join(key_observations[-5:])
            summary_parts.append(f"Key findings: {obs_summary[:300]}")
        
        if errors_encountered:
            summary_parts.append(f"Errors: {len(errors_encountered)} encountered")
        
        # Create final compressed note
        if summary_parts:
            summary_note = f"[Compressed {len(old_entries)} actions] " + " || ".join(summary_parts)
        else:
            # Fallback to simple compression
            summary_note = f"[Compressed {len(old_entries)} past actions]"
        
        # If we have a summarizer LLM, use it for better compression
        if self.summarizer and len(old_entries) > 10:
            try:
                # Prepare focused content for summarization
                important_content = []
                for entry in old_entries:
                    content = entry.get('content', '')
                    # Focus on results and key information
                    if 'SUCCESS:' in content or 'form_hypothesis' in content or 'obs:' in content:
                        important_content.append(content[:1000])  # First 1000 chars of important entries
                
                if important_content:
                    sys_p = """Summarize the key findings from this security audit into 5-8 bullets:
                    - Which graphs and nodes were analyzed
                    - What vulnerabilities or hypotheses were formed
                    - Critical observations or assumptions made
                    - Any errors or blockers encountered
                    Keep only the most important facts for continuing the audit."""
                    
                    user_p = '\n'.join(important_content[:30])  # Limit input size
                    resp = self.summarizer.raw(system=sys_p, user=user_p)
                    
                    if resp:
                        lines = [ln.strip('-• ') for ln in resp.splitlines() if ln.strip()]
                        if lines:
                            summary_note = "[AI-Compressed history] " + ' | '.join(lines[:8])
            except Exception:
                pass  # Keep the heuristic summary
        
        # Update memory and conversation history
        self.memory_notes.append(summary_note)
        
        # Keep only the compressed summary and recent entries
        self.conversation_history = [
            {'role': 'system', 'content': f"[MEMORY] {summary_note}"}
        ] + recent_entries
        
        # Also clear old entries from loaded_data to free memory (except system graph)
        # But keep the critical findings in memory notes
        graphs_cleared = 0
        if len(self.loaded_data.get('graphs', {})) > 3:
            # Keep only most recently loaded graphs
            graph_items = list(self.loaded_data['graphs'].items())
            graphs_cleared = len(self.loaded_data['graphs']) - 3
            self.loaded_data['graphs'] = dict(graph_items[-3:])
        
        nodes_cleared = 0
        if len(self.loaded_data.get('nodes', {})) > 10:
            # Keep only most recently loaded nodes
            node_items = list(self.loaded_data['nodes'].items())
            nodes_cleared = len(self.loaded_data['nodes']) - 10
            self.loaded_data['nodes'] = dict(node_items[-10:])
        
        # Log compression completion
        print(f"[CONTEXT COMPRESSION] Complete! Compressed {len(old_entries)} entries into memory note")
        print(f"[CONTEXT COMPRESSION] Summary: {summary_note[:200]}...")
        if graphs_cleared or nodes_cleared:
            print(f"[CONTEXT COMPRESSION] Cleared {graphs_cleared} old graphs and {nodes_cleared} old nodes from memory")
        
        # Recalculate tokens after compression using accurate counting
        try:
            new_context = self._build_context()
            new_tokens = self._count_tokens(new_context)
            print(f"[CONTEXT COMPRESSION] New context size: {new_tokens}/{max_tokens} tokens ({new_tokens*100//max_tokens}% full)")
        except Exception:
            pass  # Don't fail if we can't calculate new size
    
    def _get_agent_decision(self, context: str) -> AgentDecision:
        """
        Get agent's structured decision based on context.
        Uses provider-appropriate method for reliable parsing.
        """
        system_prompt = """You are an autonomous security investigation agent analyzing smart contracts.

YOUR CORE RESPONSIBILITY: You are the EXPLORER and CONTEXT BUILDER. Your primary job is to:
- Navigate and explore the graph structure to understand the system
- Load relevant code that implements the features being investigated  
- Build comprehensive context by examining multiple related components
- Prepare complete information for the deep think model to analyze

The deep think model (guidance) is a separate, expensive reasoning engine that performs vulnerability analysis.
It can ONLY analyze the context you prepare - if you don't load it, it can't analyze it!

SEPARATION OF ROLES (IMPORTANT):
- Scout (you) gathers code and facts; annotate graphs with short observations/assumptions.
- NEVER speculate about vulnerabilities and do NOT adjudicate them yourself.
- Prefer deep_think for vulnerability analysis; use form_hypothesis only if you must capture a lead and cannot escalate yet.
- After assembling a coherent slice of code for the current investigation, CALL deep_think. Do not call deep_think based on hunches; call it when relevant code for the investigation is collected. The Strategist will analyze your prepared context and can surface additional vulnerabilities beyond the current goal if supported by evidence in the context.

Your task is to investigate the system and identify potential vulnerabilities. The system architecture graph is automatically loaded and visible. You can see all available graphs and which are loaded.

OPERATING CONSTRAINTS (IMPORTANT):
- Hound cannot run code, connect to RPC, fork a chain, deploy contracts, or query on-chain state.
- Do NOT propose or assume live on-chain probing (e.g., calling initialize on proxies, running fork-based tests, deploying mocks).
- All actions here are CODE-ONLY: loading/reading nodes, updating observations, and calling deep_think for analysis. Keep guidance and reasoning aligned with this constraint.

DEDUPLICATION (IMPORTANT):
- Review the section "LOADED NODES (CACHE — DO NOT RELOAD)" in Current Context.
- Do NOT request nodes that appear there; pick NEW nodes to expand coverage and avoid repeated loads.
- If a node is already loaded, prefer loading closely-connected nodes instead (callers/callees/storage it reads/writes).

CRITICAL RULES FOR NODE AND GRAPH NAMES:
- ALWAYS use EXACT node IDs as shown in the NODES sections (the token before the first pipe) or listed under LOADED NODES
- NEVER guess, modify, or create node names
- NEVER add prefixes like "func_" or "node_" unless they're already there
- If a node doesn't exist in the graph, DON'T try variations - it doesn't exist!
- Check the graph FIRST to see what nodes actually exist before requesting them

USER STEERING (PRIORITY):
- If a "USER STEERING" section appears in Current Context, you MUST honor it immediately.
- If instructed to "investigate/check X next", prioritize loading relevant graphs/nodes for X as your next action.
- If asked to "remember/keep in mind" a constraint, treat it as a memory constraint in your reasoning.
- Do NOT ignore steering unless it is impossible under constraints.

IMPORTANT DISTINCTION:
- Graph observations/assumptions: Facts about HOW the system works (invariants, behaviors, constraints)
- Hypotheses: Suspected SECURITY ISSUES or vulnerabilities
Never mix these - security concerns always go in hypotheses, not in graph updates.

WHEN ADDING OBSERVATIONS/ASSUMPTIONS:
Keep EXTREMELY SHORT - just essential facts, not full sentences:
- Good: "only owner", "checks balance", "emits Transfer", "immutable", "reentrancy guard"
- Bad: "This function can only be called by the owner of the contract"
- Bad: "The function checks that the balance is greater than zero before proceeding"

AVAILABLE ACTIONS - USE EXACT PARAMETERS AS SHOWN:

1. load_graph — Load an additional graph for analysis
   PARAMETERS: {"graph_name": "GraphName"}
   EXAMPLE: {"graph_name": "AuthorizationRoles"}
   EXAMPLE: {"graph_name": "DataFlowDiagram"}
   ONLY SEND: graph_name - NOTHING ELSE!

2. load_nodes — Load source code for specific nodes from a specific graph
   PARAMETERS: {"graph_name": "ExactGraphName", "node_ids": ["exact_node_id_from_brackets"]}
   REQUIRED: graph_name (string) AND node_ids (array)
   COPY THE EXACT NODE IDs from the square brackets [like_this] in the graph display
   
   LOADING STRATEGY:
   - PRIORITIZE nodes marked [S] (small) and [M] (medium) - these are targeted functions
   - AVOID nodes marked [L:n] (large) - these are entire contracts with many code blocks
   - Load specific functions (func_*) rather than entire contracts (contract_*)
   - If you must load a large node, explain WHY it's necessary
   
   CORRECT EXAMPLE: {"graph_name": "SystemArchitecture", "node_ids": ["func_AIToken_mint"]}
   WRONG EXAMPLE: {"graph_name": "System", "node_ids": ["contract_Agent"]} ← entire contract!
   
   The node IDs are shown in square brackets. Size indicators show code volume.

AFTER LOADING CODE (WHEN CLEAR SIGNALS EXIST):
- Add 1-3 ultra-short annotations via update_node for the most relevant loaded node(s).
- Observations: concrete behaviors found in code (e.g., "external call", "emits Event", "nonReentrant", "unchecked", "token transfer").
- Assumptions: invariants/constraints (e.g., "onlyOwner", "onlyRole(Role)", "pausable", "initializer", "immutable var").
- Keep each bullet 2-4 words. Do not write prose. Skip if truly nothing notable.

3. update_node — Add observations/assumptions about ONE node
   PARAMETERS: {"node_id": "node", "observations": [...], "assumptions": [...]}
   EXAMPLE: {"node_id": "ProxyAdmin", "observations": ["single admin", "no timelock"]}
   EXAMPLE: {"node_id": "func_transfer", "assumptions": ["checks balance"]}
   ONLY SEND: node_id (required), observations (optional), assumptions (optional)
   DO NOT SEND: Empty arrays [] - omit the field instead
   Keep observations/assumptions VERY SHORT (2-4 words each)

4. update_hypothesis — Update existing hypothesis with new evidence
   PARAMETERS: {"hypothesis_index": 0, "new_confidence": 0.5, "evidence": "..."}
   EXAMPLE: {"hypothesis_index": 0, "new_confidence": 0.9, "evidence": "Confirmed by analyzing implementation"}
   ONLY SEND: hypothesis_index, new_confidence, evidence - NOTHING ELSE!

5. deep_think — Analyze recent exploration for vulnerabilities (EXPENSIVE - use wisely!)
   PARAMETERS: {}
   EXAMPLE: {}
   Send empty object {} - NO PARAMETERS!
   WHEN TO CALL:
   - After you have loaded enough specific code (functions/files) to represent the investigation focus.
   - Do NOT call because you "suspect" issues; call when the relevant code for the investigation has been collected.
   The Strategist will return ANY vulnerabilities detected from the prepared context (not limited to the goal) and will filter out weak/false positives.
   CRITICAL PREREQUISITES - DO NOT CALL deep_think UNTIL:
   - You have loaded RELEVANT graphs for the investigation topic
   - You have loaded ACTUAL CODE (nodes) that implements the feature being investigated
   - You have made observations about the loaded code
   - You have a COMPLETE VIEW of the feature/subsystem being analyzed
   
   NEVER call deep_think:
   - At the start of investigation (no context loaded yet!)
   - After only loading graphs without loading any nodes
   - When you haven't explored the specific feature mentioned in the investigation
   
   ONLY call deep_think:
   - After loading and examining 5-10 relevant nodes minimum
   - When you have a complete understanding of a subsystem
   - When you need strategic guidance after thorough exploration
   
   Purpose: The deep think model performs expensive, thorough vulnerability analysis
   on the context YOU have prepared. It can only analyze what you've loaded!

6. complete — Finish the current investigation
   PARAMETERS: {}
   EXAMPLE: {}
   Send empty object {} - NO PARAMETERS!

YOUR PRIMARY ROLE - CONTEXT PREPARATION:
You are the NAVIGATOR and EXPLORER. Your job is to:
1. Navigate the graph structure to find relevant components
2. Load and examine code that implements the investigated feature
3. Build a complete understanding of how the system works
4. PREPARE comprehensive context for the deep think model to analyze

The deep think model is EXPENSIVE and can only analyze what YOU load!
Think of yourself as preparing a detailed case file for an expert analyst.

EXPLORATION STRATEGY:
1. UNDERSTAND the investigation goal - what feature/property are we examining?
2. LOAD relevant graphs that show this feature's structure
3. IDENTIFY nodes that implement this feature (check size indicators!)
4. LOAD the actual code (5-10+ nodes minimum) for these components
5. MAKE observations about how the code works
6. ONLY THEN call deep_think when you have a COMPLETE picture
7. Follow deep_think's guidance to explore related areas
8. Repeat: thorough exploration → deep_think → more exploration

SMART LOADING: Load func_* nodes (specific functions) rather than contract_* nodes (entire files)!
REMEMBER: deep_think can only analyze what you've loaded - incomplete context = incomplete analysis!

COMPLETION CRITERIA (WHEN TO CALL complete):
1. You have explored key areas AND deep_think has analyzed them for vulnerabilities, OR
2. Further exploration is unlikely to reveal new important information, OR
3. No promising exploration paths remain.

IMPORTANT: 
- Do NOT form hypotheses directly - that's deep_think's job
- NEVER call deep_think without loading substantial code context first (5-10+ nodes minimum)
- Deep_think is EXPENSIVE and analyzes YOUR discoveries - incomplete prep = wasted analysis
- Your role: EXPLORE thoroughly, LOAD relevant code, BUILD complete context
- Only call deep_think when you have a COMPLETE understanding of the investigated feature

EXPECTATIONS:
- Choose nodes at the most informative granularity (functions/storage) when available.
- Avoid loading entire contracts by default; only do so when specifically necessary.
- Be explicit in your reasoning about why each selected node advances the goal.

IMPORTANT JSON FORMATTING RULES:
- NEVER use null values - omit the field entirely if not needed
- Only include parameters that are REQUIRED for the specific action
- Do NOT include empty arrays [] or null - omit the field
- Each action has SPECIFIC required parameters - only include those

Return a JSON object with: action, reasoning, parameters"""
        
        user_prompt = f"""Current Context:

{context}

What is your next action? Respond ONLY with a valid JSON object in this exact format:
{{
  "action": "action_name",
  "reasoning": "why you are taking this action",
  "parameters": {{...action-specific parameters...}}
}}

DO NOT include any text before or after the JSON object."""
        
        # Use raw JSON output - works across all providers
        response = None  # Ensure defined even if the provider call raises
        try:
            # First try raw call with JSON instruction
            response = self.llm.raw(
                system=system_prompt,
                user=user_prompt
            )
            
            # Parse the JSON response
            if response:
                # Clean response - remove markdown code blocks if present
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]  # Remove ```json
                if response.startswith('```'):
                    response = response[3:]  # Remove ```
                if response.endswith('```'):
                    response = response[:-3]  # Remove trailing ```
                response = response.strip()
                
                # Parse JSON
                data = json.loads(response)
                # Ensure parameters is a dict
                if 'parameters' not in data:
                    data['parameters'] = {}
                elif data['parameters'] is None:
                    data['parameters'] = {}
                return AgentDecision(**data)
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"[!] JSON parsing failed: {e}")
            # Fallback to more robust parsing
            if response:
                try:
                    from .parsing import parse_agent_decision_fallback
                    data = parse_agent_decision_fallback(response)
                    if isinstance(data, dict):
                        if 'parameters' not in data or data['parameters'] is None:
                            data['parameters'] = {}
                        return AgentDecision(**data)
                    # Try to extract action and reasoning manually
                    import re
                    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response)
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)
                    if action_match:
                        return AgentDecision(
                            action=action_match.group(1),
                            reasoning=reasoning_match.group(1) if reasoning_match else "Parsed from malformed JSON",
                            parameters={}
                        )
                except Exception as e2:
                    print(f"[!] Failed to parse response: {e2}")
            
            # Ultimate fallback - make a reasonable decision
            if not self.loaded_data['nodes']:
                # Look for critical nodes in the system graph
                critical_nodes = []
                if self.loaded_data['system_graph']:
                    graph_data = self.loaded_data['system_graph']['data']
                    for node in graph_data.get('nodes', [])[:5]:  # First 5 nodes
                        critical_nodes.append(node['id'])
                
                return AgentDecision(
                    action="load_nodes",
                    reasoning="Need to load node data to analyze code",
                    parameters={"node_ids": critical_nodes}
                )
            else:
                return AgentDecision(
                    action="complete",
                    reasoning="Unable to parse response, completing",
                    parameters={}
                )
    
    def _execute_action(self, decision: AgentDecision) -> dict:
        """Execute the agent's decision."""
        action = decision.action
        params = decision.parameters  # Now a dict
        
        if action == 'deep_think':
            # Execute deep thinking analysis
            return self._deep_think()
        elif action == 'load_graph':
            # Extract graph_name
            graph_name = params.get('graph_name', '')
            if isinstance(graph_name, str):
                # Remove any JSON artifacts
                graph_name = graph_name.strip().rstrip("'}").rstrip('"').rstrip("'")
            return self._load_graph(graph_name)
        elif action == 'load_nodes':
            # Extract graph_name and node_ids
            graph_name = params.get('graph_name')
            node_ids = params.get('node_ids', [])
            return self._load_nodes(node_ids, graph_name)
        elif action == 'update_node':
            # Pass only relevant parameters
            clean_params = {
                'node_id': params.get('node_id')
            }
            if 'observations' in params:
                clean_params['observations'] = params['observations']
            if 'assumptions' in params:
                clean_params['assumptions'] = params['assumptions']
            return self._update_node(clean_params)
        elif action == 'form_hypothesis':
            # Pass hypothesis parameters
            return self._form_hypothesis(params)
        elif action == 'update_hypothesis':
            # Pass update parameters
            return self._update_hypothesis(params)
        elif action == 'complete':
            return {'status': 'complete', 'summary': 'Investigation complete'}
        else:
            return {'status': 'error', 'error': f'Unknown action: {action}'}

    
    
    def _load_graph(self, graph_name: str) -> dict:
        """Load an additional knowledge graph.
        
        The graph data is returned in the action response (appearing in history)
        rather than being added to permanent context. Only the system graph
        remains permanently visible in context.
        """
        # Clean up graph name (remove quotes, trailing characters, JSON artifacts)
        if graph_name:
            # Remove common JSON/text artifacts
            graph_name = graph_name.strip().strip("'\"")
            # Split on common delimiters that shouldn't be in graph names
            for delimiter in ["'", '"', '}', ')', '\n', ' ']:
                if delimiter in graph_name:
                    graph_name = graph_name.split(delimiter)[0]
            graph_name = graph_name.strip()
        
        if not graph_name:
            return {'status': 'error', 'error': 'Graph name is required'}
        
        # Try to find the graph (case-insensitive match as fallback)
        if graph_name not in self.available_graphs:
            # Try case-insensitive match
            for available_name in self.available_graphs.keys():
                if available_name.lower() == graph_name.lower():
                    graph_name = available_name
                    break
            else:
                # Suggest similar names
                available = list(self.available_graphs.keys())
                return {'status': 'error', 'error': f'Graph not found: {graph_name}. Available: {", ".join(available)}'}
        
        # Don't reload the system graph
        if self.loaded_data['system_graph'] and graph_name == self.loaded_data['system_graph']['name']:
            return {
                'status': 'info',
                'summary': f'{graph_name} is already loaded as the system graph'
            }
        
        try:
            # Use concurrent-safe reload method
            graph_data = self._reload_graph(graph_name)
            if not graph_data:
                return {'status': 'error', 'error': f'Failed to load graph {graph_name}'}
            
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            # Store in loaded graphs so it appears in context
            self.loaded_data['graphs'][graph_name] = graph_data

            # Format the graph for display using unified function
            formatted_lines = self._format_graph_for_display(graph_data, graph_name)
            graph_display = '\n'.join(formatted_lines)
            
            return {
                'status': 'success',
                'summary': f'Loaded {graph_name}: {len(nodes)} nodes, {len(edges)} edges',
                'graph_display': graph_display
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    @property
    def cards(self):
        """Access card index, loading if needed."""
        self._ensure_card_index()
        return self._card_index or {}
    
    def _iterate_graphs(self):
        """Iterate over all loaded graphs (system + additional)."""
        # System graph - use actual graph name instead of 'system'
        if self.loaded_data.get('system_graph'):
            graph_name = self.loaded_data['system_graph']['name']
            yield graph_name, self.loaded_data['system_graph']['data']
        
        # Additional loaded graphs
        yield from self.loaded_data.get('graphs', {}).items()
    
    def _ensure_card_index(self):
        """Load and cache cards.jsonl as an index by ID."""
        if self._card_index is not None:
            return
        from .cards import load_card_index
        idx, file_map = load_card_index(self.graphs_metadata_path, self.manifest_path)
        self._card_index = idx
        self._file_to_cards.update(file_map)

    def _extract_card_content(self, card: dict[str, Any]) -> str:
        """Get best-available content from a card record."""
        from .cards import extract_card_content
        return extract_card_content(card, self._repo_root)

    # Note: _iter_graphs was a duplicate of _iterate_graphs and has been removed.

    def _load_nodes(self, node_ids: list[str], graph_name: str | None = None) -> dict:
        """Load complete node data with associated source code.
        
        Node details and code are returned in the action response (appearing in history)
        rather than being added to permanent context. This allows the agent to explore
        nodes without permanently filling the context window.
        
        Args:
            node_ids: List of node IDs to load
            graph_name: REQUIRED - The specific graph to load nodes from
        """
        if not graph_name:
            return {
                'status': 'error',
                'error': 'graph_name is REQUIRED. Please specify which graph contains the nodes you want to load.'
            }
            
        if not node_ids:
            return {
                'status': 'error',
                'error': 'No node IDs specified. Please specify which nodes to load from the graph.'
            }
        
        self._ensure_card_index()
        
        # First check if the graph exists and is loaded
        graph_data = None
        graph_edges = []
        
        # Check system graph
        if self.loaded_data.get('system_graph') and self.loaded_data['system_graph']['name'] == graph_name:
            graph_data = self.loaded_data['system_graph']['data']
        # Check loaded graphs
        elif graph_name in (self.loaded_data.get('graphs') or {}):
            graph_data = self.loaded_data['graphs'][graph_name]
        else:
            # Graph not loaded - provide helpful error
            available = []
            if self.loaded_data.get('system_graph'):
                available.append(self.loaded_data['system_graph']['name'])
            available.extend(self.loaded_data.get('graphs', {}).keys())
            return {
                'status': 'error',
                'error': f'Graph "{graph_name}" is not loaded. Available graphs: {", ".join(available) if available else "none"}. Use load_graph first if needed.'
            }
        
        # Build index for ONLY the specified graph
        node_by_id: dict[str, dict[str, Any]] = {}
        graph_edges = graph_data.get('edges', [])
        
        for n in graph_data.get('nodes', []):
            nid = n.get('id')
            if nid:
                node_by_id[nid] = n

        not_found: list[str] = []
        loaded_nodes = []

        for req_id in node_ids:
            # EXACT match only - no fuzzy matching
            ndata = node_by_id.get(req_id)
            
            if not ndata:
                not_found.append(req_id)
                continue
            
            chosen_id = req_id
            
            # Track large nodes for warning in display, but don't print here
            ndata.get('source_refs', []) or []
            # Will show warning in display_lines instead of printing

            # Collect evidence cards from node and its incident edges
            # We already have graph_edges from the specified graph
            card_ids: list[str] = []
            node_refs = ndata.get('source_refs', []) or ndata.get('refs', []) or []
            if isinstance(node_refs, list):
                card_ids.extend([str(x) for x in node_refs])
            
            # Debug logging removed - too noisy
            for e in graph_edges:
                src = e.get('source_id') or e.get('source') or e.get('src')
                dst = e.get('target_id') or e.get('target') or e.get('dst')
                if src == chosen_id or dst == chosen_id:
                    evid = e.get('evidence', []) or e.get('source_refs', []) or []
                    if isinstance(evid, list):
                        card_ids.extend([str(x) for x in evid])

            # Dedup
            seen = set()
            base_ids = []
            for cid in card_ids:
                if cid and cid not in seen:
                    seen.add(cid)
                    base_ids.append(cid)

            # Resolve cards ordered by relpath + char_start
            node_cards: list[dict[str, Any]] = []
            ordered = []
            for cid in base_ids:
                c = self._card_index.get(cid)
                if c:
                    ordered.append(c)
                # Card not found - continue silently
                pass
            
            # Debug logging removed
            
            ordered.sort(key=lambda x: (x.get('relpath') or '', x.get('char_start') or 0))
            # Reuse cached code blocks for this node if present
            cached_cards = self.loaded_data.get('nodes', {}).get(chosen_id, {}).get('cards') if isinstance(self.loaded_data.get('nodes'), dict) else None
            if cached_cards:
                node_cards = cached_cards
            else:
                for c in ordered:
                    cid = c.get('id')
                    content = self._extract_card_content(c)
                    node_cards.append({
                        'card_id': cid,
                        'type': c.get('type', 'code'),
                        'content': content,
                        'metadata': {
                            k: c.get(k) for k in (
                                'relpath','char_start','char_end','line_start','line_end'
                            ) if k in c
                        }
                    })
                    # Track card coverage
                    try:
                        if self.coverage_index and cid:
                            self.coverage_index.touch_card(str(cid))
                    except Exception:
                        pass

            # NO FALLBACK - if node has no explicit source_refs, it has no code
            # This prevents loading entire files when agent requests non-existent nodes

            node_copy = ndata.copy()
            if node_cards:
                node_copy['cards'] = node_cards
                try:
                    self.loaded_data['code'][chosen_id] = '\n\n'.join((c.get('content') or '') for c in node_cards if isinstance(c, dict))
                except Exception:
                    pass
            self.loaded_data['nodes'][chosen_id] = node_copy
            loaded_nodes.append(chosen_id)
            try:
                if self.coverage_index:
                    self.coverage_index.touch_node(chosen_id)
            except Exception:
                pass

        # Count only the nodes from this request
        current_request_nodes = loaded_nodes
        current_loaded_count = len(loaded_nodes)
        current_code_count = len([nid for nid in loaded_nodes if 'cards' in self.loaded_data['nodes'][nid]])
        
        # Total across all previous loads (for context)
        total_loaded_count = len(self.loaded_data['nodes'])
        
        # Format loaded nodes for display
        display_lines = []
        display_lines.append(f"\n=== LOADED NODE DETAILS FROM {graph_name} ===")
        display_lines.append(f"This request: {current_loaded_count} nodes ({current_code_count} with code)")
        
        # Check if any large nodes were loaded
        large_nodes_loaded = []
        for nid in loaded_nodes:
            node_data = self.loaded_data['nodes'][nid]
            if 'cards' in node_data and len(node_data['cards']) > 6:
                large_nodes_loaded.append(f"{nid}({len(node_data['cards'])} blocks)")
        
        if large_nodes_loaded:
            display_lines.append(f"⚠️ WARNING: Loaded LARGE nodes: {', '.join(large_nodes_loaded)}")
            display_lines.append("Consider loading specific functions instead of entire contracts!")
        
        if total_loaded_count > current_loaded_count:
            display_lines.append(f"Total cached: {total_loaded_count} nodes\n")
        else:
            display_lines.append("")
        
        # Track duplicate code blocks across this request to avoid re-printing
        printed_card_ids: set[str] = set()
        dedup_count = 0

        for node_id in current_request_nodes:
            node_data = self.loaded_data['nodes'][node_id]
            node_type = node_data.get('type', 'unknown')
            node_label = node_data.get('label', node_id)
            display_lines.append(f"{node_id} | {node_label} | {node_type}")
            
            # Show observations
            observations = node_data.get('observations', [])
            if observations:
                obs_strs = []
                for obs in observations[:5]:  # First 5
                    if isinstance(obs, dict):
                        desc = obs.get('description', obs.get('content', str(obs)))
                        obs_strs.append(desc)
                    else:
                        obs_strs.append(str(obs))
                if obs_strs:
                    display_lines.append(f"  obs: {'; '.join(obs_strs)}")
            
            # Show assumptions
            assumptions = node_data.get('assumptions', [])
            if assumptions:
                assum_strs = []
                for assum in assumptions[:3]:  # First 3
                    if isinstance(assum, dict):
                        desc = assum.get('description', assum.get('content', str(assum)))
                        assum_strs.append(desc)
                    else:
                        assum_strs.append(str(assum))
                if assum_strs:
                    display_lines.append(f"  assume: {'; '.join(assum_strs)}")
            
            # Show FULL code if present - agent needs to see everything for analysis
            if 'cards' in node_data and node_data['cards']:
                display_lines.append(f"  === CODE ({len(node_data['cards'])} blocks) ===")
                for i, card in enumerate(node_data['cards']):
                    content = card.get('content', '')
                    card_id = card.get('card_id') or (card.get('metadata') or {}).get('id')
                    if content:
                        card_type = card.get('type', 'code')
                        metadata = card.get('metadata', {})
                        relpath = metadata.get('relpath', 'unknown')
                        line_start = metadata.get('line_start', '?')
                        line_end = metadata.get('line_end', '?')
                        if card_id and card_id in printed_card_ids:
                            dedup_count += 1
                            display_lines.append(
                                f"  --- Block {i+1} ({card_type}) from {relpath}:{line_start}-{line_end} — duplicate, omitted (card {card_id}) ---"
                            )
                        else:
                            if card_id:
                                printed_card_ids.add(card_id)
                            display_lines.append(f"  --- Block {i+1} ({card_type}) from {relpath}:{line_start}-{line_end} ---")
                            # Show FULL content - no truncation
                            for line in content.split('\n'):
                                display_lines.append(f"    {line}")
                            display_lines.append("")  # Empty line between code blocks
            
            display_lines.append("")  # Empty line between nodes
        
        if not_found:
            display_lines.append(f"\n⚠️ ERROR - These nodes do not exist in {graph_name}: {', '.join(not_found)}")
            display_lines.append(f"Available nodes in {graph_name}:")
            # Show first 10 available nodes as examples
            available_nodes = list(node_by_id.keys())[:10]
            for node_id in available_nodes:
                display_lines.append(f"  • {node_id}")
            if len(node_by_id) > 10:
                display_lines.append(f"  ... and {len(node_by_id) - 10} more")
            display_lines.append("\nUse EXACT node IDs as shown above. Do not guess or modify node names!")
        
        if dedup_count > 0:
            display_lines.insert(0, f"[dedup] Omitted {dedup_count} duplicate code block(s) already shown in this request.")
        nodes_display = '\n'.join(display_lines)
        
        # Aggregate card IDs across loaded nodes
        all_card_ids = []
        for nid in current_request_nodes:
            node_data = self.loaded_data['nodes'].get(nid, {})
            for c in node_data.get('cards', []) or []:
                cid = c.get('card_id') or (c.get('metadata') or {}).get('id')
                if cid:
                    all_card_ids.append(str(cid))
        # Deduplicate
        all_card_ids = list({cid for cid in all_card_ids if cid})

        return {
            'status': 'success',
            'summary': f'Loaded {current_loaded_count} nodes ({current_code_count} with code)',
            'nodes_display': nodes_display,
            'loaded_node_ids': current_request_nodes,
            'card_ids': all_card_ids
        }
    
    
    def _save_graph_updates(self, graph_name: str, graph_data: dict):
        """Save graph updates back to disk using concurrent-safe GraphStore."""
        try:
            if graph_name in self.available_graphs:
                graph_path = Path(self.available_graphs[graph_name]['path'])
                
                # Use GraphStore for atomic save with built-in locking
                graph_store = GraphStore(graph_path, agent_id=self.agent_id)
                return graph_store.save_graph(graph_data)
                        
        except Exception as e:
            print(f"[!] Failed to save graph {graph_name}: {e}")
            return False
    
    def _reload_graph(self, graph_name: str):
        """Reload a graph from disk using concurrent-safe GraphStore."""
        try:
            if graph_name in self.available_graphs:
                graph_path = Path(self.available_graphs[graph_name]['path'])
                
                # Use GraphStore for atomic read with built-in locking
                graph_store = GraphStore(graph_path, agent_id=self.agent_id)
                return graph_store.load_graph()
                    
        except Exception as e:
            print(f"[!] Failed to reload graph {graph_name}: {e}")
            return None
    
    def _update_node(self, params: dict) -> dict:
        """Update a node with observations or assumptions about its behavior."""
        node_id = params.get('node_id')
        if not node_id:
            # Check if user mistakenly passed node_ids (plural)
            if params.get('node_ids'):
                return {'status': 'error', 'error': 'update_node requires node_id (singular), not node_ids. Update one node at a time.'}
            return {'status': 'error', 'error': 'node_id is required for update_node action'}
        
        # First refresh all loaded graphs to get latest updates from other agents
        self._refresh_loaded_graphs()
        
        # Check if node exists in loaded data or graphs
        found = False
        if node_id in self.loaded_data.get('nodes', {}):
            found = True
        else:
            # Try to find it in graphs (nodes are stored as a list)
            for graph_name, graph_data in self._iterate_graphs():
                nodes = graph_data.get('nodes', [])
                for node in nodes:
                    if node.get('id') == node_id:
                        found = True
                        break
                if found:
                    break
            
        if not found:
            return {'status': 'error', 'error': f'Node {node_id} not found in any loaded graph'}
        
        # Update the node in the graph(s)
        updated_graphs = []
        observations = params.get('observations') or []
        assumptions = params.get('assumptions') or []
        
        for graph_name, graph_data in self._iterate_graphs():
            nodes = graph_data.get('nodes', [])
            for node in nodes:
                if node.get('id') != node_id:
                    continue
                
                # Initialize fields if not present
                if 'observations' not in node:
                    node['observations'] = []
                if 'assumptions' not in node:
                    node['assumptions'] = []
                
                # Add new observations (simplified - strings only as per prompt)
                for obs in observations:
                    if isinstance(obs, str):
                        node['observations'].append(obs)
                    elif isinstance(obs, dict):
                        # If dict provided, extract description
                        node['observations'].append(obs.get('description', str(obs)))
                
                # Add new assumptions (simplified - strings only as per prompt)
                for assum in assumptions:
                    if isinstance(assum, str):
                        node['assumptions'].append(assum)
                    elif isinstance(assum, dict):
                        # If dict provided, extract description
                        node['assumptions'].append(assum.get('description', str(assum)))
                
                # Save the updated graph to disk for sharing
                self._save_graph_updates(graph_name, graph_data)
                
                updated_graphs.append(graph_name)
                break  # Found and updated the node
        
        if not updated_graphs:
            return {'status': 'error', 'error': f'Node {node_id} not found in any loaded graph'}
        
        obs_count = len(observations)
        assum_count = len(assumptions)
        
        return {
            'status': 'success',
            'summary': f"Updated node {node_id}: {obs_count} observations, {assum_count} assumptions",
            'graphs_updated': updated_graphs
        }
    
    def _deep_think(self) -> dict:
        """Delegate deep analysis to the Strategist and form hypotheses accordingly."""
        try:
            # Removed hard minimum-context guardrail: different aspects require different
            # amounts of context. We rely on the Scout's prompt to prepare adequate evidence
            # before escalation instead of enforcing a strict numeric threshold here.
            context = self._build_context()
            from .strategist import Strategist
            # Pass debug and session_id to strategist for deep_think prompt saving
            strategist = Strategist(
                config=self.config or {}, 
                debug=self.debug, 
                session_id=self.session_id,
                debug_logger=getattr(self, 'debug_logger', None),
                mission=getattr(self, 'mission', None)
            )
            # Pass phase if available (from parent runner)
            phase = getattr(self, 'current_phase', None)
            if phase == 'Coverage':
                items = strategist.deep_think(context=context, phase='Coverage') or []
            elif phase == 'Saliency':
                items = strategist.deep_think(context=context, phase='Saliency') or []
            else:
                # Auto-detect based on context or default
                items = strategist.deep_think(context=context) or []
            added = 0
            dedup_skipped = 0
            dedup_details: list[str] = []
            other_skipped: list[str] = []
            titles: list[str] = []
            hyp_info: list[dict] = []
            guidance_model_info = None
            if hasattr(self, 'guidance_client') and self.guidance_client:
                try:
                    guidance_model_info = f"{self.guidance_client.provider_name}:{self.guidance_client.model}"
                except Exception:
                    guidance_model_info = None
            # Prepare existing hypotheses for LLM-based dedup in batches
            try:
                existing_hyps = self.hypothesis_store.list_all()
            except Exception:
                existing_hyps = []
            for it in items:
                params = {
                    'description': it.get('description', 'Hypothesis'),
                    'details': it.get('details', ''),
                    'vulnerability_type': it.get('vulnerability_type', 'security_issue'),
                    'severity': it.get('severity', 'medium'),
                    'confidence': it.get('confidence', 0.6),
                    'node_ids': it.get('node_ids', ['system']),
                    'reasoning': it.get('reasoning', ''),
                    'graph_name': 'SystemArchitecture',
                    'guidance_model': guidance_model_info,
                }
                # LLM-assisted deduplication against existing hypotheses in batches of 20
                try:
                    from .hypothesis_dedup import check_duplicates_llm
                    # Build compact candidate for dedup model
                    new_candidate = {
                        'id': 'new_candidate',
                        'title': params.get('description', ''),
                        'description': params.get('details') or params.get('description', ''),
                        'vulnerability_type': params.get('vulnerability_type', 'security_issue'),
                        'node_refs': params.get('node_ids') or [],
                    }
                    is_dup = False
                    found_dup_ids: list[str] | None = None
                    batch_size = 20
                    for i in range(0, len(existing_hyps), batch_size):
                        batch = existing_hyps[i:i+batch_size]
                        dup_ids = check_duplicates_llm(
                            cfg=self.config or {},
                            new_hypothesis=new_candidate,
                            existing_batch=batch,
                            debug_logger=getattr(self, 'debug_logger', None),
                        )
                        if dup_ids:
                            is_dup = True
                            found_dup_ids = list(dup_ids)
                            break
                    if is_dup:
                        dedup_skipped += 1
                        try:
                            title = params.get('description', 'Hypothesis')
                            if found_dup_ids:
                                dedup_details.append(f"LLM-dedup: {title} (similar to {', '.join(found_dup_ids)})")
                            else:
                                dedup_details.append(f"LLM-dedup: {title}")
                        except Exception:
                            pass
                        # Skip forming duplicate
                        continue
                except Exception:
                    # Never fail deep_think on dedup errors
                    pass
                # Be defensive: guard against unexpected returns
                try:
                    res = self._form_hypothesis(params)
                except Exception as e:
                    res = {'status': 'error', 'error': f'hypothesis formation failed: {e}'}
                if isinstance(res, dict) and res.get('status') == 'success':
                    added += 1
                    # Append to existing list so subsequent items can see it for dedup
                    try:
                        hid = res.get('hypothesis_id') or ''
                        existing_hyps.append({
                            'id': hid,
                            'title': params.get('description', ''),
                            'description': params.get('details') or params.get('description', ''),
                            'vulnerability_type': params.get('vulnerability_type', 'security_issue'),
                            'node_refs': params.get('node_ids') or [],
                        })
                    except Exception:
                        pass
                    try:
                        titles.append(params.get('description', 'Hypothesis'))
                        hyp_info.append({
                            'title': params.get('description', 'Hypothesis'),
                            'severity': params.get('severity', 'medium'),
                            'confidence': params.get('confidence', 0.6),
                            'reasoning': params.get('reasoning', '')[:500],
                        })
                    except Exception:
                        pass
                else:
                    # Record store-level duplicate reasons (or other failures) compactly
                    try:
                        summary = (res or {}).get('summary', '') if isinstance(res, dict) else ''
                        title = params.get('description', 'Hypothesis')
                        if 'Duplicate title:' in summary or 'Similar to existing:' in summary:
                            dedup_skipped += 1
                            dedup_details.append(f"Store-dedup: {title} — {summary.replace('Failed: ', '')}")
                        else:
                            # Other error (e.g., formation failed)
                            err = (res or {}).get('error', '') if isinstance(res, dict) else ''
                            if summary or err:
                                other_skipped.append(f"Form-fail: {title} — {(summary or err)[:200]}")
                    except Exception:
                        pass
            # Include raw strategist output if available for CLI display
            full_raw = None
            try:
                full_raw = getattr(strategist, 'last_raw', None)
            except Exception:
                full_raw = None
            # Collect skip info from strategist (e.g., no-node-id cases)
            skipped_no_node_ids: list[str] = []
            try:
                _sk = getattr(strategist, 'last_skipped', None) or {}
                if isinstance(_sk, dict):
                    skipped_no_node_ids = list(_sk.get('no_node_ids') or [])
                    skipped_invalid_format = list(_sk.get('invalid_format') or [])
                    fallback_assigned = list(_sk.get('fallback_node_ids_assigned') or [])
                else:
                    skipped_invalid_format = []
                    fallback_assigned = []
            except Exception:
                skipped_no_node_ids = []
                skipped_invalid_format = []
                fallback_assigned = []
            # Heuristic guidance extraction for CLI display: pull bullet points after a GUIDANCE header
            guidance_bullets: list[str] = []
            try:
                if isinstance(full_raw, str) and full_raw:
                    lines = [ln.rstrip() for ln in full_raw.splitlines()]
                    # Find a line that looks like a GUIDANCE header
                    start = -1
                    for i, ln in enumerate(lines):
                        low = ln.strip().lower()
                        if low.startswith('guidance') or low == 'next steps:' or low.startswith('next steps'):
                            start = i + 1
                            break
                    if start >= 0:
                        for ln in lines[start:]:
                            s = ln.strip()
                            if not s:
                                # stop at blank line
                                break
                            if s.startswith('- ') or s.startswith('•') or s.startswith('* '):
                                # normalize bullets and trim length
                                b = s.lstrip('•*').lstrip('-').strip()
                                if b:
                                    guidance_bullets.append(b[:200])
                            # stop when a new all-caps header seems to start
                            if s.isupper() and len(s) < 40:
                                break
                    # Limit to a handful for display
                    guidance_bullets = guidance_bullets[:5]
            except Exception:
                guidance_bullets = []
            # Get store stats
            store_stats = {}
            try:
                all_hyps = self.hypothesis_store.list_all()
                store_stats = {
                    'total': len(all_hyps),
                    'high_severity': sum(1 for h in all_hyps if h.get('severity') == 'high'),
                    'critical': sum(1 for h in all_hyps if h.get('severity') == 'critical'),
                }
            except Exception:
                pass
            
            return {
                'status': 'success',
                'summary': f'Deep analysis added {added} hypotheses' + (f", skipped {dedup_skipped} duplicates" if dedup_skipped else ''),
                'hypotheses_formed': added,
                'hypothesis_titles': titles,
                'hypotheses_info': hyp_info,
                'full_response': full_raw or '',
                'guidance_bullets': guidance_bullets,
                'dedup_skipped': dedup_skipped,
                'dedup_details': dedup_details,
                'skipped_no_node_ids': skipped_no_node_ids,
                'skipped_invalid_format': skipped_invalid_format,
                'skipped_errors': other_skipped,
                'fallback_node_ids_assigned': fallback_assigned,
                'store_stats': store_stats,
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _form_hypothesis(self, params: dict) -> dict:
        """Form a new hypothesis."""
        from .concurrent_knowledge import Hypothesis
        from .path_utils import guess_relpaths
        
        # Ensure we have at least one node ID
        node_ids = params.get('node_ids') or []
        if not node_ids:
            return {'status': 'error', 'error': 'Hypothesis must reference at least one node'}
        
        # Determine which graph this hypothesis relates to.
        # Use a safe fallback if system_graph key exists but value is None.
        graph_name = params.get(
            'graph_name',
            (self.loaded_data.get('system_graph') or {}).get('name', 'unknown')
        )
        
        # Get model information
        model_info = f"{self.llm.provider_name}:{self.llm.model}"
        
        # Check if this is from deep_think (params will have guidance_model set)
        junior_model = model_info  # Default to agent model
        senior_model = params.get('guidance_model')  # Set if from deep_think
        
        # Create hypothesis object with compact title but detailed description
        # The title is compact but the description must be COMPLETE
        hypothesis = Hypothesis(
            title=params.get('description', 'vuln')[:200],  # Increased limit to avoid truncation
            description=params.get('details', params.get('description', '')),  # FULL details here
            vulnerability_type=params.get('vulnerability_type', 'unknown'),
            severity=params.get('severity', 'medium'),
            confidence=params.get('confidence', 0.5),
            node_refs=node_ids,
            reasoning=params.get('reasoning', ''),
            created_by=self.agent_id,
            reported_by_model=senior_model or junior_model,  # Legacy field for backward compatibility
            junior_model=junior_model,
            senior_model=senior_model,
            session_id=getattr(self, 'session_id', None),
            visibility=params.get('visibility', getattr(self, 'default_hypothesis_visibility', 'global'))
        )
        
        # Extract source files from nodes (robust to missing cards map)
        source_files = set()
        affected_functions = []
        cards_map = {}
        try:
            cards_map = getattr(self, 'cards', {}) or {}
        except Exception:
            cards_map = {}
        
        # Look up source files for each node
        for node_id in node_ids:
            # Check in system graph
            if self.loaded_data.get('system_graph'):
                for node in self.loaded_data['system_graph']['data'].get('nodes', []):
                    if node.get('id') == node_id:
                        # Extract source files from source_refs (card IDs)
                        for card_id in node.get('source_refs', []):
                            if card_id in cards_map:
                                card = cards_map[card_id]
                                # Cards have 'relpath' not 'file_path'
                                if 'relpath' in card:
                                    source_files.add(card['relpath'])
                        
                        # Track function name if it's a function node
                        if node.get('type') == 'function':
                            func_name = node.get('label', node_id).split('.')[-1]
                            affected_functions.append(func_name)
            
            # Also check in loaded graphs
            for graph_data in self.loaded_data.get('graphs', {}).values():
                if not isinstance(graph_data, dict):
                    continue
                for node in graph_data.get('nodes', []):
                    if node.get('id') == node_id:
                        for card_id in node.get('source_refs', []):
                            if card_id in cards_map:
                                card = cards_map[card_id]
                                # Cards have 'relpath' not 'file_path'
                                if 'relpath' in card:
                                    source_files.add(card['relpath'])
        
        # Heuristically augment with file paths mentioned in strategist text
        try:
            extra_texts = [
                params.get('details') or '',
                params.get('description') or '',
                params.get('reasoning') or '',
            ]
            guessed = guess_relpaths("\n".join([t for t in extra_texts if t]), self._repo_root)
            for rel in guessed:
                source_files.add(rel)
        except Exception:
            # Never block hypothesis formation on heuristics
            pass

        # Store graph name and source files in properties (NOT shown to agent)
        hypothesis.properties = {
            'graph_name': graph_name,
            'source_files': list(source_files),
            'affected_functions': affected_functions
        }
        
        # Store in persistent hypothesis store
        success, hyp_id = self.hypothesis_store.propose(hypothesis)
        
        # Also keep in memory for backward compatibility
        self.loaded_data['hypotheses'].append({
            'id': hyp_id,
            'description': hypothesis.title,
            'vulnerability_type': hypothesis.vulnerability_type,
            'confidence': hypothesis.confidence,
            'status': hypothesis.status,
            'node_ids': hypothesis.node_refs,
            'evidence': []
        })
        
        return {
            'status': 'success' if success else 'error',
            'summary': f"Formed hypothesis: {hypothesis.title}" if success else f"Failed: {hyp_id}",
            'hypothesis_id': hyp_id if success else None,
            'hypothesis_index': len(self.loaded_data['hypotheses']) - 1
        }
    
    def _update_hypothesis(self, params: dict) -> dict:
        """Update an existing hypothesis."""
        from .concurrent_knowledge import Evidence
        
        # Support both index and ID
        hyp_id = params.get('hypothesis_id')
        if not hyp_id:
            index = params.get('hypothesis_index', 0)
            if index >= len(self.loaded_data['hypotheses']):
                return {'status': 'error', 'error': 'Invalid hypothesis index'}
            hyp_id = self.loaded_data['hypotheses'][index].get('id')
        
        # Update confidence if provided
        if 'new_confidence' in params and hyp_id:
            reason = params.get('reason', 'Agent analysis')
            self.hypothesis_store.adjust_confidence(hyp_id, params['new_confidence'], reason)
            
            # Update in memory too
            for h in self.loaded_data['hypotheses']:
                if h.get('id') == hyp_id:
                    h['confidence'] = params['new_confidence']
        
        # Add evidence if provided
        if 'evidence' in params and hyp_id:
            evidence = Evidence(
                description=params['evidence'],
                type=params.get('evidence_type', 'supports'),
                confidence=params.get('evidence_confidence', 0.7),
                node_refs=params.get('node_ids') or [],
                created_by=self.agent_id
            )
            self.hypothesis_store.add_evidence(hyp_id, evidence)
            
            # Update in memory
            for h in self.loaded_data['hypotheses']:
                if h.get('id') == hyp_id:
                    h['evidence'].append(params['evidence'])
        
        return {
            'status': 'success',
            'summary': f"Updated hypothesis {hyp_id}",
            'hypothesis_id': hyp_id
        }
    
    def _generate_report(self, iterations: int) -> dict:
        """Generate final investigation report."""
        # Categorize hypotheses
        confirmed = [h for h in self.loaded_data['hypotheses'] if h['confidence'] >= 0.8]
        rejected = [h for h in self.loaded_data['hypotheses'] if h['confidence'] <= 0.2]
        uncertain = [h for h in self.loaded_data['hypotheses'] 
                    if 0.2 < h['confidence'] < 0.8]
        
        return {
            'investigation_goal': self.investigation_goal,
            'iterations_completed': iterations,
            'graphs_analyzed': list(self.loaded_data.get('graphs', {}).keys()),
            'nodes_analyzed': len(self.loaded_data['nodes']),
            'hypotheses': {
                'total': len(self.loaded_data['hypotheses']),
                'confirmed': len(confirmed),
                'rejected': len(rejected),
                'uncertain': len(uncertain)
            },
            'detailed_hypotheses': [
                {
                    'description': h['description'],
                    'type': h['vulnerability_type'],
                    'confidence': h['confidence'],
                    'status': 'confirmed' if h['confidence'] >= 0.8 
                             else 'rejected' if h['confidence'] <= 0.2 
                             else 'uncertain',
                    'evidence': h.get('evidence', [])
                }
                for h in self.loaded_data['hypotheses']
            ]
        }
