"""
Professional security audit report generator.
"""

import json
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from llm.unified_client import UnifiedLLMClient


class ReportGenerator:
    """Generate professional security audit reports."""
    
    def __init__(self, project_dir: Path, config: dict, debug: bool = False, include_all: bool = False):
        """Initialize report generator."""
        self.project_dir = project_dir
        self.config = config
        self.debug = debug
        self.include_all = include_all  # Flag to include all hypotheses, not just confirmed
        
        # Initialize reporting LLM
        self.llm = UnifiedLLMClient(
            cfg=config,
            profile="reporting",
            debug_logger=None
        )
        
        # Load project data
        self.graphs = self._load_graphs()
        self.hypotheses = self._load_hypotheses()
        self.hypothesis_metadata = self._load_hypothesis_metadata()
        self.agent_runs = self._load_agent_runs()
        self.pocs = self._load_pocs()  # Load available PoCs
        # Debug helpers for CLI
        self.last_prompt: str | None = None
        self.last_response: str | None = None
        # Optional progress callback
        self._progress_cb = None
        
        # Load card store + repo root for precise code snippets
        self.card_store: dict[str, dict[str, Any]] = {}
        self.repo_root: Path | None = None
        self._load_card_store_and_repo_root()
        # Unique counter for code blocks in the rendered report
        self._code_block_counter: int = 0

    def _load_card_store_and_repo_root(self) -> None:
        """Load card_store.json (graph evidence) and determine repo root.

        Prefers using graphs/knowledge_graphs.json metadata to locate the
        card store and repo path. Falls back silently if not present.
        """
        try:
            graphs_dir = self.project_dir / "graphs"
            kg_path = graphs_dir / "knowledge_graphs.json"
            if kg_path.exists():
                with open(kg_path) as f:
                    kg = json.load(f)
                # Repo root from manifest
                manifest = kg.get('manifest') or {}
                rp = manifest.get('repo_path')
                if rp:
                    try:
                        self.repo_root = Path(rp)
                    except Exception:
                        self.repo_root = None
                # Card store
                card_store_path = kg.get('card_store_path')
                if card_store_path:
                    try:
                        with open(card_store_path) as f:
                            store = json.load(f)
                        if isinstance(store, dict):
                            self.card_store = store
                    except Exception:
                        # Try local graphs dir fallback
                        csp = graphs_dir / "card_store.json"
                        if csp.exists():
                            with open(csp) as f2:
                                store = json.load(f2)
                            if isinstance(store, dict):
                                self.card_store = store
            else:
                # Try direct card_store.json
                csp = self.project_dir / "graphs" / "card_store.json"
                if csp.exists():
                    with open(csp) as f3:
                        store = json.load(f3)
                    if isinstance(store, dict):
                        self.card_store = store
        except Exception:
            # Leave empty on failure; fallback logic will handle
            if self.debug:
                print("[!] Failed to load card store or repo root")
    
    def _load_graphs(self) -> dict[str, Any]:
        """Load all graphs from project."""
        graphs = {}
        graphs_dir = self.project_dir / "graphs"
        
        for graph_file in graphs_dir.glob("graph_*.json"):
            with open(graph_file) as f:
                data = json.load(f)
                name = graph_file.stem.replace("graph_", "")
                graphs[name] = data
        
        return graphs
    
    def _load_hypotheses(self) -> dict[str, Any]:
        """Load hypotheses from project."""
        hypothesis_file = self.project_dir / "hypotheses.json"
        if hypothesis_file.exists():
            with open(hypothesis_file) as f:
                data = json.load(f)
                return data.get("hypotheses", {})
        return {}
    
    def _load_hypothesis_metadata(self) -> dict[str, Any]:
        """Load hypothesis metadata (e.g., finalization model)."""
        hypothesis_file = self.project_dir / "hypotheses.json"
        if hypothesis_file.exists():
            with open(hypothesis_file) as f:
                data = json.load(f)
                return data.get("metadata", {})
        return {}
    
    def _load_pocs(self) -> dict[str, dict[str, Any]]:
        """Load available PoCs for hypotheses."""
        pocs = {}
        poc_dir = self.project_dir / "poc"
        
        if not poc_dir.exists():
            return pocs
        
        # Scan each hypothesis directory
        for hyp_dir in poc_dir.iterdir():
            if not hyp_dir.is_dir():
                continue
            
            hypothesis_id = hyp_dir.name
            metadata_file = hyp_dir / "metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                # Load actual PoC files
                poc_files = []
                for file_info in metadata.get('files', []):
                    file_path = hyp_dir / file_info['name']
                    if file_path.exists():
                        try:
                            with open(file_path) as f:
                                content = f.read()
                            poc_files.append({
                                'name': file_info['name'],
                                'content': content,
                                'description': file_info.get('description', '')
                            })
                        except Exception as e:
                            # Skip binary or unreadable files
                            if self.debug:
                                print(f"[!] Could not read PoC file {file_path}: {e}")
                
                if poc_files:
                    pocs[hypothesis_id] = {
                        'metadata': metadata,
                        'files': poc_files
                    }
        
        return pocs
    
    def _format_graph_name(self, graph_name: str) -> str:
        """Format graph names to be human-readable.
        Examples:
          AuthorizationRolesActions -> Authorization, Roles and Actions
          AssetRoutingFlow -> Asset Routing and Flow
          TimelockActionLifecycle -> Timelock Action Lifecycle
        """
        if not graph_name:
            return ''
        
        # Remove 'graph_' prefix if present
        if graph_name.startswith('graph_'):
            graph_name = graph_name[6:]
        
        # Split on capitals and format
        import re
        words = re.findall(r'[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', graph_name)
        
        # Special cases for common terms
        word_map = {
            'And': 'and',
            'Or': 'or',
            'The': 'the',
            'Of': 'of',
            'In': 'in',
            'Calls': 'Calls',
            'Flow': 'Flow',
            'Map': 'Mapping'
        }
        
        formatted_words = []
        for i, word in enumerate(words):
            if i == 0:
                formatted_words.append(word)
            elif word in word_map:
                formatted_words.append(word_map[word])
            else:
                formatted_words.append(word.lower())
        
        result = ' '.join(formatted_words)
        
        # Special formatting for known patterns
        replacements = {
            'Authorization roles actions': 'Authorization, Roles and Actions',
            'Asset routing flow': 'Asset Routing and Flow',
            'Timelock action lifecycle': 'Timelock Action Lifecycle',
            'Inheritance and override map': 'Inheritance and Override Mapping',
            'Cross contract deposit withdraw calls': 'Cross-Contract Deposit/Withdraw Calls',
            'System architecture': 'System Architecture',
            'External call reentrancy': 'External Calls and Reentrancy',
            'Token asset flow': 'Token and Asset Flow'
        }
        
        for old, new in replacements.items():
            if result.lower() == old.lower():
                return new
        
        return result.title()
    
    def _extract_audit_models(self) -> dict[str, list[str]]:
        """Extract all models involved in the audit process from various sources."""
        junior_models = set()
        senior_models = set()
        
        # Extract models from hypotheses - they have junior_model and senior_model fields
        for h in self.hypotheses.values():
            if isinstance(h, dict):
                # Check for junior_model field
                junior = h.get('junior_model', '')
                if junior:
                    # Remove provider prefix if present (e.g., "OpenAI:gpt-5" -> "gpt-5")
                    if ':' in junior:
                        junior = junior.split(':', 1)[1]
                    junior_models.add(junior)
                
                # Check for senior_model field
                senior = h.get('senior_model', '')
                if senior:
                    # Remove provider prefix if present
                    if ':' in senior:
                        senior = senior.split(':', 1)[1]
                    senior_models.add(senior)
                
                # Fallback to reported_by_model for legacy data
                if not junior and not senior:
                    model = h.get('reported_by_model', '')
                    if model:
                        # Remove provider prefix if present
                        if ':' in model:
                            model = model.split(':', 1)[1]
                        junior_models.add(model)
        
        # Get models from config as final fallback
        graph_model = self.config.get('models', {}).get('graph', {}).get('model', '')
        agent_model = self.config.get('models', {}).get('agent', {}).get('model', '')
        guidance_model = self.config.get('models', {}).get('guidance', {}).get('model', '')
        finalize_model = self.config.get('models', {}).get('finalize', {}).get('model', '')
        reporting_model = self.config.get('models', {}).get('reporting', {}).get('model', '')
        
        # Use config models if no models found
        if not junior_models:
            if graph_model:
                junior_models.add(graph_model)
            if agent_model and agent_model != graph_model:
                junior_models.add(agent_model)
        
        if not senior_models and guidance_model:
            senior_models.add(guidance_model)
        
        # Deduplicate graph and agent models if they're the same
        junior_list = sorted(list(junior_models))
        if len(junior_list) == 2 and junior_list[0] == junior_list[1]:
            junior_list = [junior_list[0]]
        
        return {
            'junior': [self._format_model_name(m) for m in junior_list if m],
            'senior': [self._format_model_name(m) for m in sorted(senior_models) if m],
            'finalize': self._format_model_name(finalize_model) if finalize_model else '',
            'reporting': self._format_model_name(reporting_model) if reporting_model else ''
        }
    
    def _generate_models_table_html(self) -> str:
        """Generate an HTML table showing all models involved in the audit."""
        models = self._extract_audit_models()
        
        # Combine junior models if they're duplicates
        junior_display = ', '.join(models['junior']) if models['junior'] else 'Not specified'
        senior_display = ', '.join(models['senior']) if models['senior'] else 'Not specified'
        
        table_html = """
            <table style="width: 100%; margin: 20px 0; border-collapse: collapse; background: rgba(26,31,46,0.4); border: 1px solid rgba(136,146,176,0.2); border-radius: 8px; overflow: hidden;">
                <thead>
                    <tr style="background: rgba(100,181,246,0.1); border-bottom: 1px solid rgba(136,146,176,0.2);">
                        <th style="padding: 12px 16px; text-align: left; color: #64b5f6; font-weight: 600;">Audit Role</th>
                        <th style="padding: 12px 16px; text-align: left; color: #64b5f6; font-weight: 600;">Model(s)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(136,146,176,0.1);">
                        <td style="padding: 12px 16px; color: #8892b0;">Junior Auditors</td>
                        <td style="padding: 12px 16px; color: #e8ecf1;">{}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(136,146,176,0.1);">
                        <td style="padding: 12px 16px; color: #8892b0;">Senior Auditor</td>
                        <td style="padding: 12px 16px; color: #e8ecf1;">{}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(136,146,176,0.1);">
                        <td style="padding: 12px 16px; color: #8892b0;">Quality Assurance</td>
                        <td style="padding: 12px 16px; color: #e8ecf1;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 16px; color: #8892b0;">Report Writing</td>
                        <td style="padding: 12px 16px; color: #e8ecf1;">{}</td>
                    </tr>
                </tbody>
            </table>
        """.format(
            junior_display,
            senior_display,
            models['finalize'] or 'Not specified',
            models['reporting'] or 'Not specified'
        )
        
        return table_html
    
    def _format_model_name(self, model_name: str) -> str:
        """Format model names to be more natural.
        Examples:
          OpenAI:gpt-4o -> GPT-4
          OpenAI:gpt-4o-mini -> GPT-4 Mini
          anthropic:claude-3-opus -> Claude-3 Opus
          gpt-5 -> GPT-5
        """
        if not model_name:
            return ''
        
        # Remove provider prefix if present
        if ':' in model_name:
            model_name = model_name.split(':', 1)[1]
        
        # Format common model names
        name_map = {
            'gpt-4o': 'GPT-4',
            'gpt-4': 'GPT-4',
            'gpt-4o-mini': 'GPT-4 Mini',
            'gpt-3.5-turbo': 'GPT-3.5',
            'gpt-5': 'GPT-5',
            'gpt-5-nano': 'GPT-5 Nano',
            'gpt-5-mini': 'GPT-5 Mini',
            'claude-3-opus': 'Claude-3 Opus',
            'claude-3-sonnet': 'Claude-3 Sonnet',
            'claude-3-haiku': 'Claude-3 Haiku',
            'gemini-pro': 'Gemini Pro',
            'gemini-ultra': 'Gemini Ultra'
        }
        
        return name_map.get(model_name.lower(), model_name)
    
    def _load_agent_runs(self) -> list[dict]:
        """Load agent run summaries."""
        runs = []
        agent_dir = self.project_dir / "agent_runs"
        if agent_dir.exists():
            for run_file in agent_dir.glob("*.json"):
                with open(run_file) as f:
                    runs.append(json.load(f))
        return runs

    def _get_system_architecture_graph(self) -> dict[str, Any] | None:
        """Return the full System Architecture graph data if available."""
        if 'SystemArchitecture' in self.graphs:
            return self.graphs['SystemArchitecture']
        for key in self.graphs.keys():
            lk = key.lower()
            if 'system' in lk and ('architecture' in lk or 'overview' in lk):
                return self.graphs[key]
        return next(iter(self.graphs.values())) if self.graphs else None

    def _generate_sections(self, project_name: str, project_source: str) -> dict[str, str]:
        """Single LLM call that returns both executive summary and system overview."""
        system_graph = self._get_system_architecture_graph() or {}
        
        # Get all graph names for scope description - format them nicely
        graph_names = [self._format_graph_name(name) for name in self.graphs.keys()]
        
        # Use the new helper method to extract models
        models = self._extract_audit_models()
        junior_auditors = models['junior']
        senior_auditors = models['senior']
        finalize_model = models['finalize']
        reporting_model = models['reporting'] or 'GPT-5'
        
        # Get all unique models for analysis_models
        all_analysis_models = set()
        all_analysis_models.update(junior_auditors)
        all_analysis_models.update(senior_auditors)
        sorted(list(all_analysis_models))
        
        # Provide full system graph and full hypotheses store
        hypotheses_payload = {
            'hypotheses': self.hypotheses,
            'metadata': self.hypothesis_metadata,
        }

        prompt = (
            f"PROJECT_NAME: {project_name}\n"
            f"PROJECT_SOURCE: {project_source}\n\n"
            "SYSTEM_GRAPH_JSON\n"
            f"{json.dumps(system_graph, ensure_ascii=False)}\n\n"
            "HYPOTHESES_STORE_JSON\n"
            f"{json.dumps(hypotheses_payload, ensure_ascii=False)}\n\n"
            "AUDIT_SCOPE_GRAPHS (human-readable descriptions of review areas)\n"
            f"{json.dumps(graph_names)}\n\n"
            "Return a JSON object with exactly these keys (no prose outside JSON):\n"
            "{\n"
            "  \"application_name\": string,\n"
            "  \"executive_summary\": string,\n"
            "  \"system_overview\": string\n"
            "}\n\n"
            f"ACTUAL_TEAM_MEMBERS:\n"
            f"- Junior Auditors (Graph/Agent models): {', '.join(junior_auditors) if junior_auditors else 'Not specified'}\n"
            f"- Senior Auditors (Guidance model): {', '.join(senior_auditors) if senior_auditors else 'Not specified'}\n"
            f"- Quality assurance by: {finalize_model if finalize_model else 'Not specified'}\n"
            f"- Report written by: {reporting_model}\n\n"
            "Guidance:\n"
            f"- Application name: Extract the actual protocol/application name from the system graph nodes and contracts. Look for protocol names, diamond names, or main contract names. If unclear, use the project name '{project_name}' capitalized.\n"
            "- Executive summary (2-3 paragraphs with good spacing):\n"
            "  * First paragraph: State that the Hound security team conducted this comprehensive audit\n"
            "  * Second paragraph: List aspects from AUDIT_SCOPE_GRAPHS as readable prose, not class names\n"
            "  * Third paragraph: Brief summary of findings and security posture\n"
            "  * DO NOT mention specific model names in the text - these will be shown in a table below\n"
            "  * DO NOT make up human names like Alex, Jessica, etc.\n"
            "  * DO NOT duplicate model information that's shown in the table\n"
            "  * NEVER use: 'hypothesis', 'AI', 'model', 'automated', 'LLM', or made-up human names\n"
            "  * ALWAYS write as: 'The Hound team', 'our team', 'we'\n"
            "  * Use line breaks between paragraphs for readability\n"
            "- System overview (4-5 paragraphs with good spacing):\n"
            "  * Write as the Hound team describing what we analyzed\n"
            "  * Use concrete contract/component names from the system graph\n"
            "  * Include line breaks between paragraphs\n"
            "  * Describe architecture, data flows, and security mechanisms\n"
            "  * Focus on technical details relevant to security\n"
            "  * Use: 'The Hound team identified', 'our analysis revealed', 'we discovered'\n"
            "  * When mentioning specific findings, ONLY use actual model names from ACTUAL_TEAM_MEMBERS\n"
            "  * DO NOT invent human names - only use the exact model names provided\n"
        )

        response = self.llm.raw(
            system="You are a senior security auditor. Respond only with valid JSON.",
            user=prompt
        )
        # Save prompt/response for CLI debug
        self.last_prompt = prompt
        self.last_response = response
        from utils.json_utils import extract_json_object
        obj = extract_json_object(response)
        if isinstance(obj, dict) and 'executive_summary' in obj and 'system_overview' in obj:
            return {
                'application_name': str(obj.get('application_name') or 'Application').strip(),
                'executive_summary': str(obj.get('executive_summary') or '').strip(),
                'system_overview': str(obj.get('system_overview') or '').strip()
            }

        # Fallback: when using mock provider (or test environments), synthesize minimal sections
        try:
            provider = getattr(self.llm, 'provider_name', '')
        except Exception:
            provider = ''
        if provider == 'mock':
            # Compose a compact, deterministic summary from local data
            app_name = (project_name or 'Application').strip()
            graph_names = sorted(list(system_graph.keys())) if system_graph else []
            num_graphs = len(graph_names)
            num_hypotheses = len(self.hypotheses or {})
            # Count nodes/edges for the SystemArchitecture graph if present
            sa = self.graphs.get('SystemArchitecture') or {}
            nodes = len(sa.get('nodes', [])) if isinstance(sa, dict) else 0
            edges = len(sa.get('edges', [])) if isinstance(sa, dict) else 0
            exec_summary = (
                f"The Hound team conducted a focused audit of {app_name}.\n\n"
                f"Scope included {num_graphs} graph(s): {', '.join(graph_names) if graph_names else 'none listed'}. "
                f"We evaluated architecture, authorization, and value flows where applicable.\n\n"
                f"No LLM narrative was used for this run; this summary is synthesized from local project data. "
                f"Hypotheses present: {num_hypotheses}."
            )
            sys_overview = (
                f"System overview derived from graphs. "
                f"SystemArchitecture: nodes={nodes}, edges={edges}. "
                f"Source path: {project_source}."
            )
            return {
                'application_name': app_name,
                'executive_summary': exec_summary,
                'system_overview': sys_overview,
            }

        # Non-mock providers: surface a clear error for missing keys
        raise ValueError("LLM did not return required keys application_name, executive_summary and system_overview in JSON response")

    # Note: No fallback generators — we surface errors so the CLI can show details
    
    def generate(self, project_name: str, project_source: str,
                title: str, auditors: list[str], format: str = 'html',
                progress_callback: Callable[[dict], None] | None = None) -> str:
        """Generate the complete audit report."""
        self._progress_cb = progress_callback
        self._emit_progress('start', 'Starting report generation')
        
        # Gather report data
        report_date = datetime.now().strftime("%B %d, %Y")
        
        # Build auditors display: Hound team members (AI models as named auditors)
        # Use the helper to extract models properly
        models = self._extract_audit_models()
        
        # Build complete list of team members
        auditor_models = []
        auditor_models.extend(models['junior'])
        auditor_models.extend(models['senior'])
        
        finalize_model = models['finalize']
        reporting_model = models['reporting']
        
        if finalize_model and finalize_model not in auditor_models:
            auditor_models.append(finalize_model)
        if reporting_model and reporting_model not in auditor_models:
            auditor_models.append(reporting_model)
        
        # Remove duplicates while preserving order
        seen = set()
        auditor_models = [x for x in auditor_models if not (x in seen or seen.add(x))]
        
        if not auditor_models:
            auditor_models = ['Hound Security Team']
        
        # Preferred: generate both sections via a single LLM call
        self._emit_progress('llm', 'Generating executive summary and system overview')
        sections = self._generate_sections(project_name, project_source)
        self._emit_progress('llm_done', 'Sections generated')
        application_name = sections.get('application_name', project_name)
        executive_summary = sections.get('executive_summary', '')
        system_overview = sections.get('system_overview', '')
        
        # Get findings (confirmed or all based on include_all flag)
        findings_msg = 'Collecting ALL hypotheses (no QA performed)' if self.include_all else 'Collecting confirmed findings'
        self._emit_progress('findings', findings_msg)
        confirmed_findings = self._get_confirmed_findings()
        findings_label = 'hypotheses' if self.include_all else 'findings'
        self._emit_progress('findings_done', f"Processed {len(confirmed_findings)} {findings_label}")
        
        # No longer generating appendix
        # tested_hypotheses = self._get_all_hypotheses()
        
        # Coverage summary removed - not showing coverage metrics

        # Generate the report based on format
        if format == 'html':
            self._emit_progress('render', 'Rendering HTML report')
            return self._generate_html_report(
                title=title,
                application_name=application_name,
                report_date=report_date,
                auditors=auditor_models,
                executive_summary=executive_summary,
                system_overview=system_overview,
                findings=confirmed_findings,
                project_name=project_name,
                project_source=project_source,
                report_writer=reporting_model
            )
        elif format == 'markdown':
            self._emit_progress('render', 'Rendering Markdown report')
            return self._generate_markdown_report(
                title=title,
                application_name=application_name,
                report_date=report_date,
                auditors=auditor_models,
                executive_summary=executive_summary,
                system_overview=system_overview,
                findings=confirmed_findings,
                project_name=project_name,
                project_source=project_source,
                report_writer=reporting_model
            )
        else:
            # Default to HTML
            self._emit_progress('render', 'Rendering HTML report')
            return self._generate_html_report(
                title=title,
                application_name=application_name,
                report_date=report_date,
                auditors=auditor_models,
                executive_summary=executive_summary,
                system_overview=system_overview,
                findings=confirmed_findings,
                project_name=project_name,
                project_source=project_source,
                report_writer=reporting_model
            )
    
    def _generate_executive_summary(self, project_name: str, 
                                   project_source: str) -> str:
        """Generate executive summary using LLM with real graph data."""
        
        # Get actual graph descriptions
        graph_summary = self._summarize_graphs_for_executive()
        
        # Get findings summary
        confirmed_findings = self._get_confirmed_findings()
        findings_summary = self._summarize_findings(confirmed_findings)
        
        # Count investigations
        total_investigations = len(self.hypotheses)
        confirmed_count = len(confirmed_findings)
        
        # Use the helper to extract models properly
        models = self._extract_audit_models()
        junior_auditors = models['junior']
        senior_auditors = models['senior']
        models['finalize'] or 'unspecified'
        
        # Combine for lead auditors list (deduplicated)
        all_auditors = sorted(set(junior_auditors + senior_auditors))
        models_used = all_auditors.copy()  # Define models_used here
        if not models_used:
            models_used = ['unspecified']

        # Add warning about unreviewed findings if include_all is True
        qa_warning = ""
        findings_label = "Confirmed vulnerabilities"
        if self.include_all:
            qa_warning = """
IMPORTANT NOTE: This report includes ALL hypotheses generated during analysis, not just confirmed findings.
No quality assurance or review process has been performed on these findings.
This report may contain false positives and should be treated as a preliminary analysis only.
"""
            findings_label = "Total hypotheses (UNREVIEWED)"

        prompt = f"""You are a senior security auditor writing the executive summary for a professional audit report.

Project: {project_name}
Source: {project_source}
{qa_warning}
System Characteristics (from architecture analysis):
{graph_summary}

Audit Statistics:
- Total security concerns investigated: {total_investigations}
- {findings_label}: {confirmed_count}
{findings_summary}

Write a professional executive summary (2-3 paragraphs) that:
1) States that the Hound security team conducted this {'preliminary' if self.include_all else 'comprehensive'} security audit of {project_name}.
2) Describes WHAT the system does (type and core purpose) based on the components above.
3) Summarizes the security posture and any findings discovered.
{'4) CRITICAL: Include a clear warning that this report contains UNREVIEWED findings that have not undergone quality assurance and may contain false positives.' if self.include_all else ''}

CRITICAL INSTRUCTIONS:
- DO NOT mention any specific model names (GPT-5, Claude, etc.) - a table below will show this
- DO NOT discuss who did what parts of the analysis
- DO NOT make up human names like Alex, Jessica, Sarah, etc.
- Write as "The Hound team", "our team", "we" (first person plural)
- Focus on the AUDIT RESULTS not the audit process
- NEVER use: "hypothesis", "AI", "model", "automated", "LLM", or made-up names
- ALWAYS use: "findings", "security concerns", "team", "analysis"
- Present this as a professional security audit by the Hound team
{'- MUST include clear warning about unreviewed findings and potential false positives' if self.include_all else ''}"""

        try:
            summary = self.llm.raw(
                system="You are a senior security auditor writing a professional executive summary.",
                user=prompt
            )
            return summary
        except Exception as e:
            if self.debug:
                print(f"[!] Failed to generate executive summary: {e}")
            # Fallback summary
            return f"""We conducted a comprehensive security audit of the {project_name} system. 
The audit focused on critical security aspects including authorization mechanisms, external interfaces, 
data flow patterns, and potential vulnerability vectors.

Our analysis covered the complete system architecture with particular attention to high-risk components 
and their interactions. The audit methodology included thorough code review, architectural analysis, 
and systematic vulnerability assessment across all identified attack surfaces."""
    
    def _summarize_graphs_for_executive(self) -> str:
        """Summarize graphs for executive summary - focused on what the system does."""
        summary_parts = []
        
        for name, graph in self.graphs.items():
            nodes = graph.get('nodes', [])
            edges = graph.get('edges', [])
            
            # Focus on system/architecture graphs for main description
            if 'system' in name.lower() or 'architecture' in name.lower():
                contracts = [n for n in nodes if n.get('type') == 'contract']
                if contracts:
                    summary_parts.append(f"Core System Components ({len(contracts)} contracts):")
                    for contract in contracts[:5]:  # Top 5 contracts
                        label = contract.get('label', '')
                        props = contract.get('properties', {})
                        desc = props.get('description', '')
                        if desc:
                            summary_parts.append(f"  • {label}: {desc}")
                        else:
                            summary_parts.append(f"  • {label}")
                    if len(contracts) > 5:
                        summary_parts.append(f"  • ... and {len(contracts) - 5} additional contracts")
                
                # Note other components
                functions = [n for n in nodes if n.get('type') == 'function']
                external = [n for n in nodes if n.get('type') == 'external']
                if functions:
                    summary_parts.append(f"\nKey Functions: {len(functions)} critical functions analyzed")
                if external:
                    summary_parts.append(f"External Integrations: {len(external)} external dependencies")
            
            # Note specialized graphs
            elif any(keyword in name.lower() for keyword in ['auth', 'role', 'permission']):
                auth_nodes = len(nodes)
                summary_parts.append(f"\nAuthorization System: {auth_nodes} components")
            elif any(keyword in name.lower() for keyword in ['token', 'asset', 'vault']):
                summary_parts.append(f"\nAsset Management: {len(nodes)} components, {len(edges)} relationships")
            elif any(keyword in name.lower() for keyword in ['governance', 'timelock']):
                summary_parts.append(f"\nGovernance: {len(nodes)} governance components")
        
        return '\n'.join(summary_parts) if summary_parts else "Complex smart contract system"
    
    def _summarize_findings(self, findings: list[dict]) -> str:
        """Create a brief findings summary for executive summary."""
        if not findings:
            return "\nNo critical vulnerabilities were confirmed during this audit."
        
        # Count by severity
        severity_counts = {}
        for f in findings:
            sev = f.get('severity', 'medium')
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        parts = []
        if severity_counts.get('critical', 0) > 0:
            parts.append(f"{severity_counts['critical']} critical")
        if severity_counts.get('high', 0) > 0:
            parts.append(f"{severity_counts['high']} high")
        if severity_counts.get('medium', 0) > 0:
            parts.append(f"{severity_counts['medium']} medium")  
        if severity_counts.get('low', 0) > 0:
            parts.append(f"{severity_counts['low']} low")
            
        if parts:
            return f"\nFindings: {', '.join(parts)} severity issues confirmed"
        return ""
    
    def _analyze_scope(self) -> str:
        """Analyze graphs to understand audit scope."""
        scope_parts = []
        
        # Find system architecture graph
        system_graph = None
        for name, graph in self.graphs.items():
            if 'system' in name.lower() or 'architecture' in name.lower():
                system_graph = graph
                break
        
        if system_graph:
            # Extract key components
            nodes = system_graph.get('nodes', [])
            node_types = {}
            for node in nodes:
                node_type = node.get('type', 'unknown')
                if node_type not in node_types:
                    node_types[node_type] = []
                node_types[node_type].append(node.get('label', node.get('id', '')))
            
            # Describe main components
            if 'contract' in node_types:
                scope_parts.append(f"Core contracts: {', '.join(node_types['contract'][:5])}")
            if 'function' in node_types:
                scope_parts.append(f"Critical functions analyzed: {len(node_types['function'])}")
            if 'external' in node_types:
                scope_parts.append(f"External integrations: {', '.join(node_types['external'][:3])}")
        
        # Describe other graph focuses
        for name, graph in self.graphs.items():
            if name == 'AuthorizationRolesActions' or 'auth' in name.lower():
                scope_parts.append("Authorization and access control mechanisms")
            elif 'timelock' in name.lower():
                scope_parts.append("Timelock and governance controls")
            elif 'external' in name.lower() or 'reentrancy' in name.lower():
                scope_parts.append("External call surfaces and reentrancy protection")
            elif 'asset' in name.lower() or 'flow' in name.lower():
                scope_parts.append("Asset flow and value transfer patterns")
        
        return '\n'.join(f"- {part}" for part in scope_parts)
    
    def _get_confirmed_findings(self) -> list[dict]:
        """Get confirmed vulnerability findings (or all if include_all is True)."""
        findings = []
        
        # Collect findings based on include_all flag
        for hyp_id, hyp in self.hypotheses.items():
            # Include all hypotheses if flag is set, otherwise only confirmed
            if self.include_all or hyp.get('status') == 'confirmed':
                finding = {
                    'id': hyp_id,
                    'title': hyp.get('title', 'Unknown'),
                    'severity': hyp.get('severity', 'medium'),
                    'type': hyp.get('vulnerability_type', 'unknown'),
                    'description': hyp.get('description', ''),
                    'confidence': hyp.get('confidence', 0),
                    'affected': hyp.get('node_refs', []),
                    'reported_by_model': hyp.get('senior_model') or hyp.get('junior_model') or hyp.get('reported_by_model', 'unknown'),
                    'junior_model': hyp.get('junior_model'),
                    'senior_model': hyp.get('senior_model'),
                    'supporting_evidence': hyp.get('supporting_evidence', []),
                    'properties': hyp.get('properties', {}),
                    'qa_comment': hyp.get('qa_comment', '')  # Include QA comment if available
                }
                findings.append(finding)
        
        # Batch generate professional descriptions for all findings
        if findings:
            self._emit_progress('findings_describe', f'Generating professional descriptions for {len(findings)} findings')
            professional_results = self._batch_generate_vulnerability_descriptions(findings)
            for i, finding in enumerate(findings):
                result = professional_results.get(i, {})
                
                # Use LLM-generated description or fallback
                finding['professional_description'] = result.get('description') or \
                    self._clean_raw_description(finding['description'])
                
                # Use LLM-formatted component description or fallback
                finding['affected_description'] = result.get('affected_components') or \
                    self._describe_affected_components(finding.get('affected', []))
                
                # Extract code samples for this finding
                self._emit_progress('snippets', f"Selecting code snippets: {finding.get('title','')}"[:80])
                code_samples = self._extract_code_for_finding(finding)
                self._emit_progress('snippets_done', f"Selected {len(code_samples)} snippet(s)")
                finding['code_samples'] = code_samples
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        findings.sort(key=lambda x: severity_order.get(x['severity'], 4))
        
        return findings

    def _emit_progress(self, status: str, message: str):
        """Emit progress to callback if provided."""
        if callable(self._progress_cb):
            try:
                self._progress_cb({'status': status, 'message': message})
            except Exception:
                pass
    
    def _get_all_hypotheses(self) -> list[dict]:
        """Get all tested hypotheses for appendix."""
        tested = []
        
        for _, hyp in self.hypotheses.items():
            tested.append({
                'title': hyp.get('title', 'Unknown'),
                'type': hyp.get('vulnerability_type', 'unknown'),
                'status': hyp.get('status', 'proposed'),
                'confidence': hyp.get('confidence', 0),
                'nodes': hyp.get('node_refs', []),
                'reported_by_model': hyp.get('senior_model') or hyp.get('junior_model') or hyp.get('reported_by_model', 'unknown'),
                'junior_model': hyp.get('junior_model'),
                'senior_model': hyp.get('senior_model')
            })
        
        # Sort by confidence
        tested.sort(key=lambda x: x['confidence'], reverse=True)
        
        return tested
    
    def _generate_system_overview(self) -> str:
        """Generate system overview from graphs with security insights."""
        
        # Convert graphs to a format the LLM can understand
        graph_descriptions = self._describe_graphs_for_llm()
        
        scope = self._analyze_scope()
        prompt = f"""You are a senior security auditor writing the System Overview section of an audit report.

Context (for your drafting only):
- Tested aspects include: {scope}

Architecture and components summary for your reference:
{graph_descriptions}

Write a comprehensive System Overview (3-4 paragraphs) that:
1) Describes overall architecture and main components (type of system and core contracts).
2) Explains key relationships and data flows, and critical paths.
3) Highlights security-relevant architectural patterns and mechanisms.
4) Optionally mentions architectural concerns or areas needing attention (no vulnerability details).

Use technical but accessible language; do NOT mention graphs or AI; present as a human audit report."""

        try:
            overview = self.llm.raw(
                system="You are a senior security auditor writing a professional system overview.",
                user=prompt
            )
            return overview.strip()
        except Exception as e:
            if self.debug:
                print(f"[!] Failed to generate system overview: {e}")
            # Fallback overview
            return self._generate_fallback_overview()
    
    def _describe_graphs_for_llm(self) -> str:
        """Convert graphs to a descriptive format for LLM analysis."""
        descriptions = []
        
        for name, graph in self.graphs.items():
            nodes = graph.get('nodes', [])
            edges = graph.get('edges', [])
            
            # Clean graph name for display
            clean_name = name.replace('graph_', '').replace('_', ' ')
            descriptions.append(f"=== {clean_name} Graph ===")
            descriptions.append(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
            
            # Group nodes by type
            nodes_by_type = {}
            for node in nodes:
                node_type = node.get('type', 'unknown')
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                label = node.get('label', node.get('id', 'unnamed'))
                props = node.get('properties', {})
                
                # Include key properties
                node_desc = label
                if props.get('description'):
                    node_desc += f" - {props['description']}"
                elif props.get('signature'):
                    node_desc += f" ({props['signature']})"
                    
                nodes_by_type[node_type].append(node_desc)
            
            # Describe node types and their components
            descriptions.append("\nComponents:")
            for node_type, node_list in nodes_by_type.items():
                descriptions.append(f"\n{node_type.title()}s ({len(node_list)}):")
                # Show up to 10 most important nodes
                for node_desc in node_list[:10]:
                    descriptions.append(f"  • {node_desc}")
                if len(node_list) > 10:
                    descriptions.append(f"  ... and {len(node_list) - 10} more")
            
            # Describe key relationships
            if edges:
                edge_types = {}
                for edge in edges:
                    edge_type = edge.get('type', edge.get('label', 'relates'))
                    if edge_type not in edge_types:
                        edge_types[edge_type] = 0
                    edge_types[edge_type] += 1
                
                descriptions.append("\nRelationships:")
                for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1])[:5]:
                    descriptions.append(f"  • {edge_type}: {count} connections")
            
            descriptions.append("")  # Blank line between graphs
        
        return '\n'.join(descriptions)
    
    def _generate_fallback_overview(self) -> str:
        """Generate a fallback system overview."""
        return """The system architecture consists of multiple interconnected smart contracts implementing 
a modular design pattern. Core components are organized to separate concerns between business logic, 
access control, and data management.

The contracts interact through well-defined interfaces, with clear separation between external-facing 
functions and internal operations. Value flows are managed through controlled entry and exit points, 
with appropriate validation at each stage.

From a security perspective, the architecture demonstrates consideration for common attack vectors, 
with implemented safeguards for reentrancy protection, access control, and input validation. 
External dependencies are limited and clearly defined."""
    
    def _generate_html_report(self, **kwargs) -> str:
        """Generate HTML format report."""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{kwargs['title']}</title>
    
    <!-- Prism.js for syntax highlighting -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-solidity.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-typescript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-rust.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-go.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-java.min.js"></script>
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.7;
            color: #e8ecf1;
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 60px 40px;
        }}
        
        .header {{
            text-align: center;
            padding: 80px 40px;
            margin: -60px -40px 60px;
            background: linear-gradient(180deg, rgba(15,20,25,0.95) 0%, rgba(26,31,46,0.9) 100%);
            border-bottom: 1px solid rgba(136,146,176,0.15);
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at center, rgba(100,181,246,0.03) 0%, transparent 50%);
            animation: pulse 20s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.5; }}
            50% {{ transform: scale(1.1); opacity: 0.3; }}
        }}
        
        .logo {{
            width: 140px;
            height: 140px;
            margin: 0 auto 35px;
            position: relative;
            z-index: 1;
            background: linear-gradient(135deg, rgba(30,40,55,0.8) 0%, rgba(20,25,35,0.9) 100%);
            border-radius: 24px;
            padding: 15px;
            border: 2px solid rgba(100,181,246,0.2);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5),
                        0 0 60px rgba(100,181,246,0.2),
                        inset 0 0 30px rgba(100,181,246,0.05);
            animation: float 6s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
        }}
        
        .logo img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.5)) 
                    drop-shadow(0 0 20px rgba(100,181,246,0.4));
        }}
        
        h1 {{
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 12px;
            color: #ffffff;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 20px rgba(0,0,0,0.5);
        }}
        
        .subtitle {{
            font-size: 20px;
            color: #64b5f6;
            margin-bottom: 25px;
            font-weight: 500;
            letter-spacing: 2px;
            text-transform: uppercase;
            position: relative;
            z-index: 1;
        }}
        
        .report-meta {{
            font-size: 15px;
            color: #8892a0;
            position: relative;
            z-index: 1;
            line-height: 1.8;
        }}
        
        .report-meta strong {{
            color: #b4bcc8;
            font-weight: 600;
        }}
        
        .section {{
            margin-bottom: 60px;
            animation: fadeInUp 0.8s ease-out;
            position: relative;
        }}
        
        .section::before {{
            content: '';
            position: absolute;
            left: -40px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(180deg, transparent, #64b5f6, transparent);
            opacity: 0.3;
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        h2 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 30px;
            color: #ffffff;
            padding-bottom: 15px;
            border-bottom: 2px solid transparent;
            background: linear-gradient(90deg, #64b5f6 0%, transparent 50%) bottom left no-repeat;
            background-size: 100% 2px;
            letter-spacing: -0.3px;
            position: relative;
        }}
        
        h2::after {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(90deg, #64b5f6, #42a5f5);
            animation: expandWidth 1.5s ease-out forwards;
        }}
        
        @keyframes expandWidth {{
            to {{ width: 100%; }}
        }}
        
        h3 {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 18px;
            color: #81c7f7;
            letter-spacing: -0.2px;
        }}
        
        p {{
            margin-bottom: 18px;
            text-align: left;
            color: #c8d0db;
            line-height: 1.8;
        }}
        
        .finding {{
            background: linear-gradient(135deg, rgba(36,52,71,0.6) 0%, rgba(26,35,50,0.6) 100%);
            border-left: 4px solid #dc3545;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3),
                        inset 0 1px 0 rgba(255,255,255,0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .code-sample {{
            background: #1a1a2e;
            border: 1px solid #16213e;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }}
        
        .code-sample pre {{
            margin: 0;
            background: transparent;
            border: none;
            padding: 0;
        }}
        
        .code-sample pre code {{
            background: transparent !important;
            padding: 0 !important;
            font-size: 13px;
            line-height: 1.5;
        }}
        
        .code-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a3f5f;
        }}
        
        .code-file {{
            color: #64b5f6;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .code-lines {{
            color: #888;
            font-size: 11px;
        }}
        
        .code-explanation {{
            background: rgba(100, 181, 246, 0.1);
            border-left: 3px solid #64b5f6;
            padding: 10px;
            margin-top: 10px;
            font-size: 13px;
            color: #b3d4fc;
            font-style: italic;
        }}
        /* Enhanced code layout: gutter + content + copy */
        .code-grid {{
            display: grid;
            grid-template-columns: auto 1fr;
            column-gap: 12px;
        }}
        .code-gutter {{
            color: #6b7b94;
            text-align: right;
            user-select: none;
            padding-right: 8px;
            border-right: 1px solid #2a3f5f;
        }}
        .code-gutter pre {{ color: inherit; margin: 0; }}
        .code-content {{ overflow-x: auto; }}
        .code-content pre {{ white-space: pre; margin: 0; }}
        .code-line {{ white-space: pre; }}
        .code-header-controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .copy-btn {{
            background: rgba(100,181,246,0.12);
            border: 1px solid rgba(100,181,246,0.35);
            color: #b3d4fc;
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 6px;
            cursor: pointer;
        }}
        .copy-btn:hover {{ background: rgba(100,181,246,0.2); }}
        
        .finding::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(100,181,246,0.5), transparent);
            animation: scan 3s linear infinite;
        }}
        
        @keyframes scan {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        
        .finding:hover {{
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 8px 40px rgba(0,0,0,0.5),
                        inset 0 1px 0 rgba(255,255,255,0.2);
            background: linear-gradient(135deg, rgba(36,52,71,0.7) 0%, rgba(26,35,50,0.7) 100%);
        }}
        
        .finding.critical {{
            border-left-color: #dc3545;
        }}
        
        .finding.high {{
            border-left-color: #fd7e14;
        }}
        
        .finding.medium {{
            border-left-color: #ffc107;
        }}
        
        .finding.low {{
            border-left-color: #28a745;
        }}
        
        .severity-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 12px;
            letter-spacing: 1px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }}
        
        .severity-badge::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
            transform: rotate(45deg);
            animation: shimmer 3s infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
            100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
        }}
        
        .severity-critical {{
            background: #dc3545;
            color: white;
        }}
        
        .severity-high {{
            background: #fd7e14;
            color: white;
        }}
        
        .severity-medium {{
            background: #ffc107;
            color: #333;
        }}
        
        .severity-low {{
            background: #28a745;
            color: white;
        }}
        
        .test-item {{
            padding: 10px 0;
            border-bottom: 1px solid #3a4556;
        }}
        
        .test-item:last-child {{
            border-bottom: none;
        }}
        
        .test-status {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            vertical-align: middle;
            box-shadow: 0 0 10px currentColor;
            animation: pulse-dot 2s infinite;
        }}
        
        @keyframes pulse-dot {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(0.9); }}
        }}
        
        .status-confirmed {{
            background: #dc3545;
        }}
        
        .status-rejected {{
            background: #28a745;
        }}
        
        .status-investigating {{
            background: #ffc107;
        }}
        
        .status-proposed {{
            background: #6c757d;
        }}
        
        .footer {{
            margin-top: 80px;
            padding-top: 40px;
            border-top: 1px solid rgba(136,146,176,0.15);
            text-align: center;
            color: #64738c;
            font-size: 14px;
        }}
        
        .footer p {{
            text-align: center;
            color: #64738c;
        }}
        
        /* Proof of Concept Styles */
        .poc-section {{
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(26,31,46,0.7) 0%, rgba(15,20,25,0.8) 100%);
            border-radius: 12px;
            border: 1px solid rgba(100,181,246,0.2);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3),
                        inset 0 1px 0 rgba(255,255,255,0.05);
        }}
        
        .poc-section h4 {{
            color: #64b5f6;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .poc-file {{
            margin-bottom: 1.5rem;
            background: rgba(20,25,35,0.5);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(100,181,246,0.1);
        }}
        
        .poc-file h5 {{
            color: #90caf9;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }}
        
        .poc-description {{
            color: #8892a9;
            font-style: italic;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}
        
        .poc-file pre {{
            background: rgba(10,12,18,0.7);
            border: 1px solid rgba(100,181,246,0.1);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 0;
        }}
        
        .poc-file code {{
            color: #e0e0e0;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
        }}
        
        /* Statistics Section Styles */
        .stats-section {{
            margin: 60px 0;
            padding: 40px;
            background: linear-gradient(135deg, rgba(26,31,46,0.8) 0%, rgba(15,20,25,0.9) 100%);
            border-radius: 20px;
            border: 1px solid rgba(100,181,246,0.2);
            box-shadow: 0 10px 40px rgba(0,0,0,0.4),
                        inset 0 1px 0 rgba(255,255,255,0.05);
            position: relative;
            overflow: hidden;
        }}
        
        .stats-section::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at center, rgba(100,181,246,0.05) 0%, transparent 50%);
            animation: rotate 30s linear infinite;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
            position: relative;
            z-index: 1;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, rgba(36,52,71,0.7) 0%, rgba(26,35,50,0.7) 100%);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(100,181,246,0.15);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3),
                        inset 0 1px 0 rgba(255,255,255,0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 8px 30px rgba(0,0,0,0.5),
                        inset 0 1px 0 rgba(255,255,255,0.2);
            border-color: rgba(100,181,246,0.3);
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, transparent, rgba(100,181,246,0.4), transparent);
            opacity: 0;
            transition: opacity 0.3s;
            border-radius: 16px;
        }}
        
        .stat-card:hover::before {{
            opacity: 1;
            animation: shimmer-stat 1.5s;
        }}
        
        @keyframes shimmer-stat {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        
        .stat-number {{
            font-size: 48px;
            font-weight: 800;
            margin: 10px 0;
            background: linear-gradient(135deg, #64b5f6 0%, #42a5f5 50%, #81c7f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(100,181,246,0.5);
            animation: pulse-number 2s ease-in-out infinite;
        }}
        
        @keyframes pulse-number {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .stat-number.critical {{
            background: linear-gradient(135deg, #ff6b6b 0%, #dc3545 50%, #ff8787 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(220,53,69,0.5);
        }}
        
        .stat-number.high {{
            background: linear-gradient(135deg, #ffb347 0%, #fd7e14 50%, #ffc371 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(253,126,20,0.5);
        }}
        
        .stat-number.medium {{
            background: linear-gradient(135deg, #ffd93d 0%, #ffc107 50%, #ffe066 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(255,193,7,0.5);
        }}
        
        .stat-number.low {{
            background: linear-gradient(135deg, #6bcf7e 0%, #28a745 50%, #52d869 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(40,167,69,0.5);
        }}
        
        .stat-label {{
            font-size: 14px;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .chart-container {{
            position: relative;
            z-index: 1;
            margin-top: 30px;
        }}
        
        .chart-wrapper {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
        }}
        
        .pie-chart {{
            position: relative;
            width: 250px;
            height: 250px;
        }}
        
        .pie-chart svg {{
            width: 100%;
            height: 100%;
            filter: drop-shadow(0 4px 20px rgba(0,0,0,0.3));
        }}
        
        .chart-legend {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 200px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px;
            background: rgba(26,31,46,0.4);
            border-radius: 8px;
            transition: all 0.3s ease;
        }}
        
        .legend-item:hover {{
            background: rgba(36,52,71,0.6);
            transform: translateX(5px);
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .legend-text {{
            flex: 1;
            color: #c8d0db;
            font-size: 14px;
            font-weight: 500;
        }}
        
        .legend-count {{
            font-weight: 700;
            color: #64b5f6;
            font-size: 16px;
        }}
        
        .summary-title {{
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 30px;
            position: relative;
            z-index: 1;
        }}
        
        .summary-title::after {{
            content: '';
            display: block;
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, transparent, #64b5f6, transparent);
            margin: 15px auto 0;
        }}
        
        ul {{
            margin: 20px 0;
            padding-left: 25px;
        }}
        
        ul li {{
            margin-bottom: 12px;
            color: #c8d0db;
            line-height: 1.8;
        }}
        
        ul li::marker {{
            color: #64b5f6;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3),
                        inset 0 1px 0 rgba(255,255,255,0.05);
            background: rgba(26,35,50,0.3);
        }}
        
        th, td {{
            padding: 16px;
            text-align: left;
        }}
        
        th {{
            background: linear-gradient(135deg, rgba(36,52,71,0.8) 0%, rgba(26,35,50,0.8) 100%);
            font-weight: 600;
            color: #64b5f6;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1px;
        }}
        
        td {{
            background: rgba(26,35,50,0.4);
            border-bottom: 1px solid rgba(58,69,86,0.3);
            color: #c8d0db;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .component-diagram {{
            margin-top: 40px;
            padding: 30px;
            background: linear-gradient(135deg, rgba(36,52,71,0.4) 0%, rgba(26,35,50,0.4) 100%);
            border-radius: 12px;
            border: 1px solid rgba(100,181,246,0.2);
        }}
        
        .component-diagram h3 {{
            color: #81c7f7;
            margin-bottom: 20px;
            font-size: 18px;
            font-weight: 600;
        }}
        
        .diagram-content {{
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            color: #e8ecf1;
            background: rgba(15,20,25,0.6);
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SECURITY AUDIT REPORT</h1>
            <div class="subtitle">{kwargs.get('application_name', kwargs.get('project_name', 'Application'))}</div>
            <div class="report-meta">
                <strong>Project:</strong> {kwargs.get('project_name', '')}<br>
                <strong>Date:</strong> {kwargs['report_date']}<br>
                <strong>Audit Team:</strong> Hound Security<br>
                <strong>Lead Auditors:</strong> {', '.join(kwargs['auditors'])}
            </div>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            {self._format_paragraphs_html(kwargs['executive_summary'])}
            
            <h3 style="margin-top: 30px; color: #64b5f6;">Audit Team</h3>
            {self._generate_models_table_html()}
        </div>
        
        <div class="section">
            <h2>System Overview</h2>
            {self._format_paragraphs_html(kwargs.get('system_overview', ''))}
        </div>
        
        {self._generate_statistics_section_html(kwargs['findings'])}
        
        <div class="section">
            <h2>{'All Hypotheses (UNREVIEWED)' if self.include_all else 'Findings'}</h2>
            {self._add_unreviewed_warning_html() if self.include_all else ''}
            {self._format_findings_html(kwargs['findings'])}
        </div>
        
        <div class="footer">
            <p>© {datetime.now().year} Hound Security Team<br>
            Report prepared by: {kwargs.get('report_writer', 'Hound Team')}<br>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _format_paragraphs_html(self, text: str) -> str:
        """Format text into HTML paragraphs."""
        paragraphs = (text or '').strip().split('\n\n')
        return '\n'.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())
    
    def _format_poc_html(self, poc_data: dict[str, Any]) -> str:
        """Format PoC files as HTML with syntax highlighting."""
        if not poc_data or not poc_data.get('files'):
            return ''
        
        html_parts = ['<div class="poc-section">']
        html_parts.append('<h4>Proof of Concept</h4>')
        
        for poc_file in poc_data['files']:
            name = poc_file['name']
            content = poc_file['content']
            description = poc_file.get('description', '')
            
            # Detect language from file extension
            lang = self._detect_language(name)
            
            # Format the PoC
            html_parts.append('<div class="poc-file">')
            html_parts.append(f'<h5>{self._escape_html(name)}</h5>')
            
            if description:
                html_parts.append(f'<p class="poc-description">{self._escape_html(description)}</p>')
            
            # Render using the same enhanced renderer as findings
            lines_count = max(1, content.count('\n') + 1)
            sample = {
                'file': name,
                'start_line': 1,
                'end_line': lines_count,
                'code': content,
                'language': lang,
                'explanation': ''
            }
            html_parts.append(self._render_code_sample(sample))
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _format_component_diagram_html(self, diagram: str) -> str:
        """Format component diagram into HTML."""
        if not diagram or not diagram.strip():
            return ''
        
        return f'''
        <div class="component-diagram">
            <h3>System Architecture Diagram</h3>
            <pre class="diagram-content">{diagram}</pre>
        </div>
        '''
    
    def _generate_statistics_section_html(self, findings: list[dict]) -> str:
        """Generate beautiful statistics section with charts and numbers."""
        if not findings:
            return ''
        
        # Count by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for f in findings:
            sev = f.get('severity', 'low')
            if sev in severity_counts:
                severity_counts[sev] += 1
        
        total_findings = len(findings)
        
        # Calculate percentages for pie chart
        percentages = {}
        for sev, count in severity_counts.items():
            if count > 0:
                percentages[sev] = (count / total_findings) * 100
        
        # Generate pie chart SVG
        pie_chart_svg = self._generate_pie_chart_svg(severity_counts)
        
        html = f'''
        <div class="stats-section">
            <h2 class="summary-title">Security Analysis Statistics</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_findings}</div>
                    <div class="stat-label">Total Findings</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number critical">{severity_counts["critical"]}</div>
                    <div class="stat-label">Critical Issues</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number high">{severity_counts["high"]}</div>
                    <div class="stat-label">High Severity</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number medium">{severity_counts["medium"]}</div>
                    <div class="stat-label">Medium Severity</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-number low">{severity_counts["low"]}</div>
                    <div class="stat-label">Low Severity</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3 style="text-align: center; color: #81c7f7; margin-bottom: 30px; font-size: 20px;">Severity Distribution</h3>
                <div class="chart-wrapper">
                    <div class="pie-chart">
                        {pie_chart_svg}
                    </div>
                    <div class="chart-legend">
                        {self._generate_legend_html(severity_counts, percentages)}
                    </div>
                </div>
            </div>
        </div>
        '''
        
        return html
    
    def _generate_pie_chart_svg(self, severity_counts: dict[str, int]) -> str:
        """Generate an SVG pie chart for severity distribution."""
        total = sum(severity_counts.values())
        if total == 0:
            return ''
        
        # Colors for each severity
        colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#28a745'
        }
        
        # Calculate angles
        angles = []
        current_angle = -90  # Start from top
        for sev in ['critical', 'high', 'medium', 'low']:
            if severity_counts[sev] > 0:
                angle = (severity_counts[sev] / total) * 360
                angles.append((sev, current_angle, angle))
                current_angle += angle
        
        # Generate SVG paths
        paths = []
        cx, cy, r = 125, 125, 100
        
        for sev, start_angle, sweep_angle in angles:
            # Convert to radians
            start_rad = math.radians(start_angle)
            end_rad = math.radians(start_angle + sweep_angle)
            
            # Calculate arc endpoints
            x1 = cx + r * math.cos(start_rad)
            y1 = cy + r * math.sin(start_rad)
            x2 = cx + r * math.cos(end_rad)
            y2 = cy + r * math.sin(end_rad)
            
            # Large arc flag
            large_arc = 1 if sweep_angle > 180 else 0
            
            # Create path
            path = f'M{cx},{cy} L{x1:.2f},{y1:.2f} A{r},{r} 0 {large_arc},1 {x2:.2f},{y2:.2f} Z'
            
            paths.append(f'''
                <path d="{path}" 
                      fill="{colors[sev]}" 
                      stroke="rgba(255,255,255,0.1)" 
                      stroke-width="2"
                      opacity="0.9">
                    <animate attributeName="opacity" 
                             values="0.9;1;0.9" 
                             dur="3s" 
                             repeatCount="indefinite"/>
                </path>
            ''')
        
        svg = f'''
        <svg viewBox="0 0 250 250" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <g filter="url(#glow)">
                {''.join(paths)}
            </g>
            <circle cx="125" cy="125" r="60" 
                    fill="rgba(15,20,25,0.8)" 
                    stroke="rgba(100,181,246,0.3)" 
                    stroke-width="2"/>
            <text x="125" y="125" 
                  text-anchor="middle" 
                  dominant-baseline="middle" 
                  font-size="32" 
                  font-weight="700" 
                  fill="#64b5f6">{total}</text>
            <text x="125" y="145" 
                  text-anchor="middle" 
                  dominant-baseline="middle" 
                  font-size="12" 
                  font-weight="600" 
                  fill="#8892b0" 
                  text-transform="uppercase" 
                  letter-spacing="1">Issues</text>
        </svg>
        '''
        
        return svg
    
    def _generate_legend_html(self, severity_counts: dict[str, int], percentages: dict[str, float]) -> str:
        """Generate legend HTML for the pie chart."""
        colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#28a745'
        }
        
        items = []
        for sev in ['critical', 'high', 'medium', 'low']:
            if severity_counts[sev] > 0:
                items.append(f'''
                <div class="legend-item">
                    <div class="legend-color" style="background: {colors[sev]};"></div>
                    <div class="legend-text">{sev.capitalize()}</div>
                    <div class="legend-count">{severity_counts[sev]}</div>
                    <div style="color: #64738c; font-size: 12px;">({percentages[sev]:.1f}%)</div>
                </div>
                ''')
        
        return '\n'.join(items)
    
    def _add_unreviewed_warning_html(self) -> str:
        """Add a warning box for unreviewed findings."""
        return """
        <div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h3 style="color: #856404; margin-top: 0;">⚠️ WARNING: Unreviewed Findings</h3>
            <p style="color: #856404; margin-bottom: 0;">
                This report includes ALL hypotheses generated during the analysis, not just confirmed findings.
                No quality assurance or review process has been performed. These findings may contain false positives
                and should be independently verified before taking action.
            </p>
        </div>
        """
    
    def _format_findings_html(self, findings: list[dict]) -> str:
        """Format findings into HTML with code samples."""
        if not findings:
            if self.include_all:
                return '<p><em>No hypotheses were generated during this analysis.</em></p>'
            else:
                return '<p><em>No confirmed vulnerabilities were identified during this audit.</em></p>'
        
        html_parts = []
        for finding in findings:
            severity = finding['severity']
            
            # Format code samples
            code_html = ''
            for sample in finding.get('code_samples', []):
                code_html += self._render_code_sample(sample)
            
            # Add QA comment if available
            qa_comment_html = ''
            if finding.get('qa_comment'):
                qa_comment_html = f'''
                <div class="qa-comment" style="margin-top: 1em; padding: 15px; background: rgba(70, 130, 180, 0.1); border-left: 3px solid #4682b4; border-radius: 8px;">
                    <strong style="color: #64b5f6;">QA Review:</strong> 
                    <span style="color: #c3cfe2; line-height: 1.6;">{self._escape_html(finding['qa_comment'])}</span>
                </div>
                '''
            
            # Check if there's a PoC for this hypothesis
            poc_html = ''
            hypothesis_id = finding.get('id', '')
            if hypothesis_id and hypothesis_id in self.pocs:
                poc_data = self.pocs[hypothesis_id]
                poc_html = self._format_poc_html(poc_data)
            
            html_parts.append(f'''
            <div class="finding {severity}">
                <span class="severity-badge severity-{severity}">{severity}</span>
                <h3>{finding['title']}</h3>
                <div class="vulnerability-description">
                    {self._format_paragraphs_html(finding.get('professional_description', finding['description']))}
                </div>
                <p><strong>Affected Components:</strong> {finding.get('affected_description', self._describe_affected_components(finding.get('affected', [])))}</p>
                {qa_comment_html}
                {code_html}
                {poc_html}
            </div>
            ''')
        
        return '\n'.join(html_parts)
    
    def _describe_affected_components(self, node_refs: list[str]) -> str:
        """Generate human-readable descriptions of affected components."""
        if not node_refs:
            return "Various system components"
        
        # Just pass the raw node references to the LLM
        return ', '.join(node_refs[:3])  # Will be formatted by LLM in batch
    
    def _batch_generate_vulnerability_descriptions(self, findings: list[dict]) -> dict[int, str]:
        """Batch generate professional vulnerability descriptions using LLM."""
        if not findings:
            return {}
        
        # Build a single prompt for all vulnerabilities
        vulnerabilities_json = []
        for i, finding in enumerate(findings):
            vulnerabilities_json.append({
                "index": i,
                "title": finding.get('title', 'Unknown'),
                "type": finding.get('type', 'unknown'),
                "severity": finding.get('severity', 'medium'),
                "affected_components_raw": finding.get('affected', [])[:3],
                "raw_description": finding.get('description', '')
            })
        
        prompt = f"""You are writing a professional security audit report. Convert these raw vulnerability data into clear, professional descriptions.

Vulnerabilities to describe:
{json.dumps(vulnerabilities_json, ensure_ascii=False, indent=2)}

For EACH vulnerability, provide:
1. A professional description (2-3 paragraphs) that:
   - Naturally mentions the vulnerability type (logic error, access control, etc.) in the first sentence
   - Explains the root cause and technical details
   - Describes the attack vector and potential impact
2. A human-readable description of the affected components

Return a JSON object with this structure:
{{
  "0": {{
    "description": "Professional vulnerability description...",
    "affected_components": "the AgentOwnerRegistry contract, specifically the setWorkAddress() function"
  }},
  "1": {{
    "description": "Professional vulnerability description...", 
    "affected_components": "the MintingFacet contract"
  }},
  ...
}}

Rules for descriptions:
- Write in third person, professional tone
- Be concise but thorough
- DO NOT include metadata prefixes like "VULNERABILITY TYPE:", "ROOT CAUSE:", etc.
- DO NOT mention discovery methods or analysis process

Rules for affected components:
- Convert raw node names like "func_MintingFacet__performMinting" to readable format like "the MintingFacet contract, specifically the performMinting() function"
- Use proper articles and grammar
- Group related components naturally (contracts first, then specific functions)
- Examples:
  - ["AgentOwnerRegistry"] → "the AgentOwnerRegistry contract"
  - ["func_mint__publicMint", "MintingFacet"] → "the MintingFacet contract, specifically the publicMint() function"
  - ["Contract1", "Contract2"] → "the Contract1 and Contract2 contracts"
"""

        try:
            response = self.llm.raw(
                system="You are a security expert writing clear vulnerability descriptions. Respond only with valid JSON.",
                user=prompt
            )
            from utils.json_utils import extract_json_object
            results = extract_json_object(response)
            
            if isinstance(results, dict):
                # Extract both descriptions and component formatting
                processed = {}
                for k, v in results.items():
                    idx = int(k)
                    if isinstance(v, dict):
                        processed[idx] = {
                            'description': v.get('description', ''),
                            'affected_components': v.get('affected_components', '')
                        }
                    else:
                        # Fallback for old format
                        processed[idx] = {
                            'description': v,
                            'affected_components': ''
                        }
                return processed
            else:
                return {}
        except Exception as e:
            if self.debug:
                print(f"[!] Failed to batch generate descriptions: {e}")
            return {}
    
    def _generate_vulnerability_description(self, finding: dict) -> str:
        """Generate a professional vulnerability description using LLM."""
        
        # Extract key information from the raw description
        raw_desc = finding.get('description', '')
        title = finding.get('title', 'Unknown')
        vuln_type = finding.get('type', 'unknown')
        severity = finding.get('severity', 'medium')
        affected = finding.get('affected', [])
        affected_desc = self._describe_affected_components(affected)
        
        prompt = f"""You are writing a professional security audit report. Convert this raw vulnerability data into a clear, professional description.

Vulnerability Title: {title}
Type: {vuln_type}
Severity: {severity}
Affected Components: {affected_desc}

Raw Technical Details:
{raw_desc}

Write a professional vulnerability description (2-3 paragraphs) that:
1. Naturally mentions this is a {vuln_type} vulnerability in the first sentence
2. Clearly explains the root cause and technical details
3. Describes the potential attack vector and impact

Rules:
- Write in third person, professional tone
- Be concise but thorough
- Focus on technical accuracy
- DO NOT include metadata like "VULNERABILITY TYPE:", "ROOT CAUSE:", etc.
- DO NOT repeat the same information multiple times
- DO NOT mention discovery methods or analysis process
- Present as a clean, professional finding description
"""

        try:
            description = self.llm.raw(
                system="You are a security expert writing clear vulnerability descriptions.",
                user=prompt
            )
            return description.strip()
        except Exception as e:
            if self.debug:
                print(f"[!] Failed to generate description for {title}: {e}")
            # Fallback: clean up the raw description
            return self._clean_raw_description(raw_desc)
    
    def _clean_raw_description(self, raw_desc: str) -> str:
        """Clean up raw description as fallback."""
        # Remove metadata prefixes
        lines = raw_desc.split('\n')
        cleaned = []
        
        for line in lines:
            # Skip lines that are just metadata labels
            if any(line.startswith(prefix) for prefix in 
                   ['VULNERABILITY TYPE:', 'ROOT CAUSE:', 'ATTACK VECTOR:', 
                    'AFFECTED NODES:', 'AFFECTED CODE:', 'SEVERITY:', 'REASONING:',
                    'Location:', 'Technical Details:']):
                continue
            # Skip duplicate content lines
            if line.strip() and not any(c.strip() == line.strip() for c in cleaned):
                cleaned.append(line)
        
        result = ' '.join(cleaned)
        # Clean up excessive whitespace
        result = ' '.join(result.split())
        
        return result if result else raw_desc
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))

    def _dedent_code(self, code: str) -> str:
        """Remove common leading indentation from a code block while preserving relative indent."""
        lines = code.split('\n')
        indents: list[int] = []
        for ln in lines:
            if ln.strip() == '':
                continue
            expanded = ln.replace('\t', '    ')
            indents.append(len(expanded) - len(expanded.lstrip(' ')))
        if not indents:
            return code
        trim = min(indents)
        if trim <= 0:
            return code
        out_lines: list[str] = []
        for ln in lines:
            expanded = ln.replace('\t', '    ')
            out_lines.append(expanded[trim:] if len(expanded) >= trim else expanded.lstrip(' '))
        return '\n'.join(out_lines)

    def _render_code_sample(self, sample: dict) -> str:
        """Render a single code sample with line numbers and copy button."""
        try:
            fpath = str(sample.get('file') or 'unknown')
            start = int(sample.get('start_line') or 1)
            end = int(sample.get('end_line') or start)
            lang = str(sample.get('language') or self._detect_language(fpath))
            raw_code = str(sample.get('code') or '')
        except Exception:
            fpath, start, end, lang, raw_code = 'unknown', 1, 1, 'plaintext', str(sample)
        code = self._dedent_code(raw_code.rstrip('\n'))
        lines = code.split('\n')
        gutter_numbers = '\n'.join(str(start + i) for i in range(len(lines)))
        self._code_block_counter += 1
        code_id = f"code-block-{self._code_block_counter}"
        explanation_html = ''
        if sample.get('explanation'):
            explanation_html = f'<div class="code-explanation">{self._escape_html(str(sample.get("explanation","")))}</div>'
        return f'''
        <div class="code-sample">
            <div class="code-header">
                <span class="code-file">{self._escape_html(fpath)}</span>
                <div class="code-header-controls">
                    <span class="code-lines">Lines {start}-{end}</span>
                    <button class="copy-btn" onclick="(async()=>{{try{{await navigator.clipboard.writeText(document.getElementById('{code_id}').innerText);}}catch(e){{}}}})()">Copy</button>
                </div>
            </div>
            <div class="code-grid">
                <div class="code-gutter"><pre>{gutter_numbers}</pre></div>
                <div class="code-content"><pre id="{code_id}"><code class="language-{lang}">{self._escape_html(code)}</code></pre></div>
            </div>
            {explanation_html}
        </div>
        '''
    
    def _format_test_coverage_html(self, hypotheses: list[dict]) -> str:
        """Format test coverage into HTML."""
        if not hypotheses:
            return '<p><em>No security tests were recorded.</em></p>'
        
        html_parts = ['<div class="test-coverage">']
        
        # Group by type
        by_type = {}
        for hyp in hypotheses:
            vuln_type = hyp['type']
            if vuln_type not in by_type:
                by_type[vuln_type] = []
            by_type[vuln_type].append(hyp)
        
        for vuln_type, items in by_type.items():
            html_parts.append(f'<h3>{vuln_type.replace("_", " ").title()}</h3>')
            for item in items:
                status_class = f"status-{item['status']}"
                html_parts.append(f'''
                <div class="test-item">
                    <span class="test-status {status_class}"></span>
                    <strong>{item['title']}</strong>
                    <span style="color: #888; margin-left: 10px;">
                        ({item['status']} - {item['confidence']:.0%} confidence)
                    </span>
                </div>
                ''')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _generate_markdown_report(self, **kwargs) -> str:
        """Generate Markdown format report."""
        
        md = f"""# SECURITY AUDIT REPORT

**{kwargs.get('application_name', kwargs.get('project_name', 'Application'))}**

**Date:** {kwargs['report_date']}  
**Performed by:** {', '.join(kwargs['auditors'])}

---

## Executive Summary

{kwargs['executive_summary']}

## System Overview

{kwargs.get('system_overview', '')}

## Scope & Methodology

### Project Information

| Field | Value |
|-------|-------|
| Project Name | {kwargs['project_name']} |
| Repository | {kwargs['project_source']} |
| Audit Date | {kwargs['report_date']} |
| Auditors | {', '.join(kwargs['auditors'])} |

### Methodology

The audit employed a comprehensive security assessment methodology including:

- Static code analysis and manual code review
- Architectural security assessment  
- Attack surface mapping and threat modeling
- Vulnerability pattern matching and invariant analysis
- External dependency and integration review

## Findings

{self._format_findings_markdown(kwargs['findings'])}

---

*Generated by Hound Security Analysis Platform*  
*© {datetime.now().year} - Security Report*
"""
        
        return md
    
    def _format_findings_markdown(self, findings: list[dict]) -> str:
        """Format findings for Markdown."""
        if not findings:
            return '*No confirmed vulnerabilities were identified during this audit.*'
        
        md_parts = []
        for finding in findings:
            # Include QA comment if available
            qa_comment = ''
            if finding.get('qa_comment'):
                qa_comment = f"\n\n**QA Review:** {finding['qa_comment']}"
            
            md_parts.append(f"""### [{finding['severity'].upper()}] {finding['title']}

**Affected:** {finding.get('affected_description', self._describe_affected_components(finding.get('affected', [])))}  

{finding.get('professional_description', finding['description'])}{qa_comment}

---""")
        
        return '\n\n'.join(md_parts)
    
    def _extract_code_for_finding(self, finding: dict) -> list[dict]:
        """Extract relevant code snippets for a finding.

        Priority order:
        1) Use graph node -> card -> source slice mappings for exact snippets
        2) Prefer letting the reporting model select concise snippets with reasons
        3) Fall back to file-level LLM selection if mapping unavailable
        """
        # 1) Try graph/card evidence mapping to identify affected files
        files_ctx = self._collect_files_from_cards(finding)
        if files_ctx:
            # 2) Ask reporting model to pick small, relevant snippets with explanations
            picked = self._select_snippets_with_reporting_llm(finding, files_ctx)
            if picked:
                return picked
            # If model selection fails, fall back to direct card slices (limited)
            mapped_samples = self._extract_code_via_cards(finding)
            if mapped_samples:
                return mapped_samples

        # 3) Fallback to previous file-level LLM approach
        return self._extract_code_via_llm_file_scan(finding)

    def _collect_files_from_cards(self, finding: dict) -> dict[str, str]:
        """Return map of relpath -> full file content for files referenced by evidence.

        Sources considered (in order):
        - Graph node card refs → card.relpath
        - Hypothesis properties.source_files (set by agent)
        - Supporting evidence file hints (e.g., {'file': 'src/x.rs'})
        """
        files: dict[str, str] = {}
        try:
            # Need a repo root to load code
            if not self.repo_root or not self.repo_root.exists():
                return files

            rels: list[str] = []

            # 1) From graph nodes referenced by the finding
            target_ids = set(str(x) for x in finding.get('affected', []) if x)
            if target_ids:
                for graph in self.graphs.values():
                    for node in graph.get('nodes', []):
                        nid = str(node.get('id') or '')
                        nlabel = str(node.get('label') or '')
                        if nid in target_ids or nlabel in target_ids:
                            refs = node.get('source_refs') or node.get('refs') or []
                            if isinstance(refs, list):
                                for r in refs:
                                    c = self.card_store.get(str(r)) if self.card_store else None
                                    if isinstance(c, dict) and c.get('relpath'):
                                        rels.append(c['relpath'])

            # 2) From hypothesis properties (source_files)
            try:
                for sf in (finding.get('properties') or {}).get('source_files', []) or []:
                    if isinstance(sf, str) and sf:
                        rels.append(sf)
            except Exception:
                pass

            # 3) From supporting evidence structures
            try:
                for ev in finding.get('supporting_evidence', []) or []:
                    if isinstance(ev, dict):
                        for key in ('file', 'location'):
                            val = ev.get(key)
                            if isinstance(val, str) and val:
                                rels.append(val)
            except Exception:
                pass

            # Dedup preserving order
            seen = set()
            ordered_rels = []
            for r in rels:
                if not r:
                    continue
                # Normalize path style a bit
                rr = str(r).lstrip('/')
                if rr not in seen:
                    seen.add(rr)
                    ordered_rels.append(rr)

            # Load file contents
            for rel in ordered_rels:
                fpath = self.repo_root / rel
                # Heuristics: also try 'src/<rel>' and drop 'src/' prefix
                cand_paths = [fpath]
                if not fpath.exists():
                    if rel.startswith('src/'):
                        cand_paths.append(self.repo_root / rel[4:])
                    else:
                        cand_paths.append(self.repo_root / 'src' / rel)
                loaded = False
                for cand in cand_paths:
                    try:
                        if cand.exists():
                            content = cand.read_text(encoding='utf-8', errors='ignore')
                            # Use the original rel key for stability
                            files[rel] = content
                            loaded = True
                            break
                    except Exception:
                        continue
                if not loaded:
                    continue

            # Limit to 3 files for token control
            if len(files) > 3:
                # Keep the first 3 in discovered order
                keep = list(files.keys())[:3]
                return {k: files[k] for k in keep}
            return files
        except Exception:
            return {}

    def _select_snippets_with_reporting_llm(self, finding: dict, files_ctx: dict[str, str]) -> list[dict]:
        """Use reporting LLM to select concise snippets across provided files with explanations.
        Strongly prefer functions explicitly referenced by the finding.
        """
        if not files_ctx:
            return []
        try:
            title = finding.get('title', 'Unknown')
            ftype = finding.get('type', 'unknown')
            desc = finding.get('professional_description') or finding.get('description') or ''
            target_funcs = sorted(list(self._derive_target_functions(finding)))
            func_index = self._index_functions(files_ctx)
            payload = {
                'finding': {
                    'title': title,
                    'type': ftype,
                    'description': desc,
                },
                'files': [{'path': p, 'content': c} for p, c in files_ctx.items()],
                'target_functions': target_funcs,
                'functions_index': func_index
            }
            system = "You are a senior security auditor preparing a report. Return only valid JSON."
            user = (
                "You are given a vulnerability finding and a set of source files.\n"
                "Pick up to 3 short code snippets (5–18 lines each) that BEST illustrate the vulnerability.\n"
                "You are provided with an index of functions detected per file and a list of TARGET function names (if any).\n"
                "STRICT PREFERENCE: choose snippets from TARGET functions if they are present.\n"
                "Avoid constructors unless they are explicitly listed as TARGET functions.\n"
                "If TARGET functions are not present, choose the function(s) most directly responsible.\n"
                "Always include the matching function header line in the snippet.\n"
                "If multiple files are relevant, you may select at most one snippet per file.\n"
                "Avoid redundant or overly long snippets.\n\n"
                f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Return JSON with this shape:\n"
                "{\n  \"snippets\": [\n    {\n      \"file\": <path>,\n      \"start_line\": <1-based int>,\n      \"end_line\": <1-based int>,\n      \"explanation\": <why this snippet is relevant>\n    }\n  ]\n}\n"
                "Constraints:\n- Ensure line numbers correspond to the provided file content.\n- Keep each snippet under 20 lines and maximize signal.\n- If nothing clearly relevant is present, return an empty snippets array."
            )
            response = self.llm.raw(system=system, user=user)
            from utils.json_utils import extract_json_object
            obj = extract_json_object(response)
            results: list[dict] = []
            if isinstance(obj, dict) and isinstance(obj.get('snippets'), list):
                for s in obj['snippets'][:3]:
                    fpath = s.get('file')
                    start = int(s.get('start_line', 0) or 0)
                    end = int(s.get('end_line', 0) or 0)
                    expl = str(s.get('explanation') or '').strip()
                    # Normalize path reported by the model
                    norm = self._normalize_reported_path(str(fpath) if fpath else '', files_ctx)
                    if not norm:
                        continue
                    if start <= 0 or end <= 0 or end < start:
                        continue
                    lines = files_ctx[norm].split('\n')
                    start0 = max(0, start - 1)
                    end0 = min(len(lines), end)
                    # Enforce max 20 lines
                    if end0 - start0 > 20:
                        end0 = start0 + 20
                    code = '\n'.join(lines[start0:end0])
                    results.append({
                        'file': norm,
                        'start_line': start0 + 1,
                        'end_line': end0,
                        'code': code,
                        'language': self._detect_language(norm),
                        'explanation': expl or 'Relevant to the vulnerability as selected by analysis.'
                    })
            # Validate against target functions; if mismatch and we have targets, use deterministic extraction
            if results and self._derive_target_functions(finding):
                if not self._snippets_match_targets(results, func_index, self._derive_target_functions(finding)):
                    det = self._deterministic_snippets_by_function(files_ctx, func_index, self._derive_target_functions(finding))
                    if det:
                        return det
            return results
        except Exception:
            return []

    def _derive_target_functions(self, finding: dict) -> set:
        """Collect intended function names from hypothesis/finding metadata and affected nodes."""
        names = set()
        try:
            # From properties
            props = finding.get('properties') or {}
            for fn in props.get('affected_functions', []) or []:
                # Normalize: strip qualifiers and ()
                base = str(fn).split('.')[-1].strip()
                base = base.replace('()', '')
                if base:
                    names.add(base)
            # From affected_description like "specifically the foo() function"
            desc = finding.get('affected_description') or ''
            import re
            for m in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(\)', desc):
                names.add(m)
            # From affected node refs via graphs (function-type nodes)
            targets = set(str(x) for x in finding.get('affected', []) if x)
            for graph in self.graphs.values():
                for node in graph.get('nodes', []):
                    if (node.get('id') in targets) or (node.get('label') in targets):
                        ntype = (node.get('type') or '').lower()
                        if ntype == 'function':
                            label = node.get('label') or node.get('id') or ''
                            # Try to extract final token as function name
                            # Patterns: Contract.function, func_Contract__function, function
                            parts = re.split(r'[\._]', label)
                            candidate = parts[-1]
                            candidate = candidate.replace('()', '')
                            if candidate:
                                names.add(candidate)
        except Exception:
            pass
        return names

    def _index_functions(self, files_ctx: dict[str, str]) -> dict[str, list[dict[str, int]]]:
        """Build a coarse function index per file for several languages.

        Supports: Solidity, Rust, Python, Go, JS/TS.
        Uses simple regexes and defines function end as the next header or EOF.
        """
        import re
        idx: dict[str, list[dict[str, int]]] = {}

        # Regex patterns by extension
        patterns = {
            '.sol': [
                (re.compile(r'\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('), 'function'),
                (re.compile(r'\bconstructor\s*\('), 'constructor')
            ],
            '.rs': [
                (re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('), 'function')
            ],
            '.py': [
                (re.compile(r'^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('), 'function')
            ],
            '.go': [
                (re.compile(r'^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\('), 'function')
            ],
            '.js': [
                (re.compile(r'^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('), 'function'),
                (re.compile(r'^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?function\s*\('), 'function'),
                (re.compile(r'^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'), 'function')
            ],
            '.ts': [
                (re.compile(r'^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('), 'function'),
                (re.compile(r'^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?function\s*\('), 'function'),
                (re.compile(r'^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'), 'function')
            ],
        }

        for path, content in files_ctx.items():
            ext = Path(path).suffix.lower()
            pats = patterns.get(ext, [])
            if not pats:
                # Default: try common C-style function pattern as a very rough fallback
                pats = [(re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_\*\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('), 'function')]
            lines = content.split('\n')
            headers: list[tuple[int, str, str]] = []
            for i, line in enumerate(lines, start=1):
                for rx, kind in pats:
                    m = rx.search(line)
                    if m:
                        name = m.group(1) if m.groups() else 'constructor'
                        headers.append((i, kind, name))
                        break
            entries = []
            for j, (start, kind, name) in enumerate(headers):
                end = len(lines)
                if j + 1 < len(headers):
                    end = headers[j + 1][0] - 1
                entries.append({'name': name, 'kind': kind, 'start_line': start, 'end_line': end})
            if entries:
                idx[path] = entries
        return idx

    def _snippets_match_targets(self, snippets: list[dict], func_index: dict[str, list[dict[str, int]]], targets: set) -> bool:
        """Check if at least one snippet falls within a targeted function."""
        if not targets:
            return True
        for sn in snippets:
            path = sn.get('file')
            start = sn.get('start_line')
            end = sn.get('end_line')
            if not path or path not in func_index:
                continue
            for e in func_index[path]:
                if e['start_line'] <= start <= e['end_line'] and e['start_line'] <= end <= e['end_line']:
                    if e['name'] in targets or (e.get('kind') == 'constructor' and 'constructor' in targets):
                        return True
        return False

    def _deterministic_snippets_by_function(self, files_ctx: dict[str, str], func_index: dict[str, list[dict[str, int]]], targets: set) -> list[dict]:
        """If targets are available, cut short snippets directly from those function bodies."""
        results: list[dict] = []
        for path, entries in func_index.items():
            for e in entries:
                if e['name'] in targets or (e.get('kind') == 'constructor' and 'constructor' in targets):
                    lines = files_ctx[path].split('\n')
                    start0 = e['start_line'] - 1
                    end0 = min(e['end_line'], e['start_line'] + 18)  # cap ~18 lines
                    code = '\n'.join(lines[start0:end0])
                    results.append({
                        'file': path,
                        'start_line': start0 + 1,
                        'end_line': end0,
                        'code': code,
                        'language': self._detect_language(path),
                        'explanation': f"Relevant function '{e['name']}' referenced by the finding."
                    })
                    if len(results) >= 3:
                        return results
        return results

    def _extract_code_via_cards(self, finding: dict) -> list[dict]:
        """Use node->card mappings to pull precise source slices.
        Merges adjacent/overlapping card ranges per file to avoid duplicates.
        """
        try:
            if not self.card_store or not self.repo_root or not self.repo_root.exists():
                return []
            # Collect affected node identifiers from hypothesis
            target_ids = set(str(x) for x in finding.get('affected', []) if x)
            if not target_ids:
                return []
            # Gather card IDs from matching nodes across all graphs
            card_ids: list[str] = []
            matched_nodes = 0
            for graph in self.graphs.values():
                for node in graph.get('nodes', []):
                    nid = str(node.get('id') or '')
                    nlabel = str(node.get('label') or '')
                    if nid in target_ids or nlabel in target_ids:
                        refs = node.get('source_refs') or node.get('refs') or []
                        if isinstance(refs, list):
                            for r in refs:
                                if r and isinstance(r, str):
                                    card_ids.append(r)
                        matched_nodes += 1
            if not card_ids and self.debug:
                print(f"[!] No card refs found for affected nodes: {sorted(list(target_ids))}")
            # Group ranges per file
            per_file: dict[str, list[tuple]] = {}
            for cid in card_ids:
                c = self.card_store.get(cid)
                if not isinstance(c, dict):
                    continue
                rel = c.get('relpath')
                cs = c.get('char_start')
                ce = c.get('char_end')
                if not rel or not isinstance(cs, int) or not isinstance(ce, int) or ce <= cs:
                    continue
                per_file.setdefault(rel, []).append((cs, ce))

            samples: list[dict] = []
            for rel, ranges in per_file.items():
                fpath = self.repo_root / rel
                try:
                    text = fpath.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    # Fallback: try to stitch from stored card content if file unreadable
                    text = None
                # Merge overlapping/adjacent ranges
                merged = []
                for (cs, ce) in sorted(ranges):
                    if not merged:
                        merged.append([cs, ce])
                    else:
                        last = merged[-1]
                        if cs <= last[1] + 80:  # merge if close/overlapping
                            last[1] = max(last[1], ce)
                        else:
                            merged.append([cs, ce])
                # Convert to snippets (limit per file)
                for cs, ce in merged[:2]:
                    if text is not None:
                        # Expand to whole line boundaries for readability and determinism
                        line_start_idx = text.rfind('\n', 0, max(0, cs)) + 1
                        line_end_idx = text.find('\n', min(len(text), max(cs, ce)))
                        if line_end_idx == -1:
                            line_end_idx = len(text)
                        # Compute 1-based line numbers
                        start_line = 1 + text.count('\n', 0, line_start_idx)
                        end_line = 1 + text.count('\n', 0, line_end_idx)
                        code = text[line_start_idx:line_end_idx]
                    else:
                        # Fall back to concatenating card contents that intersect this range
                        parts = []
                        for cid, card in self.card_store.items():
                            if card.get('relpath') == rel:
                                ccs = card.get('char_start')
                                cce = card.get('char_end')
                                if isinstance(ccs, int) and isinstance(cce, int):
                                    if not (cce <= cs or ccs >= ce):
                                        parts.append(card.get('content') or '')
                        code = '\n'.join(p for p in parts if p).strip()
                        # Unknown exact lines without file; mark as unknown bounds
                        start_line, end_line = 1, max(1, code.count('\n') + 1)
                    samples.append({
                        'file': rel,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': code.rstrip('\n'),
                        'language': self._detect_language(rel),
                        'explanation': 'Merged evidence from graph node → source mapping'
                    })
                    if len(samples) >= 3:
                        break
                if len(samples) >= 3:
                    break
            return samples
        except Exception:
            return []

    def _char_range_to_lines(self, text: str, start: int, end: int) -> tuple:
        """Translate character offsets to 1-based line numbers inclusive."""
        # Clamp inputs
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        # Count newlines up to positions
        before = text[:start]
        within = text[start:end]
        start_line = 1 + before.count('\n')
        end_line = start_line + within.count('\n')
        return start_line, end_line

    def _normalize_reported_path(self, reported: str, files_ctx: dict[str, str]) -> str | None:
        """Map an LLM-reported file path to a key in files_ctx.

        Accepts exact match, suffix matches, and basename matches when unambiguous.
        Returns the normalized key or None if no match.
        """
        if not reported:
            return None
        # Exact
        if reported in files_ctx:
            return reported
        # Normalize leading slash
        cand = reported.lstrip('/')
        if cand in files_ctx:
            return cand
        # Try suffix match
        best_key = None
        best_len = -1
        for key in files_ctx.keys():
            if key.endswith(cand) or cand.endswith(key):
                match_len = len(key) if cand.endswith(key) else len(cand)
                if match_len > best_len:
                    best_key = key
                    best_len = match_len
        if best_key:
            return best_key
        # Basename unique match
        from pathlib import Path as _P
        base = _P(cand).name
        if base:
            matches = [k for k in files_ctx.keys() if _P(k).name == base]
            if len(matches) == 1:
                return matches[0]
        return None

    def _extract_code_via_llm_file_scan(self, finding: dict) -> list[dict]:
        """Fallback: scan likely files and ask LLM for relevant line ranges."""
        code_samples: list[dict] = []
        
        # First check if the finding itself has source_files in properties
        if 'properties' in finding and 'source_files' in finding.get('properties', {}):
            affected_files = set(finding['properties']['source_files'])
        else:
            affected_files = set()
        
        # Also get affected files from node_refs via graph nodes
        for node_ref in finding.get('affected', []):
            if '/' in node_ref or '.sol' in node_ref:
                affected_files.add(node_ref)
            else:
                for graph in self.graphs.values():
                    for node in graph.get('nodes', []):
                        if node.get('label') == node_ref or node.get('id') == node_ref:
                            if 'file' in node:
                                affected_files.add(node['file'])
                            elif 'source' in node:
                                affected_files.add(node['source'])
                            elif 'properties' in node and 'file' in node['properties']:
                                affected_files.add(node['properties']['file'])
        
        # Also check supporting evidence for file references
        for evidence in finding.get('supporting_evidence', []):
            if isinstance(evidence, dict):
                if 'file' in evidence:
                    affected_files.add(evidence['file'])
                if 'location' in evidence:
                    affected_files.add(evidence['location'])
        
        if not affected_files and self.debug:
            print(f"[!] No source files found for finding: {finding.get('title', 'Unknown')}")
        
        # Determine source base path
        source_base_path = self.repo_root
        if not source_base_path or not source_base_path.exists():
            # Last-resort heuristics
            project_file = self.project_dir / "project.json"
            if project_file.exists():
                try:
                    with open(project_file) as f:
                        project_data = json.load(f)
                        source_base_path = Path(project_data.get('source_path', ''))
                except Exception:
                    source_base_path = None
        if not source_base_path or not source_base_path.exists():
            return code_samples
        
        # Ask LLM to identify relevant lines per file
        for file_path in list(affected_files)[:3]:
            try:
                source_path = source_base_path / file_path
                if not source_path.exists():
                    if file_path.startswith('src/'):
                        source_path = source_base_path / file_path[4:]
                    elif not file_path.startswith('/'):
                        source_path = source_base_path / 'src' / file_path
                    if not source_path.exists():
                        continue
                file_content = source_path.read_text(encoding='utf-8', errors='ignore')
                prompt = f"""Given this security finding and the source code, identify the most relevant code section.

Finding Title: {finding['title']}
Finding Type: {finding['type']}
Description: {finding['description']}

Source File: {file_path}
File Content:
```
{file_content}
```

Return a JSON object with:
{{
  "relevant_lines": {{"start": <line_number>, "end": <line_number>}},
  "explanation": "Brief explanation of why this code is relevant"
}}

Only include the most relevant 10-20 lines that directly relate to the vulnerability."""
                response = self.llm.raw(
                    system="You are a code analysis expert. Return only valid JSON.",
                    user=prompt
                )
                from utils.json_utils import extract_json_object
                result = extract_json_object(response)
                if result and 'relevant_lines' in result:
                    lines = file_content.split('\n')
                    start = max(1, int(result['relevant_lines']['start'])) - 1
                    end = max(start + 1, int(result['relevant_lines']['end']))
                    code_samples.append({
                        'file': str(file_path),
                        'start_line': start + 1,
                        'end_line': end,
                        'code': '\n'.join(lines[start:end]),
                        'language': self._detect_language(file_path),
                        'explanation': result.get('explanation', '')
                    })
            except Exception:
                continue
        return code_samples
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            '.sol': 'solidity',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.py': 'python',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.vyper': 'python',
            '.vy': 'python'
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'plaintext')
    
    def _format_test_coverage_markdown(self, hypotheses: list[dict]) -> str:
        """Format test coverage for Markdown."""
        if not hypotheses:
            return '*No security tests were recorded.*'
        
        md_parts = []
        
        # Group by type
        by_type = {}
        for hyp in hypotheses:
            vuln_type = hyp['type']
            if vuln_type not in by_type:
                by_type[vuln_type] = []
            by_type[vuln_type].append(hyp)
        
        for vuln_type, items in by_type.items():
            md_parts.append(f"### {vuln_type.replace('_', ' ').title()}\n")
            for item in items:
                status_icon = {
                    'confirmed': '🔴',
                    'rejected': '✅',
                    'investigating': '🟡',
                    'proposed': '⚫'
                }.get(item['status'], '⚫')
                
                md_parts.append(f"- {status_icon} **{item['title']}** "
                              f"*({item['status']} - {item['confidence']:.0%})*")
            md_parts.append("")
        
        return '\n'.join(md_parts)
