"""Strategist (senior) planner.

Phase 2 introduces a minimal Strategist that can compose planning prompts and
return structured plan items. The CLI will wire this in a later step.
"""

from typing import Any

from pydantic import BaseModel, Field

from llm.token_tracker import get_token_tracker
from llm.tokenization import count_tokens
from llm.unified_client import UnifiedLLMClient


class PlanItemSchema(BaseModel):
    goal: str = Field(description="Investigation goal or question")
    focus_areas: list[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10, description="1-10, 10 = highest")
    reasoning: str = Field(default="", description="Why this is promising")
    category: str = Field(default="aspect", description="aspect | suspicion")
    expected_impact: str = Field(default="medium", description="high | medium | low")


class PlanBatch(BaseModel):
    investigations: list[PlanItemSchema] = Field(default_factory=list)


class HypothesisItemJSON(BaseModel):
    title: str
    type: str = Field(default="security_issue")
    root_cause: str = Field(default="")
    attack_vector: str = Field(default="")
    node_ids: list[str] = Field(default_factory=list)
    affected_code: list[str] = Field(default_factory=list)
    severity: str = Field(default="medium")
    confidence: str | float = Field(default="medium")
    reasoning: str = Field(default="")


class HypothesisBatchJSON(BaseModel):
    hypotheses: list[HypothesisItemJSON] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


def _choose_profile(cfg: dict[str, Any]) -> str:
    # Prefer explicit strategist, then guidance, then agent as last resort
    try:
        models = cfg.get("models", {})
        if "strategist" in models:
            return "strategist"
        if "guidance" in models:
            return "guidance"
        return "agent"
    except Exception:
        return "agent"


class Strategist:
    """Senior planning agent."""

    def __init__(self, config: dict[str, Any] | None = None, debug: bool = False, session_id: str | None = None, debug_logger=None, mission: str | None = None):
        self.config = config or {}
        profile = _choose_profile(self.config)
        
        # Initialize or reuse debug logger
        self.debug_logger = debug_logger
        if debug and self.debug_logger is None:
            from analysis.debug_logger import DebugLogger
            self.debug_logger = DebugLogger(session_id or "strategist")
        
        self.profile = profile
        self.llm = UnifiedLLMClient(cfg=self.config, profile=profile, debug_logger=self.debug_logger)
        # Overarching mission to keep in strategist context when available
        self.mission = mission
        # Two-pass review toggle (off by default; enabled via config)
        try:
            self.two_pass_review = bool(self.config.get('strategist_two_pass_review', False))
        except Exception:
            self.two_pass_review = False

    def _context_limit(self) -> int:
        try:
            models = (self.config or {}).get('models', {})
            mcfg = models.get(self.profile, {})
            return int(mcfg.get('max_context') or (self.config or {}).get('context', {}).get('max_tokens', 256000))
        except Exception:
            return 256000

    def _log_usage(self, step: str, system: str, user: str):
        try:
            tracker = get_token_tracker()
            last = tracker.get_last_usage()
            # Compute prompt tokens approximately if provider didn't report
            input_tokens = (last or {}).get('input_tokens') or 0
            provider = (last or {}).get('provider') or self.llm.provider_name
            model = (last or {}).get('model') or self.llm.model
            if not input_tokens:
                try:
                    input_tokens = count_tokens(system + "\n\n" + user, provider, model)
                except Exception:
                    input_tokens = 0
            limit = self._context_limit()
            pct = min(100, int((input_tokens * 100) / max(1, limit)))
            msg = f"[{self.profile}] {step}: input={input_tokens} tok, limit={limit}, context={pct}% ({provider}:{model})"
            if self.debug_logger and hasattr(self.debug_logger, 'log_event'):
                try:
                    self.debug_logger.log_event('LLM Token Usage', msg)
                except Exception:
                    pass
            else:
                print(msg)
        except Exception:
            pass

    def plan_next(self, *, graphs_summary: str, completed: list[str], n: int = 5,
                  hypotheses_summary: str | None = None, coverage_summary: str | None = None,
                  ledger_summary: str | None = None, phase_hint: str | None = None) -> list[dict[str, Any]]:
        """Plan the next n investigations from comprehensive audit context.

        Returns a list of dicts compatible with downstream display and PlanStore.

        Prompt design notes (simple and commented for clarity):
        - Planning should start broad (aspect frames) then home in (suspicions) as evidence accumulates.
        - We do NOT encode a rigid slot ratio here – we instruct the model to maintain a sensible
          balance (e.g., ~60% aspects early, shifting to more suspicions later). This keeps logic simple
          and avoids brittle heuristics in code.
        - Each item must include a clear rationale; we ask the model to fold "why now" and
          "exit criteria" into the single 'reasoning' field, avoiding schema churn.
        """
        # Build mode-specific system prompt
        if phase_hint == 'Coverage':
            system = (
                "You are a senior security auditor in SWEEP MODE (Phase 1).\n"
                "Your goal: Systematically analyze each component for vulnerabilities.\n\n"
                "OPERATING CONSTRAINTS:\n"
                "- Static analysis only - no runtime execution\n"
                "- All actions must be CODE-ONLY: reading files, mapping flows, static reasoning\n\n"
                "SWEEP MODE STRATEGY:\n"
                "- Analyze each medium-sized logical unit (contract/module/class)\n"
                "- Wide sweep to visit every component and find bugs\n"
                "- Target medium-sized units: contracts, modules, classes, services\n"
                "- NOT individual functions or broad cross-cutting concerns\n"
                "- Output ASPECT items only — one per component\n"
                "- Maximum 1 item per component, spread across different modules\n"
                "- NEVER suggest investigating a component that was already completed\n"
                "- If no unanalyzed components remain, return an empty list\n\n"
                "INVESTIGATION GUIDELINES:\n"
                "- Focus on achieving broad coverage of unvisited components\n"
                "- Prioritize components handling critical state or permissions\n"
                "- Look for common vulnerability patterns\n"
                "- Avoid repeating completed investigations\n\n"
                "FOR EACH ITEM include WHY NOW and EXIT CRITERIA in 'reasoning'.\n"
                "Category should be 'aspect', expected_impact realistic.\n"
            )
        elif phase_hint == 'Saliency':
            system = (
                "You are a senior security auditor in INTUITION MODE (Phase 2).\n"
                "Your goal: Use intuition to find HIGH-IMPACT vulnerabilities.\n\n"
                "OPERATING CONSTRAINTS:\n"
                "- Static analysis only - no runtime execution\n"
                "- All actions must be CODE-ONLY: reading files, mapping flows, static reasoning\n\n"
                "INTUITION MODE STRATEGY:\n"
                "- Follow your instincts about what feels most vulnerable\n"
                "- Deep-dive into the most promising, impactful areas\n"
                "- PRIORITIZE MONETARY IMPACT above all else\n"
                "- Look for CONTRADICTIONS between assumptions and observations\n"
                "- Target suspicious cross-component interactions\n"
                "- Focus on invariant violations and high-confidence bugs\n"
                "- Output primarily SUSPICION items (specific vulnerabilities)\n"
                "- Flexible granularity - zoom into specific functions or patterns\n\n"
                "KEY INTUITION TARGETS:\n"
                "1. VALUE AT RISK: Where can money be stolen or locked?\n"
                "2. CONTRADICTIONS: What doesn't match between docs and code?\n"
                "3. AUTH BYPASSES: Where might permission checks fail?\n"
                "4. STATE CORRUPTION: What could break critical invariants?\n\n"
                "FOR EACH ITEM include WHY NOW and EXIT CRITERIA in 'reasoning'.\n"
                "Category should be 'suspicion' for bugs, 'aspect' for deep dives.\n"
            )
        else:
            # Auto mode - generic system prompt
            system = (
                "You are a senior security auditor planning an audit roadmap.\n"
                "You have access to all graphs, annotations, previous findings, and coverage data.\n\n"
                "OPERATING CONSTRAINTS:\n"
                "- Static analysis only - no runtime execution\n"
                "- All actions must be CODE-ONLY: reading files, mapping flows, static reasoning\n\n"
                "Adapt your strategy based on coverage:\n"
                "- If coverage < 90%: Use Sweep mode (systematic component analysis)\n"
                "- If coverage >= 90%: Use Intuition mode (deep, high-impact exploration)\n\n"
                "FOR EACH ITEM include WHY NOW and EXIT CRITERIA in 'reasoning'.\n"
            )

        completed_str = "\n".join(f"- {c}" for c in completed) if completed else "(none)"
        hypotheses_str = hypotheses_summary or "(no hypotheses formed yet)"
        coverage_str = coverage_summary or "(no coverage data)"

        # Coverage targets extraction removed - not needed without coverage top-up
        
        # Calculate planning iteration count (this is passed as part of completed list)
        planning_iteration = len(completed) // n + 1 if n > 0 else 1
        
        # Build mode-specific user prompt
        base_context = (
            f"ALL GRAPHS WITH ANNOTATIONS:\n{graphs_summary}\n\n"
            f"CURRENT HYPOTHESES (vulnerabilities found):\n{hypotheses_str}\n\n"
            f"COMPLETED INVESTIGATIONS (with results):\n{completed_str}\n\n"
            f"COVERAGE STATUS:\n{coverage_str}\n\n"
            f"AUDIT PROGRESS:\n"
            f"- Planning iteration: {planning_iteration}\n"
            f"- Investigations completed: {len(completed)}\n\n"
        )
        
        if phase_hint == 'Coverage':
            # Phase 1: Wide sweep for shallow bugs
            user = (
                base_context +
                f"CURRENT MODE: SWEEP (Phase 1 - Wide exploration)\n\n"
                f"Plan the top {n} NEW investigations for systematic coverage.\n\n"
                f"CRITICAL: DO NOT REPEAT ANY COMPLETED INVESTIGATIONS LISTED ABOVE!\n"
                f"If you cannot find {n} new components to analyze, return fewer items.\n"
                f"If all major components have been analyzed, return NO items.\n\n"
                f"REQUIREMENTS:\n"
                f"  • Pick medium-sized components: contracts, modules, classes (NOT individual functions)\n"
                f"  • Goal format: \"Vulnerability analysis of [Component]\"\n"
                f"  • Category: 'aspect' (always)\n"
                f"  • Assign priority 1-10 (10 = highest) for each investigation\n"
                f"  • One investigation per component, spread across different modules\n"
                f"  • Exclude: interfaces, tests, mocks, vendor libraries\n"
                f"  • Focus on achieving broad coverage of the codebase\n"
                f"  • NEVER repeat a component that appears in COMPLETED INVESTIGATIONS\n\n"
                f"PRIORITIZATION:\n"
                f"1. Unvisited major components\n"
                f"2. Components handling critical state or permissions\n"
                f"3. Components with complex logic\n"
                f"4. Components interacting with external systems\n\n"
                f"For each investigation, include WHY NOW and EXIT CRITERIA in 'reasoning'.\n"
                f"If no new components remain, respond with an empty list of investigations."
            )
        elif phase_hint == 'Saliency':
            # Phase 2: Intuition-guided deep exploration
            user = (
                base_context +
                f"CURRENT MODE: INTUITION (Phase 2 - Deep exploration)\n\n"
                f"Plan the top {n} NEW investigations using INTUITION to find high-impact bugs.\n\n"
                f"USE YOUR INTUITION TO PRIORITIZE:\n"
                f"  1. MONETARY FLOWS: Areas where funds can be stolen, locked, or misdirected\n"
                f"  2. CONTRADICTIONS: Salient mismatches between assumptions and observations\n"
                f"  3. HIGH-IMPACT AUTH: Critical authentication/authorization vulnerabilities\n"
                f"  4. STATE CORRUPTION: Manipulation affecting critical invariants\n\n"
                f"TARGET THESE AREAS:\n"
                f"  • Value transfer mechanisms and payment flows\n"
                f"  • Clear contradictions in the annotated graphs\n"
                f"  • Suspicious permission check patterns\n"
                f"  • Complex cross-component value interactions\n"
                f"  • Areas that \"feel\" vulnerable based on patterns\n\n"
                f"GOAL EXAMPLES:\n"
                f"  - \"Investigate potential theft in [payment flow]\"\n"
                f"  - \"Analyze contradiction: [assumption X] vs [observation Y]\"\n"
                f"  - \"Deep dive: authorization bypass in [critical function]\"\n\n"
                f"Category: Primarily 'suspicion' for specific high-impact bugs\n"
                f"Priority: Assign 1-10 (10 = highest) based on anticipated impact\n"
                f"Approach: Follow your intuition about what feels most vulnerable\n\n"
                f"For each investigation, include WHY NOW and EXIT CRITERIA in 'reasoning'."
            )
        else:
            # Auto mode - include both descriptions
            user = (
                base_context +
                f"PHASE: {phase_hint or 'auto'}\n\n"
                f"Plan the top {n} NEW investigations.\n\n"
                f"Determine phase based on coverage percentage and adapt strategy accordingly.\n"
                f"If coverage < 90%, use Coverage mode. Otherwise, use Intuition mode.\n\n"
                f"For each investigation, include WHY NOW and EXIT CRITERIA in 'reasoning'."
            )

        # Allow fine-grained reasoning control for planning step
        plan_effort = None
        try:
            mdl_cfg = (self.config or {}).get('models', {}).get(self.profile, {})
            plan_effort = mdl_cfg.get('plan_reasoning_effort')
        except Exception:
            plan_effort = None
        plan: PlanBatch = self.llm.parse(system=system, user=user, schema=PlanBatch, reasoning_effort=plan_effort)
        # Log usage and context after call
        self._log_usage('plan_next', system, user)
        items = []
        for it in plan.investigations[:n]:
            items.append({
                "goal": it.goal,
                "focus_areas": it.focus_areas,
                "priority": it.priority,
                "reasoning": it.reasoning,
                "category": it.category,
                "expected_impact": it.expected_impact,
            })
        # DISABLED: Coverage top-up removed because we don't have card count information
        # To properly implement this, we would need:
        # 1. Number of code cards per node
        # 2. Number of visited cards per node
        # Then we could prioritize nodes with many unvisited cards
        pass
        return items

    def revise_after(self, last_result: dict[str, Any]) -> None:
        return None

    def deep_think(self, *, context: str, phase: str = None) -> list[dict[str, Any]]:
        """Perform senior deep analysis on the prepared context and emit hypothesis items.

        Args:
            context: The prepared investigation context
            phase: 'Coverage' or 'Saliency' (if not provided, defaults to Saliency)

        Returns a list of dicts with keys:
          description, details, vulnerability_type, severity, confidence, node_ids, reasoning
        """
        # Extract valid node IDs from context for validation
        # Robustly parse IDs from the formatted context produced by agent_core._format_graph_for_display()
        # Sources:
        #  - NODES sections: lines like "  <id>|<type>|<label>..."
        #  - LOADED NODES section: comma-separated IDs on indented lines
        import re
        valid_node_ids: set[str] = set()
        try:
            lines = context.splitlines()
            in_nodes = False
            in_loaded_nodes = False
            for ln in lines:
                s = ln.strip()
                # Section toggles
                if s.startswith('NODES ('):
                    in_nodes = True
                    in_loaded_nodes = False
                    continue
                if s.startswith('EDGES (') or s.startswith('--- Graph:') or s.startswith('===') or s == '':
                    in_nodes = False
                if s.startswith('=== LOADED NODES'):
                    in_loaded_nodes = True
                    in_nodes = False
                    continue
                if in_loaded_nodes and (s.startswith('===') or s.startswith('---') or s.startswith('Total:') or s.startswith('ACTIONS') or s.startswith('SYSTEM ARCHITECTURE') or s == ''):
                    in_loaded_nodes = False

                # Collect IDs
                if in_nodes and ln.startswith('  '):
                    # Node lines are indented, with id before the first '|'
                    try:
                        nid = ln.strip().split('|', 1)[0]
                        if nid:
                            valid_node_ids.add(nid)
                    except Exception:
                        pass
                elif in_loaded_nodes and ln.startswith('  '):
                    # Comma-separated ids
                    try:
                        for nid in ln.strip().split(','):
                            nid = nid.strip()
                            if nid:
                                valid_node_ids.add(nid)
                    except Exception:
                        pass

            # Backward compatibility: also pick up any bracketed IDs if present in future
            bracket_ids = re.findall(r"\[([a-zA-Z0-9_\.\-:]+)\]", context)
            for bid in bracket_ids:
                valid_node_ids.add(bid)
        except Exception:
            valid_node_ids = set()
        
        # Use phase parameter if provided, otherwise default to Phase 2 (Saliency)
        is_phase1 = (phase == 'Coverage')
        if is_phase1:
            system = (
                "You are a security auditor analyzing code components for vulnerabilities.\n"
                "Your task: Identify security vulnerabilities in the provided code.\n\n"
                "INSTRUCTIONS:\n"
                "- Look for ALL types of vulnerabilities in this component\n"
                "- Consider issues like: missing validation, access control, overflow, reentrancy, logic errors, etc.\n"
                "- Provide thorough analysis of the component\n"
                "- Focus on real, exploitable vulnerabilities\n"
                "- If there are no vulnerabilities found, say 'NO_HYPOTHESES: true'\n\n"
                "DEDUPLICATION:\n"
                "- The context lists EXISTING HYPOTHESES. Do NOT propose duplicates.\n"
                "- Skip any issues that have already been found.\n"
            )
        else:
            # Phase 2 (Saliency) - use the original complex prompt
            system = (
                "You are a deep-thinking senior security auditor.\n"
                "Your job is to: (1) think deeply about the active investigation aspect,\n"
                "(2) uncover real, non-trivial vulnerabilities as clear hypotheses, and (3) advise the Scout on next steps.\n"
                "Additionally, if the prepared context reveals other vulnerabilities not strictly tied to the investigation goal, include them as well.\n\n"
                "OPERATING CONSTRAINTS (IMPORTANT):\n"
                "- Hound performs static analysis only - it cannot execute code or interact with live systems.\n"
                "- Do NOT recommend or assume runtime execution or live system probing.\n"
                "- All GUIDANCE must be CODE-ONLY: which files/functions/classes/methods to inspect and what to verify statically.\n"
                "- You MAY include a theoretical exploit plan clearly labeled as \"theoretical/manual reproduction outside Hound\".\n\n"
                "CRITICAL: Base your analysis on the investigation goal and the exploration/history shown in the context,\n"
                "but do NOT limit yourself to only that goal — surface ANY vulnerabilities you can justify from the provided context.\n"
                "ANTI–FALSE-POSITIVE GUARDRAILS:\n"
                "- Propose a hypothesis only if the ROOT CAUSE is explicitly evidenced in the provided code.\n"
                "- Cite specific files/functions in Affected Code; include exact node IDs from the graphs.\n"
                "- Verify that required preconditions are plausible given the code; check for guards/require/reentrancy/permissions that would mitigate the issue.\n"
                "- If evidence is weak or ambiguous, lower confidence to low or omit the hypothesis entirely.\n"
                "- Prefer fewer, higher-quality hypotheses over speculative ones.\n"
                "DEDUPLICATION:\n"
                "- The context lists EXISTING HYPOTHESES. Do NOT propose duplicates.\n"
                "- Treat items as duplicates if they share the same root cause and substantially the same affected code path(s)/function(s), even if phrased differently.\n"
                "- If you would repeat an existing hypothesis, skip it and focus on novel issues.\n"
                "If you are highly confident there are no vulnerabilities in scope, say so.\n"
            )
        # Prepend global mission if provided
        mission_block = ""
        if isinstance(self.mission, str) and self.mission.strip():
            mission_block = f"GLOBAL MISSION: {self.mission.strip()}\n\n"

        # Simpler user prompt for Phase 1
        if is_phase1:
            # Extract existing hypotheses from context for prominent display
            existing_hyps_section = ""
            if "=== EXISTING HYPOTHESES" in context:
                try:
                    start = context.index("=== EXISTING HYPOTHESES")
                    end = context.find("\n===", start + 1)
                    if end == -1:
                        end = context.find("\n\n", start)
                    if end != -1:
                        existing_hyps_section = context[start:end].strip()
                except Exception:
                    pass
            
            user = (
                mission_block +
                "CONTEXT (code being analyzed):\n" + context + "\n\n"
                + ("⚠️ CRITICAL: DO NOT DUPLICATE THESE EXISTING HYPOTHESES ⚠️\n" + existing_hyps_section + "\n\n" if existing_hyps_section else "") +
                "Identify and describe all security vulnerabilities in this code.\n\n"
                "OUTPUT FORMAT:\n"
                "List each bug on a separate line using this format:\n"
                "Title | Type | Root Cause | Attack Vector | Affected Node IDs | Affected Code | Severity | Confidence | Reasoning\n\n"
                "- Security vulnerabilities likely exist in this code.\n"
                "- Provide up to 5 bugs maximum\n"
                "- If you cannot identify any issues, respond with: NO_HYPOTHESES: true\n"
                "- Also provide brief GUIDANCE on what to check next (2-3 suggestions)\n"
                "\nOUTPUT STRICTNESS:\n"
                "- When producing HYPOTHESES, ensure each hypothesis is kept on ONE line (no wrapping). Use commas/semicolons to keep it compact.\n"
            )
        else:
            # Original complex prompt for Phase 2
            # Extract existing hypotheses from context for prominent display
            existing_hyps_section = ""
            if "=== EXISTING HYPOTHESES" in context:
                try:
                    start = context.index("=== EXISTING HYPOTHESES")
                    end = context.find("\n===", start + 1)
                    if end == -1:
                        end = context.find("\n\n", start)
                    if end != -1:
                        existing_hyps_section = context[start:end].strip()
                except Exception:
                    pass
            
            user = (
                mission_block +
                "CONTEXT (includes === INVESTIGATION GOAL === and compressed history):\n" + context + "\n\n"
                "⚠️ CRITICAL DEDUPLICATION REQUIREMENT ⚠️\n"
                + (f"{existing_hyps_section}\n\n" if existing_hyps_section else "") +
                "DO NOT propose any hypothesis that duplicates the above existing hypotheses!\n"
                "Check EACH hypothesis you're about to propose against the existing list.\n\n"
                "OUTPUT INSTRUCTIONS:\n"
                "1) HYPOTHESES (max 5, EXACTLY ONE PER LINE, exactly this pipe-separated format; DO NOT wrap across lines; use commas/semicolons instead of line breaks):\n"
                "   Title | Type | Root Cause | Attack Vector | Affected Node IDs | Affected Code | Severity | Confidence | Reasoning\n"
                "   - severity: critical|high|medium|low; confidence: high|medium|low\n"
                "   - Keep Title concise and actionable.\n"
                "   - CRITICAL: Affected Node IDs must be EXACT node IDs as shown in the NODES sections (token before the first pipe) or listed under LOADED NODES (e.g., func_transfer, contract_Token, state_balances)\n"
                "   - Use a comma-separated list of actual node IDs. Do NOT invent new IDs or use descriptions.\n"
                "   - You MUST provide at least one valid node ID for each hypothesis\n"
                "   - Affected Code should reference concrete functions/files if possible.\n"
                "   - Hard limit: keep each hypothesis under 240 characters to avoid line wrapping.\n\n"
                "2) GUIDANCE (next steps for the Scout):\n"
                "   - Provide 2–5 concrete CODE-ONLY actions to gather evidence or rule in/out the hypotheses (load/read specific files, trace auth/dataflow, check invariants).\n"
                "   - Reference specific nodes/functions/files the Scout should load or analyze next.\n"
                "   - Do NOT suggest runtime execution or live system interaction.\n\n"
                "3) If NO credible hypothesis is found, include a line: NO_HYPOTHESES: true (still provide GUIDANCE).\n"
                "4) You MAY include additional hypotheses that are unrelated to the exact goal if the current context clearly supports them (avoid false positives).\n"
            )

        # Save deep_think prompts to debug files if debug logger is available
        if self.debug_logger:
            try:
                from datetime import datetime
                from pathlib import Path
                # Prefer debug logger's output_dir for consistency; fallback to CWD/.hound_debug
                base_dir = getattr(self.debug_logger, 'output_dir', None)
                if not base_dir:
                    base_dir = Path.cwd() / '.hound_debug'
                base_path = Path(base_dir)
                # Create deep_think directory scoped by session id
                session_dir = base_path / str(getattr(self.debug_logger, 'session_id', 'strategist')) / 'deep_think_prompts'
                session_dir.mkdir(parents=True, exist_ok=True)
                # Timestamped filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                prompt_path = session_dir / f'deep_think_{timestamp}.txt'
                # Write full prompt for reproduction
                with open(prompt_path, 'w') as f:
                    f.write("=" * 80 + "\n")
                    f.write("DEEP THINK PROMPT\n")
                    f.write(f"Generated at: {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("SYSTEM PROMPT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(system)
                    f.write("\n\n")
                    f.write("USER PROMPT:\n")
                    f.write("-" * 40 + "\n")
                    f.write(user)
                    f.write("\n\n")
                    f.write("=" * 80 + "\n")
                    f.write("NOTE: Combine the system and user prompts when testing manually.\n")
                    f.write("=" * 80 + "\n")
                # Optionally log an event into the HTML debug log
                try:
                    if hasattr(self.debug_logger, 'log_event'):
                        self.debug_logger.log_event('DeepThink Prompt Saved', str(prompt_path))
                except Exception:
                    pass
            except Exception:
                # Never fail deep_think on debug save issues
                pass

        # First try a strict JSON response for robust parsing
        items: list[dict[str, Any]] | None = None
        json_used = False
        fallback_node_ids_assigned: list[str] = []
        try:
            hyp_effort = None
            try:
                mdl_cfg = (self.config or {}).get('models', {}).get(self.profile, {})
                hyp_effort = mdl_cfg.get('hypothesize_reasoning_effort')
            except Exception:
                hyp_effort = None
            json_user = (
                user
                + "\n\nSTRICT OUTPUT: Return ONLY JSON with keys 'hypotheses' (list) and 'guidance' (list of strings).\n"
                + "Each hypothesis must include: title,type,root_cause,attack_vector,node_ids (array),affected_code (array),severity,confidence (high|medium|low or 0.0-1.0),reasoning.\n"
                + "Example: {\"hypotheses\":[{\"title\":\"...\",\"type\":\"...\",\"root_cause\":\"...\",\"attack_vector\":\"...\",\"node_ids\":[\"func_x\"],\"affected_code\":[\"path:func\"],\"severity\":\"high\",\"confidence\":\"medium\",\"reasoning\":\"...\"}],\"guidance\":[\"...\"]}"
            )
            batch = self.llm.parse(system=system, user=json_user, schema=HypothesisBatchJSON, reasoning_effort=hyp_effort)
            self._log_usage('deep_think(json)', system, json_user)
            # Keep raw JSON for display
            try:
                import json as _json
                self.last_raw = _json.dumps(batch.model_dump(), ensure_ascii=False, indent=2)
            except Exception:
                self.last_raw = str(batch)
            # Convert to internal items
            def _conf_to_float(val: str | float) -> float:
                try:
                    if isinstance(val, int | float):
                        v = float(val)
                        return max(0.0, min(1.0, v))
                except Exception:
                    pass
                s = str(val).lower()
                if 'high' in s:
                    return 0.9
                if 'low' in s:
                    return 0.4
                return 0.6
            items = []
            for h in batch.hypotheses:
                conf = _conf_to_float(h.confidence)
                node_ids = [n for n in (h.node_ids or []) if isinstance(n, str) and n.strip()]
                # If no valid node IDs, generate fallback
                if not node_ids:
                    try:
                        import hashlib as _hash
                        sig = (h.title or '') + '|' + (h.type or '')
                        nid = 'fallback_' + _hash.md5(sig.encode()).hexdigest()[:10]
                    except Exception:
                        nid = 'fallback_unknown'
                    node_ids = [nid]
                    try:
                        fallback_node_ids_assigned.append(h.title[:140] if h.title else 'unknown')
                    except Exception:
                        pass
                # Build details similarly to pipe format
                details = (
                    f"VULNERABILITY TYPE: {h.type}\n"
                    f"ROOT CAUSE: {h.root_cause}\n"
                    f"ATTACK VECTOR: {h.attack_vector}\n"
                    f"AFFECTED NODES: {', '.join(node_ids)}\n"
                    f"AFFECTED CODE: {', '.join(h.affected_code or [])}\n"
                    f"SEVERITY: {h.severity}\n"
                    f"REASONING: {h.reasoning}"
                )
                items.append({
                    'description': h.title,
                    'details': details,
                    'vulnerability_type': h.type,
                    'severity': str(h.severity).lower() if h.severity else 'medium',
                    'confidence': conf,
                    'node_ids': node_ids,
                    'reasoning': h.reasoning or '',
                })
            json_used = True
        except Exception:
            items = None

        if items is None:
            # Fallback to raw text and legacy parsing
            try:
                hyp_effort = None
                try:
                    mdl_cfg = (self.config or {}).get('models', {}).get(self.profile, {})
                    hyp_effort = mdl_cfg.get('hypothesize_reasoning_effort')
                except Exception:
                    hyp_effort = None
                resp = self.llm.raw(system=system, user=user, reasoning_effort=hyp_effort)
                self._log_usage('deep_think', system, user)
                try:
                    self.last_raw = resp
                except Exception:
                    pass
            except Exception:
                return []

        if items is None:
            raw_text = str(resp)
        else:
            raw_text = str(self.last_raw or '')
        # Group multi-line hypotheses: accumulate lines until we have at least 8 pipes (9 fields)
        grouped: list[str] = []
        invalid_groups: list[str] = []
        cur: list[str] = []
        cur_pipes = 0
        for raw_ln in raw_text.splitlines():
            ln = raw_ln.strip()
            if not ln:
                # Blank line: flush current candidate
                if cur:
                    joined = ' '.join(cur)
                    if cur_pipes >= 8:
                        grouped.append(joined)
                    else:
                        invalid_groups.append(joined)
                cur, cur_pipes = [], 0
                continue
            if '|' not in ln:
                # Non-pipe line likely guidance or prose; attach to current if building
                if cur:
                    cur.append(ln)
                continue
            # Pipe line: add and update count
            cur.append(ln)
            cur_pipes = ' '.join(cur).count('|')
            if cur_pipes >= 8:
                grouped.append(' '.join(cur))
                cur, cur_pipes = [], 0
        # Flush tail (complete or incomplete)
        if cur:
            joined = ' '.join(cur)
            if cur_pipes >= 8:
                grouped.append(joined)
            else:
                invalid_groups.append(joined)

        lines = [ln for ln in grouped if ln and '|' in ln]
        # Prepare containers (some may be reused when JSON path was used)
        items = items or []
        skipped_no_node_ids: list[str] = []
        skipped_invalid_format: list[str] = []

        # Salvage invalid groups by padding missing fields and extracting hints
        salvaged: list[str] = []
        for raw in invalid_groups:
            if '|' not in raw:
                continue
            parts = [p.strip() for p in raw.split('|')]
            # Pad to 9 fields
            while len(parts) < 9:
                parts.append('')
            # Heuristic: if affected_nodes empty, try to extract node-like tokens
            if not parts[4]:
                try:
                    import re as _re
                    toks = _re.findall(r"\b(?:func|contract|state|module|service|ext_service)_[A-Za-z0-9_\.:-]+", raw)
                    if toks:
                        parts[4] = ','.join(list(dict.fromkeys(toks))[:4])
                except Exception:
                    pass
            # Default severity/confidence if missing
            if not parts[6]:
                parts[6] = 'medium'
            if not parts[7]:
                parts[7] = 'medium'
            salvaged.append(' | '.join(parts[:9]))

        # Merge salvaged into lines (avoid dupes)
        for s in salvaged:
            if s not in lines:
                lines.append(s)
        for ln in lines:
            parts = [p.strip() for p in ln.split('|')]
            title = parts[0] if len(parts) > 0 else "Hypothesis"
            vuln_type = parts[1].lower() if len(parts) > 1 else "security_issue"
            root_cause = parts[2] if len(parts) > 2 else ""
            attack_vector = parts[3] if len(parts) > 3 else ""
            affected_nodes = parts[4] if len(parts) > 4 else ""
            affected_code = parts[5] if len(parts) > 5 else ""
            severity = parts[6].lower() if len(parts) > 6 else "medium"
            conf_word = parts[7].lower() if len(parts) > 7 else "medium"
            reasoning = parts[8] if len(parts) > 8 else ""

            confidence = 0.6
            if 'high' in conf_word:
                confidence = 0.9
            elif 'low' in conf_word:
                confidence = 0.4

            # Parse and validate node IDs
            raw_node_ids = [n.strip() for n in affected_nodes.split(',') if n.strip()]
            node_ids = []
            
            for nid in raw_node_ids:
                # Check if this looks like a valid node ID (no spaces, reasonable length)
                if nid in valid_node_ids:
                    # It's a valid node ID from the context
                    node_ids.append(nid)
                elif len(nid) < 50 and ' ' not in nid and nid != '':
                    # Might be a node ID not in context but looks valid
                    node_ids.append(nid)
                # Otherwise skip it (it's likely a description)
            
            # If no valid node IDs, try to infer; else fallback to 'system'
            if not node_ids:
                # Try extract node-like tokens from entire line
                try:
                    import re as _re
                    toks = _re.findall(r"\b(?:func|contract|state|module|service|ext_service)_[A-Za-z0-9_\.:-]+", ln)
                    if toks:
                        node_ids = list(dict.fromkeys(toks))[:3]
                except Exception:
                    pass
                if not node_ids:
                    # Fallback to a unique placeholder node id to avoid store-level cross-collision
                    try:
                        import hashlib as _hash
                        sig = (title or '') + '|' + (vuln_type or '')
                        nid = 'fallback_' + _hash.md5(sig.encode()).hexdigest()[:10]
                    except Exception:
                        nid = 'fallback_unknown'
                    node_ids = [nid]
                    try:
                        fallback_node_ids_assigned.append(title[:140])
                    except Exception:
                        pass
            
            details = (
                f"VULNERABILITY TYPE: {vuln_type}\n"
                f"ROOT CAUSE: {root_cause}\n"
                f"ATTACK VECTOR: {attack_vector}\n"
                f"AFFECTED NODES: {', '.join(node_ids)}\n"
                f"AFFECTED CODE: {affected_code}\n"
                f"SEVERITY: {severity}\n"
                f"REASONING: {reasoning}"
            )

            items.append({
                'description': title,
                'details': details,
                'vulnerability_type': vuln_type,
                'severity': severity,
                'confidence': confidence,
                'node_ids': node_ids,
                'reasoning': reasoning,
            })
        # Record invalid-format items (not enough fields)
        try:
            for raw in invalid_groups[:5]:
                text = raw.replace('\n',' ').strip()
                if '|' in text:
                    skipped_invalid_format.append(text[:160])
        except Exception:
            pass
        # Second-pass self-critique to reduce false positives (optional)
        if not getattr(self, 'two_pass_review', False):
            # Return first-pass items directly when two-pass review is disabled
            # Expose skip info for the caller (agent) to print nicely
            try:
                self.last_skipped = {
                    'no_node_ids': skipped_no_node_ids,
                    'invalid_format': skipped_invalid_format,
                    'fallback_node_ids_assigned': fallback_node_ids_assigned,
                    'parsed_lines': len(items) if json_used else len(lines),
                    'raw_text_len': len(raw_text or ''),
                    'used_json': json_used,
                }
            except Exception:
                pass
            return items

        # Two-pass enabled: review the candidates
        class _ReviewItem(BaseModel):
            description: str
            vulnerability_type: str
            severity: str
            confidence: float
            node_ids: list[str]
            reasoning: str
            accept: bool = Field(description="Accept only if evidence in context clearly supports root cause and no strong mitigation exists.")
            reason: str = Field(description="Why accepted/rejected; cite mitigating checks if rejecting.")

        class _ReviewBatch(BaseModel):
            items: list[_ReviewItem]

        review_instr = (
            "You previously proposed candidate hypotheses. Now act as a skeptical reviewer.\n"
            "Reject any item lacking explicit evidence of the ROOT CAUSE in the provided context, or where guards/permissions clearly mitigate it.\n"
            "Prefer fewer, higher-quality items. Keep at most 3 accepted items. Return JSON.\n"
        )

        cand_lines = []
        for i, it in enumerate(items[:5], 1):
            cand_lines.append(
                f"{i}. {it['description']} | type={it['vulnerability_type']} | sev={it['severity']} | conf={it['confidence']} | nodes={','.join(it.get('node_ids') or [])}"
            )
        review_user = (
            "CONTEXT (same as above):\n" + context + "\n\n"
            "CANDIDATES:\n" + "\n".join(cand_lines) + "\n\n"
            "Respond with JSON: {\"items\":[{...}]}, fields: description,vulnerability_type,severity,confidence,node_ids,reasoning,accept,reason."
        )

        try:
            reviewed = self.llm.parse(system=review_instr, user=review_user, schema=_ReviewBatch)
            self._log_usage('deep_think_review', review_instr, review_user)
            accepted = [it.model_dump() for it in reviewed.items if it.accept]
            # Sort by severity and confidence, cap at 3
            def _sev_rank(s):
                return {"critical":3,"high":2,"medium":1,"low":0}.get(str(s).lower(),1)
            accepted.sort(key=lambda x: (_sev_rank(x.get('severity','medium')), x.get('confidence',0.0)), reverse=True)
            try:
                self.last_skipped = {
                    'no_node_ids': skipped_no_node_ids,
                    'invalid_format': skipped_invalid_format,
                    'fallback_node_ids_assigned': fallback_node_ids_assigned,
                    'parsed_lines': len(items) if json_used else len(lines),
                    'raw_text_len': len(raw_text or ''),
                    'used_json': json_used,
                }
            except Exception:
                pass
            return accepted[:3]
        except Exception:
            # Fallback: basic filter by severity/confidence and cap
            def _sev_rank(s):
                return {"critical":3,"high":2,"medium":1,"low":0}.get(str(s).lower(),1)
            items.sort(key=lambda x: (_sev_rank(x.get('severity','medium')), x.get('confidence',0.0)), reverse=True)
            try:
                self.last_skipped = {
                    'no_node_ids': skipped_no_node_ids,
                    'invalid_format': skipped_invalid_format,
                    'fallback_node_ids_assigned': fallback_node_ids_assigned,
                    'parsed_lines': len(items) if json_used else len(lines),
                    'raw_text_len': len(raw_text or ''),
                    'used_json': json_used,
                }
            except Exception:
                pass
            return items[:3]

__all__ = ["Strategist", "PlanItemSchema", "PlanBatch"]
