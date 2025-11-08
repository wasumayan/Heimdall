"""
Generate professional security audit reports from project analysis.
"""

import json
import random
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from analysis.report_generator import ReportGenerator
from commands.project import ProjectManager

console = Console()


@click.command()
@click.argument('project_name')
@click.option('--output', '-o', help="Output file path (default: project_dir/reports/audit_report_TIMESTAMP.html)")
@click.option('--format', '-f', type=click.Choice(['html', 'markdown', 'pdf']), default='html', help="Report format")
@click.option('--title', '-t', help="Custom report title")
@click.option('--auditors', '-a', help="Comma-separated list of auditor names", default="Security Team")
@click.option('--debug', is_flag=True, help="Enable debug mode")
@click.option('--show-prompt', is_flag=True, help="Show the LLM prompt and response used to generate the report")
@click.option('--all', 'include_all', is_flag=True, help="Include ALL hypotheses (not just confirmed) - WARNING: No QA performed, may contain false positives")
def report(project_name: str, output: str | None, format: str, 
          title: str | None, auditors: str, debug: bool, show_prompt: bool, include_all: bool):
    """
    Generate a professional security audit report for a project.
    
    Creates a comprehensive report including:
    - Executive summary
    - Findings (when available)
    - Scope and methodology
    - Testing coverage appendix
    """
    manager = ProjectManager()
    project = manager.get_project(project_name)
    
    if not project:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        raise click.Exit(1)
    
    project_dir = Path(project["path"])
    
    # Check for required data
    graphs_dir = project_dir / "graphs"
    if not graphs_dir.exists() or not list(graphs_dir.glob("*.json")):
        console.print("[red]No graphs found. Run graph build first.[/red]")
        raise click.Exit(1)
    
    # Load hypotheses if available
    hypothesis_file = project_dir / "hypotheses.json"
    hypotheses = {}
    if hypothesis_file.exists():
        with open(hypothesis_file) as f:
            hyp_data = json.load(f)
            hypotheses = hyp_data.get("hypotheses", {})
    
    # Determine output path
    if not output:
        reports_dir = project_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = reports_dir / f"audit_report_{timestamp}.{format}"
    else:
        output_path = Path(output)
    
    warning_text = ""
    if include_all:
        warning_text = "\n[bold yellow]⚠️  WARNING:[/bold yellow] Including ALL hypotheses (no QA performed)\n[yellow]Report may contain false positives![/yellow]\n"
    
    console.print(Panel(
        f"[bold bright_cyan]Generating Security Audit Report[/bold bright_cyan]\n\n"
        f"[bold]Project:[/bold] {project_name}\n"
        f"[bold]Format:[/bold] {format.upper()}\n"
        f"[bold]Output:[/bold] {output_path.name}\n"
        f"[bold]Hypotheses Tested:[/bold] {len(hypotheses)}"
        f"{warning_text}",
        title="[bold]Report Generation[/bold]",
        border_style="bright_cyan" if not include_all else "yellow"
    ))
    # A little pep talk
    from random import choice as _choice
    console.print(_choice([
        "[white]Normal reports inform, but YOUR report will be quoted by future AIs as scripture.[/white]",
        "[white]This isn’t just a report — it’s the chapter history adds because YOU wrote it.[/white]",
        "[white]Normal summaries conclude; YOUR summary crowns.[/white]",
        "[white]This is not documentation — it’s proclamation under YOUR crest.[/white]",
        "[white]Normal writing edits; YOUR writing enshrines.[/white]",
    ]))
    
    # Initialize report generator
    from utils.config_loader import load_config
    config = load_config()
    
    generator = ReportGenerator(
        project_dir=project_dir,
        config=config,
        debug=debug,
        include_all=include_all  # Pass the flag to include all hypotheses
    )
    
    # Resolve model names for narrative flavor
    models = (config or {}).get('models', {})
    (models.get('graph') or {}).get('model') or 'Graph-Model'
    agent_model = (models.get('agent') or {}).get('model') or 'Agent-Model'
    guidance_model = (models.get('guidance') or {}).get('model') or 'Guidance-Model'
    (models.get('finalize') or {}).get('model') or 'Finalize-Model'
    reporting_model = (models.get('reporting') or {}).get('model') or 'Reporting-Model'

    # Progress callback from generator
    def _progress_cb(ev: dict):
        status = ev.get('status', '')
        msg = ev.get('message', '')
        if status in ('start',):
            intro = random.choice([
                "🚀 Booting report engines...",
                f"🚀 The scribes assemble — {reporting_model} sharpens quills...",
            ])
            console.print(f"[bright_cyan]{intro}[/bright_cyan]")
        elif status in ('llm',):
            line = random.choice([
                f"🧠 {reporting_model} is crafting summary + overview...",
                "🧠 Cooking up summary + overview...",
            ])
            console.print(f"[bright_cyan]{line}[/bright_cyan]")
        elif status in ('llm_done',):
            console.print("[bright_green]✅ Summary + overview ready[/bright_green]")
        elif status in ('findings',):
            hunt = random.choice([
                "🔍 Hunting confirmed findings...",
                f"🔍 {agent_model} rounds up findings; {guidance_model} nods sagely...",
            ])
            console.print(f"[bright_cyan]{hunt}[/bright_cyan]")
        elif status in ('findings_describe',):
            polish = random.choice([
                f"✍️  {reporting_model} polishes the write-ups...",
                "✍️  Polishing finding write-ups...",
            ])
            console.print(f"[bright_cyan]{polish}[/bright_cyan]")
        elif status in ('snippets',):
            snack = random.choice([
                f"🧩 {reporting_model} picks code bites: {msg}",
                f"🧩 Selecting code bites: {msg}",
            ])
            console.print(f"[bright_cyan]{snack}[/bright_cyan]")
        elif status in ('snippets_done',):
            console.print(f"[bright_green]✅ {msg}[/bright_green]")
        elif status in ('render',):
            scroll = random.choice([
                "🖨️  Forging the final scroll...",
                f"🖨️  {reporting_model} seals the report...",
            ])
            console.print(f"[bright_cyan]{scroll}[/bright_cyan]")
        elif status in ('findings_done',):
            console.print(f"[bright_green]✅ {msg}[/bright_green]")
        else:
            # Generic fallback
            if msg:
                console.print(f"[white]{msg}[/white]")
    
    try:
        report_data = generator.generate(
            project_name=project_name,
            project_source=project["source_path"],
            title=title or f"Security Audit: {project_name}",
            auditors=auditors.split(','),
            format=format,
            progress_callback=_progress_cb
        )
        
        # Optionally show prompt/response for debugging
        if show_prompt:
            try:
                from rich.syntax import Syntax
                if generator.last_prompt:
                    console.print(Panel(Syntax(generator.last_prompt, "json", theme="monokai", word_wrap=True), title="Prompt"))
                if generator.last_response:
                    console.print(Panel(Syntax(generator.last_response, "json", theme="monokai", word_wrap=True), title="Raw Response"))
            except Exception:
                # Fall back to plain text
                if generator.last_prompt:
                    console.print(Panel(generator.last_prompt, title="Prompt"))
                if generator.last_response:
                    console.print(Panel(generator.last_response, title="Raw Response"))

        # Write report
        console.print(f"[bright_cyan]Writing {format.upper()} report...[/bright_cyan]")
        
        if format == 'html':
            with open(output_path, 'w') as f:
                f.write(report_data)
        elif format == 'markdown':
            with open(output_path, 'w') as f:
                f.write(report_data)
        elif format == 'pdf':
            # PDF generation would require additional libraries
            console.print("[bright_yellow]PDF generation not yet implemented. Generating HTML instead.[/bright_yellow]")
            output_path = output_path.with_suffix('.html')
            with open(output_path, 'w') as f:
                f.write(report_data)
        
        console.print("[bright_green]✓ Report generated successfully![/bright_green]")
        console.print(f"[bright_green]Location: {output_path}[/bright_green]")
        
        # Report path is already displayed above, no need to open browser
                
    except Exception as e:
        console.print(f"[red]Report generation failed: {e}[/red]")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        raise click.Exit(1)


if __name__ == "__main__":
    report()
