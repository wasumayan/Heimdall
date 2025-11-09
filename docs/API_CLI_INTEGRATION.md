# API/CLI Integration Guide

This document describes the current integration status of Hound and BRAMA agents.

## Current Status: Fully Integrated ✅

✅ **Infrastructure Complete**: Both Hound and BRAMA are integrated via subprocess calls  
✅ **Path Resolution**: Backend automatically finds Hound in either `agents/hound` or root `hound/`  
✅ **Error Handling**: Both integrations have proper error handling and fallbacks  
✅ **Mock Data**: System falls back to mock data if agents are not available  
✅ **Red-Team Features**: BRAMA includes comprehensive red-teaming capabilities:
   - Security headers analysis
   - SSL/TLS certificate checks
   - Endpoint discovery
   - Technology stack fingerprinting
   - CORS misconfiguration detection
   - HTTP methods testing
   - Information disclosure checks  

## Integration Details

### 1. Hound Integration ✅

#### Current Implementation (Working)
- ✅ Creates Hound project via CLI: `hound project create <name> <path>`
- ✅ Builds knowledge graphs: `hound graph build <project_name>`
- ✅ Runs agent analysis: `hound agent audit <project_name>`
- ✅ Finalizes hypotheses: `hound finalize <project_name>`
- ✅ **Whitelist Builder**: Auto-generates file whitelist if not provided (within LOC budget)
- ✅ Reads findings from: `~/.hound/projects/{project_name}/hypotheses.json`
- ✅ Transforms output to Heimdall format (preserves all Hound fields)
- ✅ Uses xAI (Grok) for AI analysis
- ✅ **Telemetry Support**: Proxies SSE streams from Hound's telemetry server

#### Implementation Notes

**A. Hound Script Path**
- **Location**: `backend/main.py`
- **Implementation**: Uses `HOUND_PATH / "scripts" / "hound"` or checks for pip-installed `hound` command
- **Status**: ✅ Working - automatically resolves path

**B. Hound Configuration**
- **Location**: Hound uses `config.yaml` with LLM API keys
- **Implementation**: Uses xAI provider configured in Hound's config.yaml
- **Status**: ✅ Working - uses XAI_API_KEY from environment

**C. Hound Project Management**
- **Location**: `backend/main.py`
- **Implementation**: Creates new project for each scan with unique name
- **Status**: ✅ Working - project names include timestamp for uniqueness

**D. Hound Output Parsing**
- **Location**: `backend/main.py` `transform_hound_output()` function
- **Implementation**: Parses `hypotheses.json` structure and maps to Heimdall format
- **Status**: ✅ Working - transforms Hound findings to standard format

#### Testing Hound Integration

```bash
# 1. Test Hound CLI directly
cd agents/hound
source .venv/bin/activate
python3 hound.py project create test_project /path/to/code
python3 hound.py graph build test_project
python3 hound.py agent audit test_project
python3 hound.py finalize test_project

# 2. Test whitelist builder
python3 whitelist_builder.py --input /path/to/code --output whitelist.txt --limit-loc 50000

# 3. Check output format
cat ~/.hound/projects/test_project/hypotheses.json

# 4. Test via Heimdall backend
cd ../../backend
source venv/bin/activate
uvicorn main:app --reload
# Then test via frontend or curl
```

### 2. BRAMA Integration ✅

#### Current Implementation (Working)
- ✅ Uses wrapper script: `agents/brama/scan_url.py`
- ✅ Calls via subprocess with URL argument
- ✅ Returns JSON format compatible with Heimdall
- ✅ Includes comprehensive red-team scanning
- ✅ Uses xAI (Grok) for domain analysis

#### Implementation Notes

**A. BRAMA API Keys**
- **Location**: `agents/brama/scan_url.py` and `agentBrama.py`
- **Implementation**: Reads from environment variables
- **Required Keys**:
  - `XAI_API_KEY` - Required (only one required!)
  - `VT_API_KEY` - Optional (xAI fallback available)
  - `BRAVE_API_KEY` - Optional (xAI fallback available)
  - `VOYAGE_API_KEY` - Optional (educational mode only)
  - `UMBRELLA_API_CLIENT` / `UMBRELLA_API_SECRET` - Optional
  - `URL_HAUSE_KEY` - Optional
- **Status**: ✅ Working - only XAI_API_KEY required, others optional with fallbacks

**B. BRAMA Dependency Management**
- **Location**: `agents/brama/requirements.txt`
- **Implementation**: Uses isolated virtual environment
- **Status**: ✅ Working - subprocess isolation prevents conflicts

**C. BRAMA Output Format**
- **Location**: `agents/brama/scan_url.py` `scan_url()` function
- **Implementation**: Comprehensive parsing of domain analysis + red-team findings
- **Status**: ✅ Working - returns structured JSON with all findings

**D. BRAMA Error Handling**
- **Location**: `backend/main.py` `run_brama_scan()` function
- **Implementation**: Graceful error handling with fallbacks
- **Status**: ✅ Working - continues even if optional APIs fail

#### Testing BRAMA Integration

```bash
# 1. Set API keys
export ANTHROPIC_API_KEY="your_key"
export VT_API_KEY="your_key"
export BRAVE_API_KEY="your_key"

# 2. Test BRAMA wrapper directly
cd agents/brama
python3 scan_url.py https://example.com

# 3. Test via Heimdall backend
cd ../../backend
source venv/bin/activate
uvicorn main:app --reload
# Then test via frontend or curl
```

## Configuration Requirements

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Required (Only One!)
XAI_API_KEY=your_xai_key_here

# Optional (Enhanced Features)
VT_API_KEY=your_virustotal_key
BRAVE_API_KEY=your_brave_key

# Optional (Additional Services)
VOYAGE_API_KEY=your_key
UMBRELLA_API_CLIENT=your_client
UMBRELLA_API_SECRET=your_secret
URL_HAUSE_KEY=your_key
```

### Hound Config File

Hound uses `config.yaml` in the Hound directory. Configure it to use xAI:

```yaml
models:
  scout:
    provider: xai
    model: grok-beta
  strategist:
    provider: xai
    model: grok-beta
  lightweight:
    provider: xai
    model: grok-beta
```

See [CONFIGURATION.md](../CONFIGURATION.md) for detailed setup instructions.

## Integration Status Checklist

### Hound ✅
- [x] Verify Hound CLI works: `hound project list`
- [x] Test project creation: `hound project create test /path/to/code`
- [x] Test agent run: `hound agent run test`
- [x] Verify hypotheses.json format matches parser
- [x] Test full integration via Heimdall API
- [x] Handle edge cases (empty results, errors, timeouts)

### BRAMA ✅
- [x] Set required API keys (only XAI_API_KEY needed)
- [x] Test BRAMA wrapper: `python3 scan_url.py <url>`
- [x] Verify JSON output format
- [x] Test with various URL types (http, https, domains)
- [x] Test error handling (invalid URLs, API failures)
- [x] Test full integration via Heimdall API
- [x] Handle rate limits and API quotas (graceful fallbacks)

## Common Issues and Solutions

### Issue: Hound script not found
**Solution**: 
- Check if Hound is in `agents/hound` or root `hound/`
- Verify `scripts/hound` exists and is executable
- If pip-installed, use `hound` command instead of script path

### Issue: BRAMA API key errors
**Solution**:
- Verify all required keys are set: `echo $ANTHROPIC_API_KEY`
- Check key validity by testing BRAMA directly
- Ensure keys are passed to subprocess (already handled via `env=os.environ.copy()`)

### Issue: Dependency conflicts
**Solution**:
- Use separate virtual environments (already set up)
- Install agents in isolated environments
- Use subprocess calls (already implemented)

### Issue: Timeout errors
**Solution**:
- Increase timeout values in `run_hound_scan()` and `run_brama_scan()`
- Add progress tracking for long-running scans
- Implement async/background job processing

## Current Status Summary

✅ **MVP Complete**: Both Hound and BRAMA are fully integrated and working  
✅ **Production Ready**: All core functionality implemented and tested  
✅ **Documentation**: Complete setup and configuration guides available  
✅ **GitHub**: Repository live at https://github.com/wasumayan/Heimdall

## Future Enhancements

1. **Performance Optimization**: Cache results, parallel processing, rate limiting
2. **Enhanced Monitoring**: Log scan durations, success rates, errors
3. **UX Improvements**: Progress indicators, better error messages
4. **Additional Features**: More scan types, custom rules, scheduled scans

## API/CLI Command Reference

### Hound Commands
```bash
# Create project
hound project create <name> <source_path>

# Build knowledge graphs
hound graph build <project_name> --files <comma-separated-list>

# Run agent analysis
hound agent audit <project_name> --iterations 20

# Finalize hypotheses
hound finalize <project_name>

# Generate whitelist (recommended for large repos)
python3 hound/whitelist_builder.py --input <repo_path> --output whitelist.txt --limit-loc 50000

# List projects
hound project list

# Generate report
hound report generate <project_name> --format html

# View hypotheses
cat ~/.hound/projects/<project_name>/hypotheses.json
```

### BRAMA Commands
```bash
# Scan URL (via wrapper)
python3 agents/brama/scan_url.py <url>

# Run interactive BRAMA (for testing)
python3 agents/brama/agentBrama.py
```

## Support

For issues:
1. Check agent logs in subprocess output
2. Verify API keys and configuration
3. Test agents directly before testing integration
4. Review error messages in backend responses

