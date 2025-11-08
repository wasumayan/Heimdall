# API/CLI Integration Guide

This document describes what needs to be updated to call the actual Hound and BRAMA APIs/CLIs.

## Current Status

✅ **Infrastructure Complete**: Both Hound and BRAMA are integrated via subprocess calls  
✅ **Path Resolution**: Backend automatically finds Hound in either `agents/hound` or root `hound/`  
✅ **Error Handling**: Both integrations have proper error handling and fallbacks  
✅ **Mock Data**: System falls back to mock data if agents are not available  
✅ **Red-Team Features**: BRAMA now includes comprehensive red-teaming capabilities:
   - Security headers analysis
   - SSL/TLS certificate checks
   - Endpoint discovery
   - Technology stack fingerprinting
   - CORS misconfiguration detection
   - HTTP methods testing
   - Information disclosure checks  

## What Needs to Be Updated

### 1. Hound Integration

#### Current Implementation
- ✅ Creates Hound project via CLI: `hound project create <name> <path>`
- ✅ Runs agent analysis: `hound agent run <project_name>`
- ✅ Reads findings from: `~/.hound/projects/{project_name}/hypotheses.json`

#### What May Need Adjustment

**A. Hound Script Path**
- **Location**: `backend/main.py` line ~279
- **Current**: Uses `HOUND_PATH / "scripts" / "hound"`
- **May need**: Check if Hound is installed via `pip install` vs. local directory
- **Fix**: If Hound is pip-installed, use `hound` command directly instead of script path

**B. Hound Configuration**
- **Location**: Hound requires `config.yaml` with LLM API keys
- **Current**: Assumes config exists or uses defaults
- **May need**: Create default config or prompt user to configure
- **Fix**: Check for `config.yaml` in Hound directory, create template if missing

**C. Hound Project Management**
- **Location**: `backend/main.py` line ~282-301
- **Current**: Creates new project for each scan
- **May need**: Reuse existing projects or cleanup old projects
- **Fix**: Add project cleanup logic or reuse existing projects

**D. Hound Output Parsing**
- **Location**: `backend/main.py` `transform_hound_output()` function
- **Current**: Parses `hypotheses.json` structure
- **May need**: Adjust based on actual Hound output format
- **Fix**: Test with real Hound output and adjust field mappings

#### Testing Hound Integration

```bash
# 1. Test Hound CLI directly
cd agents/hound
source .venv/bin/activate
python3 scripts/hound project create test_project /path/to/code
python3 scripts/hound agent run test_project

# 2. Check output format
cat ~/.hound/projects/test_project/hypotheses.json

# 3. Test via Heimdall backend
cd ../../backend
source venv/bin/activate
uvicorn main:app --reload
# Then test via frontend or curl
```

### 2. BRAMA Integration

#### Current Implementation
- ✅ Uses wrapper script: `agents/brama/scan_url.py`
- ✅ Calls via subprocess with URL argument
- ✅ Returns JSON format compatible with Heimdall

#### What May Need Adjustment

**A. BRAMA API Keys**
- **Location**: `agents/brama/scan_url.py` and `agentBrama.py`
- **Current**: Reads from environment variables
- **Required Keys**:
  - `ANTHROPIC_API_KEY` - Required
  - `VT_API_KEY` - Required (VirusTotal)
  - `BRAVE_API_KEY` - Required
  - `VOYAGE_API_KEY` - Optional
  - `UMBRELLA_API_CLIENT` - Optional
  - `UMBRELLA_API_SECRET` - Optional
  - `URL_HAUSE_KEY` - Optional
- **Fix**: Document required keys, create `.env.example`, validate keys before scanning

**B. BRAMA Dependency Conflicts**
- **Location**: `agents/brama/requirements.txt`
- **Current**: Has dependency conflicts (h11 version issue)
- **May need**: Fix requirements.txt or use `--no-deps` flag
- **Fix**: Update requirements.txt or install in isolated environment

**C. BRAMA Output Format**
- **Location**: `agents/brama/scan_url.py` `scan_url()` function
- **Current**: Basic parsing of domain analysis
- **May need**: More sophisticated parsing of VirusTotal, URLhaus, Umbrella results
- **Fix**: Enhance parsing to extract specific vulnerability types and details

**D. BRAMA Error Handling**
- **Location**: `backend/main.py` `run_brama_scan()` function
- **Current**: Basic error handling
- **May need**: Better handling of API rate limits, timeouts, missing keys
- **Fix**: Add retry logic, better error messages, API key validation

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
# Hound Configuration (if using API keys)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# BRAMA Configuration (Required)
ANTHROPIC_API_KEY=your_key_here
VT_API_KEY=your_virustotal_key
BRAVE_API_KEY=your_brave_key

# Optional BRAMA Keys
VOYAGE_API_KEY=your_key
UMBRELLA_API_CLIENT=your_client
UMBRELLA_API_SECRET=your_secret
URL_HAUSE_KEY=your_key
```

### Hound Config File

If Hound requires `config.yaml`, create it in `agents/hound/`:

```yaml
models:
  scout:
    provider: openai
    model: gpt-4
  strategist:
    provider: openai
    model: gpt-4
  lightweight:
    provider: openai
    model: gpt-3.5-turbo
```

## Integration Checklist

### Hound
- [ ] Verify Hound CLI works: `hound project list`
- [ ] Test project creation: `hound project create test /path/to/code`
- [ ] Test agent run: `hound agent run test`
- [ ] Verify hypotheses.json format matches parser
- [ ] Test full integration via Heimdall API
- [ ] Handle edge cases (empty results, errors, timeouts)

### BRAMA
- [ ] Set all required API keys
- [ ] Test BRAMA wrapper: `python3 scan_url.py <url>`
- [ ] Verify JSON output format
- [ ] Test with various URL types (http, https, domains)
- [ ] Test error handling (invalid URLs, API failures)
- [ ] Test full integration via Heimdall API
- [ ] Handle rate limits and API quotas

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

## Next Steps

1. **Test with Real Data**: Run both integrations with actual repositories and URLs
2. **Refine Parsing**: Adjust output parsers based on real agent outputs
3. **Add Monitoring**: Log scan durations, success rates, errors
4. **Improve UX**: Add progress indicators, better error messages
5. **Optimize Performance**: Cache results, parallel processing, rate limiting

## API/CLI Command Reference

### Hound Commands
```bash
# Create project
hound project create <name> <source_path>

# List projects
hound project list

# Run agent analysis
hound agent run <project_name>

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

