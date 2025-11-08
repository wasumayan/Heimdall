# Heimdall Configuration Guide

## API Keys Setup

Heimdall uses **xAI (Grok)** for both Hound and BRAMA agents. Configure API keys using one of these methods:

### Method 1: Environment Variables (Recommended)

Create a `.env` file in the `backend/` directory:

```bash
cd backend
cp .env.example .env
# Then edit .env with your API keys
```

Or set them directly in your shell:

```bash
export XAI_API_KEY="your_key"
export VT_API_KEY="your_key"
export BRAVE_API_KEY="your_key"
```

### Method 2: System Environment Variables

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export XAI_API_KEY="your_key"
export VT_API_KEY="your_key"
export BRAVE_API_KEY="your_key"
```

Then reload: `source ~/.zshrc`

## Required API Keys

### xAI API Key (Required - Only One!)

1. **XAI_API_KEY** (Required - This is the ONLY required key)
   - Get from: https://console.x.ai/
   - Used for AI analysis in both Hound (codebase audits) and BRAMA (website scans)
   - This is the primary LLM provider for Heimdall
   - **No Chrome/Chromium APIs needed** - Heimdall uses pure Python libraries
   - **No browser automation** - All scanning uses HTTP requests and standard libraries
   - **This is the only required key** - Everything else is optional!

## Optional API Keys (Enhanced Features)

### For BRAMA (Website Scans) - Optional but Recommended

2. **VT_API_KEY** (Optional)
   - Get from: https://www.virustotal.com/gui/my-apikey
   - Used for VirusTotal scans
   - **Workaround**: If not provided, BRAMA will use xAI to analyze domains directly

3. **BRAVE_API_KEY** (Optional)
   - Get from: https://api.search.brave.com/register
   - Used for web search (especially for phone number analysis)
   - **Workaround**: If not provided, phone analysis will use xAI pattern matching

## Optional API Keys (All Optional - Not Required!)

These keys enhance functionality but are **NOT required**:

- `VT_API_KEY` - VirusTotal scans (optional - xAI fallback available)
- `BRAVE_API_KEY` - Web search for phone numbers (optional - xAI fallback available)
- `VOYAGE_API_KEY` - Only for educational mode (not used in main scanning)
- `UMBRELLA_API_CLIENT` / `UMBRELLA_API_SECRET` - Cisco Umbrella (optional)
- `URL_HAUSE_KEY` - URLHaus malware database (optional)

**Note**: 
- No Chrome/Chromium browser APIs needed
- No browser automation required
- All scanning uses standard Python libraries (requests, ssl, socket)
- Red-team scanner works with zero API keys (only needs XAI_API_KEY for domain analysis)

## Verify Configuration

After setting API keys, verify they're loaded:

```bash
cd backend
source venv/bin/activate
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('XAI_API_KEY:', 'SET' if os.getenv('XAI_API_KEY') else 'NOT SET')"
```

## Hound Configuration

Hound uses xAI by default. To configure Hound to use xAI:

1. Copy the example config: `cp agents/hound/config.yaml.example agents/hound/config.yaml`
2. Edit `config.yaml` and set all models to use xAI provider:

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

3. Ensure `XAI_API_KEY` is set in your environment

## Troubleshooting

### API Keys Not Working

1. Check if `.env` file exists in `backend/` directory
2. Verify keys are set: `echo $XAI_API_KEY`
3. Restart the backend after setting keys
4. Check backend logs for API key errors

### BRAMA Errors

- **Minimum**: Only XAI_API_KEY is required
- **Enhanced features**: VT_API_KEY and BRAVE_API_KEY provide additional scan capabilities
- If optional keys are missing, BRAMA will use xAI-only analysis (still functional!)
- Verify XAI_API_KEY is valid by testing BRAMA directly
- Check API quotas/limits

### Hound Errors

- Ensure `XAI_API_KEY` is set
- Check Hound's `config.yaml` if using custom configuration
- Verify Hound can access the codebase path
- Make sure `xai-sdk` is installed in Hound's environment

