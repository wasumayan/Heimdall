# Whitelist Builder Guide

## Overview

The whitelist builder automatically generates a list of files to analyze within a specified lines-of-code (LOC) budget. This is **highly recommended** for large repositories to:

- Focus analysis on important files
- Stay within LLM context limits
- Improve audit performance
- Reduce token costs

## How It Works

1. **Scans the repository** for relevant source files
2. **Counts lines of code** for each file
3. **Selects files** within the LOC budget (prioritizing larger files)
4. **Generates a comma-separated list** for Hound's `--files` argument

## Usage

### Automatic (Recommended)

The whitelist builder runs automatically when:
- `auto_generate_whitelist` is `true` (default)
- No manual `graph_file_filter` is provided

**Default LOC Budget**: 50,000 lines

### Manual Configuration

1. **Via UI**:
   - Enter GitHub repo URL
   - Click "Configure" in the whitelist card
   - Choose "Auto-generate" or "Manual"
   - Set LOC budget (10,000 - 200,000)
   - Or enter comma-separated file paths

2. **Via API**:
   ```json
   {
     "github_repo": "https://github.com/username/repo",
     "auto_generate_whitelist": true,
     "whitelist_loc_budget": 50000
   }
   ```

### Command Line

```bash
cd hound
python3 whitelist_builder.py \
  --input /path/to/repo \
  --output whitelist.txt \
  --limit-loc 50000 \
  --print-summary \
  --verbose
```

## Recommended LOC Budgets

- **Small projects** (<20k LOC): 20,000
- **Medium projects** (20-80k LOC): 50,000 (default)
- **Large projects** (>80k LOC): 100,000+

## File Selection

The builder:
- Includes common source file extensions (`.py`, `.js`, `.ts`, `.sol`, `.rs`, `.go`, etc.)
- Excludes common directories (`.git`, `node_modules`, `venv`, `dist`, `build`, etc.)
- Prioritizes larger files (more important code)
- Stays within LOC budget

## Example Output

```
src/main.py,src/utils.py,lib/helpers.js,config/settings.json
```

This list is then passed to Hound's `graph build` command via the `--files` argument.

## Integration

The whitelist builder is automatically integrated into the audit workflow:

1. Repository is cloned to temporary directory
2. Whitelist builder runs (if auto-generation enabled)
3. Generated whitelist is used for `graph build`
4. Hound analyzes only the whitelisted files
5. Results are returned with full findings

## Troubleshooting

### No files selected
- Increase LOC budget
- Check that repository has source files
- Verify file extensions are supported

### Too many files selected
- Decrease LOC budget
- Use manual whitelist for precise control

### Whitelist generation fails
- Check repository is accessible
- Verify Python dependencies are installed
- Check backend logs for errors

## References

- Based on: https://muellerberndt.medium.com/hunting-for-security-bugs-in-code-with-ai-agents-a-full-walkthrough-a0dc24e1adf0
- Hound documentation: https://github.com/scabench-org/hound

