# GitHub Issue: Implement Card-Based Coverage Tracking for Smart Coverage Top-Up

## Problem Statement

The current coverage tracking system only tracks which nodes have been visited, but doesn't track:
1. How many code cards (source files) are associated with each node
2. Which specific cards have been analyzed for each node

This leads to the coverage top-up feature suggesting irrelevant micro-details (like `AccessControl.DEFAULT_ADMIN_ROLE`) instead of focusing on nodes with many unanalyzed code cards.

## Current Limitations

### What We Have Now
- `SessionTracker` tracks visited nodes and cards separately
- Coverage reports show unvisited nodes as a flat list
- No connection between nodes and their associated cards
- Coverage top-up was disabled because it can't make intelligent decisions

### Problems This Causes
1. **Poor Prioritization**: Coverage suggestions include constants, roles, and other micro-details that have no associated code
2. **No Card Awareness**: Can't identify nodes with many unanalyzed files
3. **Wasted Iterations**: Agent spends time on trivial nodes while missing important components with extensive code

## Proposed Solution

### 1. Enhanced Node Metadata Structure

Each node in the knowledge graph should include:
```json
{
  "id": "contract_Token",
  "type": "contract",
  "label": "Token Contract",
  "artifacts": [
    {
      "type": "code_card",
      "path": "contracts/Token.sol",
      "lines": 450,
      "complexity": "high"
    },
    {
      "type": "code_card", 
      "path": "contracts/extensions/TokenVesting.sol",
      "lines": 200,
      "complexity": "medium"
    }
  ]
}
```

### 2. Enhanced Coverage Tracking

The `SessionTracker` should maintain:
```python
class EnhancedCoverageData:
    node_card_mapping: dict[str, list[str]]  # node_id -> [card_paths]
    visited_cards_per_node: dict[str, set[str]]  # node_id -> {visited_card_paths}
    
    def get_node_coverage(self, node_id: str) -> dict:
        total_cards = len(self.node_card_mapping.get(node_id, []))
        visited_cards = len(self.visited_cards_per_node.get(node_id, set()))
        return {
            'total_cards': total_cards,
            'visited_cards': visited_cards,
            'coverage_percent': (visited_cards / total_cards * 100) if total_cards > 0 else 100,
            'unvisited_cards': total_cards - visited_cards
        }
```

### 3. Smart Coverage Summary Generation

The coverage summary passed to the Strategist should include:
```
=== COVERAGE STATUS ===
Overall: 75% nodes visited, 60% cards analyzed

HIGH-VALUE UNVISITED TARGETS (nodes with many unanalyzed cards):
- contract_Vault (12 unvisited cards out of 15)
- module_Governance (8 unvisited cards out of 10)
- service_Oracle (6 unvisited cards out of 8)

FULLY ANALYZED COMPONENTS:
- contract_Token (5/5 cards analyzed)
- module_Access (3/3 cards analyzed)
```

### 4. Strategist Coverage Top-Up Logic

With this information, the Strategist can make intelligent decisions:
```python
def select_coverage_target(coverage_data: EnhancedCoverageData) -> str:
    # Get all nodes with unvisited cards
    candidates = []
    for node_id in coverage_data.node_card_mapping:
        node_cov = coverage_data.get_node_coverage(node_id)
        if node_cov['unvisited_cards'] > 0:
            candidates.append({
                'node_id': node_id,
                'unvisited_cards': node_cov['unvisited_cards'],
                'total_cards': node_cov['total_cards'],
                'priority_score': node_cov['unvisited_cards'] * (1 - node_cov['coverage_percent']/100)
            })
    
    # Sort by priority score (most unvisited cards with lowest coverage)
    candidates.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Return top candidate
    return candidates[0]['node_id'] if candidates else None
```

## Implementation Steps

### Phase 1: Knowledge Graph Enhancement
1. Modify graph generation to include `artifacts` field for each node
2. Map code files to their corresponding nodes during graph creation
3. Store this mapping in the knowledge graph JSON

### Phase 2: Coverage Tracking Enhancement
1. Extend `SessionTracker` to maintain node-card mappings
2. Track card visits per node (not just global card visits)
3. Calculate per-node coverage statistics

### Phase 3: Coverage Reporting
1. Generate enhanced coverage summaries with card counts
2. Prioritize nodes by unvisited card count
3. Pass this to Strategist in a structured format

### Phase 4: Strategist Integration
1. Re-enable coverage top-up in Saliency mode only
2. Use card-based prioritization for target selection
3. Skip nodes with no cards or all cards visited

## Success Criteria

1. **No More Micro-Details**: Coverage top-up should never suggest constants, roles, or other nodes without code cards
2. **Smart Prioritization**: Always suggest nodes with the most unvisited code cards first
3. **Clear Rationale**: Coverage suggestions should show card counts: "contract_Vault has 12 unanalyzed code files"
4. **Measurable Impact**: Increase in meaningful vulnerability discoveries per investigation

## Benefits

1. **Efficiency**: Focus on code-rich components that are likely to contain vulnerabilities
2. **Completeness**: Ensure all significant code is analyzed
3. **Intelligence**: Make data-driven decisions about what to analyze next
4. **Transparency**: Clear metrics showing why certain components are prioritized

## Notes

- This requires changes to both the graph generation phase and the runtime agent
- The knowledge graph schema needs to be extended to include artifact metadata
- Backward compatibility: System should work with old graphs that don't have artifact data

## Related Files

- `analysis/session_tracker.py` - Needs enhancement for node-card tracking
- `analysis/strategist.py` - Currently has coverage top-up disabled (line 350-355)
- `knowledge_graph/` - Graph generation needs to include artifact mapping
- `commands/agent.py` - Coverage summary generation (around line 1308)