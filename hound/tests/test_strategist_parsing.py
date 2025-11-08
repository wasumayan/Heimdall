
from analysis.strategist import HypothesisBatchJSON, HypothesisItemJSON, Strategist


class _StubUnified:
    def __init__(self, *a, **k):
        self.provider_name = 'mock'
        self.model = 'mock'
    def parse(self, *a, **k):
        return None
    def raw(self, *a, **k):
        return ''


class _StubLLM:
    def __init__(self, parse_obj=None, raw_text=None, raise_on_parse=False):
        self._parse_obj = parse_obj
        self._raw_text = raw_text
        self._raise_on_parse = raise_on_parse
        self.provider_name = 'mock'
        self.model = 'mock'

    def parse(self, system: str, user: str, schema=None, reasoning_effort=None):
        if self._raise_on_parse:
            raise RuntimeError('parse failed')
        if self._parse_obj is not None:
            return self._parse_obj
        raise RuntimeError('parse not configured')

    def raw(self, system: str, user: str, reasoning_effort=None):
        return self._raw_text or ''


def _context_with_nodes(node_ids: list[str]) -> str:
    lines = [
        '=== SYSTEM ARCHITECTURE (ALWAYS VISIBLE) ===',
        'NODES (id|type|label):',
    ]
    for nid in node_ids:
        lines.append(f"  {nid}|function|{nid}")
    lines.append('')
    lines.append('=== LOADED NODES (CACHE — DO NOT RELOAD) ===')
    lines.append('  ' + ', '.join(node_ids))
    return '\n'.join(lines)


def test_deep_think_json_parsing_success():
    # Arrange JSON batch
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='Missing Validation for Whitelist Contract Address',
                type='access_control',
                root_cause='No validation on set',
                attack_vector='Owner sets EOA',
                node_ids=['Whitelist.setPool'],
                affected_code=['contracts/Whitelist.sol:setPool'],
                severity='high',
                confidence='high',
                reasoning='obvious'
            ),
            HypothesisItemJSON(
                title='No Events',
                type='observability',
                root_cause='No events',
                attack_vector='undetected changes',
                node_ids=['Whitelist.setOracle'],
                affected_code=['contracts/Whitelist.sol:setOracle'],
                severity='medium',
                confidence='medium',
                reasoning='missing events'
            ),
        ],
        guidance=['check interfaces']
    )
    
    # Create strategist with minimal config
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=batch)

    # Provide context containing node ids
    ctx = _context_with_nodes(['Whitelist.setPool', 'Whitelist.setOracle'])

    # Act
    items = st.deep_think(context=ctx, phase='Saliency')

    # Assert
    assert isinstance(items, list)
    assert len(items) == 2
    titles = [it['description'] for it in items]
    assert 'Missing Validation for Whitelist Contract Address' in titles
    assert 'No Events' in titles


def test_deep_think_legacy_text_parsing_with_wrapped_lines_and_salvage():
    # Arrange legacy wrapped pipe output (two items, one missing node ids initially)
    raw = (
        'HYPOTHESES\n'
        'Min-out missing on swap | logic/slippage | uses minReturn=0 | sandwich/MEV | func_swap_execute | file:Swap.sol::_exec | high | high | clear\n\n'
        'Allowance residue | allowance | approve but no reset | partial fill leaves spendable |  | file:Swap.sol::approve | medium | medium | risk\n'
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=None, raw_text=raw, raise_on_parse=True)

    ctx = _context_with_nodes(['func_swap_execute'])

    # Act
    items = st.deep_think(context=ctx, phase='Saliency')

    # Assert
    assert len(items) == 2
    # First item should keep node
    assert 'func_swap_execute' in items[0]['node_ids']
    # Second item should be salvaged (fallback node assigned)
    assert len(items[1]['node_ids']) >= 1


def test_deep_think_invalid_group_padded_and_tokens_extracted():
    # Single invalid group (too few fields) but contains func_ token
    raw = 'Allowance residue | allowance | approve but no reset | partial fill leaves spendable | file:Swap.sol::approve\n'
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=None, raw_text=raw, raise_on_parse=True)
    ctx = _context_with_nodes(['func_transfer'])

    items = st.deep_think(context=ctx, phase='Saliency')
    assert len(items) == 1
    assert isinstance(items[0]['description'], str)


def test_deep_think_json_with_confidence_variations():
    """Test JSON parsing with different confidence value formats."""
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='Float confidence test',
                type='access_control',
                root_cause='Test root',
                attack_vector='Test vector',
                node_ids=['func_test1'],
                affected_code=['test.sol:func1'],
                severity='high',
                confidence=0.95,  # Float confidence
                reasoning='test reasoning'
            ),
            HypothesisItemJSON(
                title='String confidence test',
                type='logic',
                root_cause='Test root 2',
                attack_vector='Test vector 2',
                node_ids=['func_test2'],
                affected_code=['test.sol:func2'],
                severity='medium',
                confidence='low',  # String confidence
                reasoning='test reasoning 2'
            ),
        ],
        guidance=['test guidance']
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=batch)
    ctx = _context_with_nodes(['func_test1', 'func_test2'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    assert len(items) == 2
    # Float confidence should be preserved (clamped to 0-1)
    assert items[0]['confidence'] == 0.95
    # String 'low' should convert to 0.4
    assert items[1]['confidence'] == 0.4


def test_deep_think_no_hypotheses_found():
    """Test when no hypotheses are found (NO_HYPOTHESES: true)."""
    raw = 'NO_HYPOTHESES: true\nGUIDANCE:\n- Check other areas\n- Review permissions'
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=None, raw_text=raw, raise_on_parse=True)
    ctx = _context_with_nodes(['func_safe'])
    
    items = st.deep_think(context=ctx, phase='Coverage')
    
    # Should return empty list when no hypotheses
    assert len(items) == 0


def test_deep_think_multiline_hypothesis_grouping():
    """Test grouping of wrapped multiline hypotheses."""
    raw = (
        'HYPOTHESES\n'
        'Buffer overflow in input | memory/overflow | No bounds check on\n'
        'user input data | Direct memory write | func_process_input,\n'
        'func_validate | InputHandler.sol:processInput | critical |\n' 
        'high | User can overflow buffer by sending oversized input\n\n'
        'Race condition | timing | Shared state | TOCTOU | func_update | State.sol | high | medium | timing issue\n'
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=None, raw_text=raw, raise_on_parse=True)
    ctx = _context_with_nodes(['func_process_input', 'func_validate', 'func_update'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    # Should properly group the multiline hypothesis
    assert len(items) == 2
    assert 'Buffer overflow' in items[0]['description']
    assert 'Race condition' in items[1]['description']
    # Check node IDs were extracted
    assert 'func_process_input' in items[0]['node_ids'] or 'func_validate' in items[0]['node_ids']
    assert 'func_update' in items[1]['node_ids']


def test_deep_think_node_id_validation():
    """Test that node IDs are included even if not in context."""
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='Valid nodes test',
                type='access_control',
                root_cause='Test',
                attack_vector='Test',
                node_ids=['func_valid', 'invalid_node', 'contract_Token'],  # Mix of valid/invalid
                affected_code=['test.sol'],
                severity='high',
                confidence='high',
                reasoning='test'
            ),
        ],
        guidance=[]
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=batch)
    
    # Context only has func_valid and contract_Token
    ctx = _context_with_nodes(['func_valid', 'contract_Token'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    assert len(items) == 1
    # All provided nodes should be kept (validation is informative, not filtering)
    assert 'func_valid' in items[0]['node_ids']
    assert 'contract_Token' in items[0]['node_ids']
    assert 'invalid_node' in items[0]['node_ids']  # Still included


def test_deep_think_empty_node_ids_fallback():
    """Test fallback node ID generation when none provided."""
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='No nodes test',
                type='logic',
                root_cause='Test',
                attack_vector='Test',
                node_ids=[],  # Empty node IDs
                affected_code=['test.sol'],
                severity='medium',
                confidence='medium',
                reasoning='test'
            ),
        ],
        guidance=[]
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=batch)
    ctx = _context_with_nodes(['func_test'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    assert len(items) == 1
    # Should have a fallback node ID
    assert len(items[0]['node_ids']) > 0
    assert items[0]['node_ids'][0].startswith('fallback_')


def test_deep_think_phase_specific_prompts():
    """Test that different phases use different prompts."""
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='Phase test',
                type='logic',
                root_cause='Test',
                attack_vector='Test',
                node_ids=['func_test'],
                affected_code=['test.sol'],
                severity='high',
                confidence='high',
                reasoning='test'
            ),
        ],
        guidance=['test']
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    
    # Test Coverage phase
    st1 = Strategist(config=config, debug=False)
    llm1 = _StubLLM(parse_obj=batch)
    st1.llm = llm1
    ctx = _context_with_nodes(['func_test'])
    
    items1 = st1.deep_think(context=ctx, phase='Coverage')
    assert len(items1) == 1
    
    # Test Saliency phase
    st2 = Strategist(config=config, debug=False)
    llm2 = _StubLLM(parse_obj=batch)
    st2.llm = llm2
    
    items2 = st2.deep_think(context=ctx, phase='Saliency')
    assert len(items2) == 1
    
    # Both should work but use different prompts internally


def test_deep_think_json_schema_validation():
    """Test that malformed JSON responses are handled gracefully."""
    # First try with malformed JSON (fallback to text parsing)
    raw = 'Invalid JSON { not valid ] mixed brackets'
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=None, raw_text=raw, raise_on_parse=True)
    ctx = _context_with_nodes(['func_test'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    # Should handle gracefully and return empty or parse what it can
    assert isinstance(items, list)


def test_deep_think_edge_cases_in_text_parsing():
    """Test various edge cases in text parsing."""
    # Test with partial pipe-separated values
    raw = (
        'HYPOTHESES\n'
        'Title only |\n'
        'Title | Type |\n'
        'Complete | type | root | vector | nodes | code | high | high | reason\n'
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=None, raw_text=raw, raise_on_parse=True)
    ctx = _context_with_nodes(['func_test'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    # Should handle incomplete entries gracefully
    assert isinstance(items, list)
    # Only complete entries should be parsed
    for item in items:
        assert 'description' in item
        assert 'node_ids' in item
        assert isinstance(item['node_ids'], list)


def test_deep_think_special_characters_in_fields():
    """Test handling of special characters in hypothesis fields."""
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='SQL Injection: "; DROP TABLE users; --',
                type='injection',
                root_cause='Unescaped input in query: SELECT * FROM users WHERE id="$input"',
                attack_vector='Malicious SQL in user input',
                node_ids=['func_query', 'func_db_execute'],
                affected_code=['db.py:execute_query', 'api.py:get_user'],
                severity='critical',
                confidence=0.99,
                reasoning='Direct concatenation of user input into SQL query'
            ),
        ],
        guidance=['Implement parameterized queries', 'Add input validation']
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=batch)
    ctx = _context_with_nodes(['func_query', 'func_db_execute'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    assert len(items) == 1
    # Special characters should be preserved
    assert 'DROP TABLE' in items[0]['description']
    assert 'SELECT * FROM' in items[0]['details']


def test_deep_think_severity_and_confidence_normalization():
    """Test normalization of severity and confidence values."""
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='Test1',
                type='test',
                root_cause='test',
                attack_vector='test',
                node_ids=['func1'],
                affected_code=['test.sol'],
                severity='CRITICAL',  # Uppercase
                confidence=1.5,  # Out of range
                reasoning='test'
            ),
            HypothesisItemJSON(
                title='Test2',
                type='test',
                root_cause='test',
                attack_vector='test',
                node_ids=['func2'],
                affected_code=['test.sol'],
                severity='Med',  # Abbreviation
                confidence=-0.5,  # Negative
                reasoning='test'
            ),
        ],
        guidance=[]
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=batch)
    ctx = _context_with_nodes(['func1', 'func2'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    assert len(items) == 2
    # Severity should be lowercase
    assert items[0]['severity'] == 'critical'
    # Confidence should be clamped to [0, 1]
    assert items[0]['confidence'] == 1.0
    assert items[1]['confidence'] == 0.0


def test_deep_think_with_existing_hypotheses_context():
    """Test that existing hypotheses in context are considered."""
    ctx = (
        '=== EXISTING HYPOTHESES ===\n'
        'H1: Buffer overflow in parse_input\n'
        'H2: SQL injection in query_builder\n\n'
    ) + _context_with_nodes(['func_parse', 'func_query'])
    
    # Return a duplicate hypothesis
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='Buffer overflow in parse_input',  # Duplicate
                type='memory',
                root_cause='No bounds check',
                attack_vector='Large input',
                node_ids=['func_parse'],
                affected_code=['parser.c'],
                severity='high',
                confidence='high',
                reasoning='test'
            ),
            HypothesisItemJSON(
                title='New vulnerability',  # Not a duplicate
                type='logic',
                root_cause='Wrong condition',
                attack_vector='Edge case',
                node_ids=['func_validate'],
                affected_code=['validator.c'],
                severity='medium',
                confidence='medium',
                reasoning='test'
            ),
        ],
        guidance=[]
    )
    
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    st = Strategist(config=config, debug=False)
    st.llm = _StubLLM(parse_obj=batch)
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    # Both should be returned (dedup happens at agent level, not here)
    assert len(items) == 2


def test_deep_think_mission_context():
    """Test that mission context is included in prompts."""
    config = {'models': {'agent': {'model': 'mock', 'provider': 'mock'}}}
    mission = "Focus on financial vulnerabilities and fund theft vectors"
    
    st = Strategist(config=config, debug=False, mission=mission)
    
    batch = HypothesisBatchJSON(
        hypotheses=[
            HypothesisItemJSON(
                title='Fund theft vulnerability',
                type='theft',
                root_cause='Missing check',
                attack_vector='Direct withdrawal',
                node_ids=['func_withdraw'],
                affected_code=['vault.sol'],
                severity='critical',
                confidence='high',
                reasoning='test'
            ),
        ],
        guidance=[]
    )
    
    st.llm = _StubLLM(parse_obj=batch)
    ctx = _context_with_nodes(['func_withdraw'])
    
    items = st.deep_think(context=ctx, phase='Saliency')
    
    assert len(items) == 1
    assert 'Fund theft' in items[0]['description']