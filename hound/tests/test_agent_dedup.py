import json

from analysis.agent_core import AutonomousAgent as Scout


class _StubLLMClient:
    def __init__(self):
        self.provider_name = 'mock'
        self.model = 'mock'

    def raw(self, system: str, user: str, reasoning_effort=None):
        return '{}'

    def parse(self, system: str, user: str, schema=None, reasoning_effort=None):
        raise RuntimeError('not used in these tests')


class _FakeStore:
    def __init__(self, existing=None, propose_result=(True, 'hyp_new')):
        self._existing = existing or []
        self._propose_result = propose_result

    def list_all(self):
        return list(self._existing)

    def propose(self, hypothesis):
        return self._propose_result

    def _load_data(self):
        return {'hypotheses': {}}

    def add_evidence(self, hypothesis_id, evidence):
        return True

    def adjust_confidence(self, hypothesis_id, confidence, reason):
        return True


class _FakeStrategist:
    def __init__(self, items):
        self._items = items

    def deep_think(self, *, context: str, phase: str = None):
        return list(self._items)


def _make_agent(tmp_path, monkeypatch, strategist_items, store, dup_ids=None):
    # Minimal files to satisfy init
    proj = tmp_path / 'proj'
    (proj / 'graphs').mkdir(parents=True)
    (proj / 'manifest').mkdir()
    (proj / 'graphs' / 'knowledge_graphs.json').write_text(json.dumps({'graphs': {}}))
    (proj / 'manifest' / 'manifest.json').write_text(json.dumps({'repo_path': str(proj)}))

    # Monkeypatch LLM client used in agent init
    # Patch the real UnifiedLLMClient used by agent to a stub so construction works
    monkeypatch.setattr('llm.unified_client.UnifiedLLMClient', lambda **kwargs: _StubLLMClient())
    # Monkeypatch Strategist symbol used inside _deep_think
    fake_strat = _FakeStrategist(strategist_items)
    # Patch the import target used inside _deep_think: analysis.strategist.Strategist
    monkeypatch.setattr('analysis.strategist.Strategist', lambda **kwargs: fake_strat)
    # Monkeypatch LLM-dedup function
    if dup_ids is not None:
        monkeypatch.setattr('analysis.hypothesis_dedup.check_duplicates_llm', lambda **kw: set(dup_ids))
    else:
        monkeypatch.setattr('analysis.hypothesis_dedup.check_duplicates_llm', lambda **kw: set())

    agent = Scout(
        graphs_metadata_path=proj / 'graphs' / 'knowledge_graphs.json',
        manifest_path=proj / 'manifest',
        agent_id='test_agent',
        config={'models': {'agent': {'provider': 'mock', 'model': 'mock'}}},
        debug=False,
        session_id='sess'
    )
    # Replace hypothesis store
    agent.hypothesis_store = store
    # Ensure no guidance client needed
    agent.guidance_client = None
    return agent


def _mk_item(title='H1', node_ids=None):
    return {
        'description': title,
        'details': 'x',
        'vulnerability_type': 'access_control',
        'severity': 'high',
        'confidence': 0.9,
        'node_ids': node_ids or ['func_a'],
        'reasoning': 'r',
    }


def test_store_level_dedup_reason(monkeypatch, tmp_path):
    items = [_mk_item('Duplicate Title', ['func_x'])]
    existing = [{
        'id': 'hyp_old', 'title': 'Duplicate Title', 'description': 'desc',
        'vulnerability_type': 'access_control', 'node_refs': ['func_x']
    }]
    store = _FakeStore(existing=existing, propose_result=(False, 'Similar to existing: hyp_old'))
    agent = _make_agent(tmp_path, monkeypatch, items, store, dup_ids=set())

    res = agent._deep_think()
    assert res['status'] == 'success'
    assert res['hypotheses_formed'] == 0
    assert res.get('dedup_skipped', 0) >= 1
    details = ' '.join(res.get('dedup_details') or [])
    assert 'Store-dedup' in details and 'hyp_old' in details


def test_llm_dedup_reason(monkeypatch, tmp_path):
    items = [_mk_item('Very Similar', ['func_y'])]
    store = _FakeStore(existing=[{'id': 'hyp_1', 'title': 'Prev', 'description': 'd', 'vulnerability_type': 'access_control', 'node_refs': ['func_y']}], propose_result=(True, 'hyp_new'))
    # LLM dedup flags as duplicate -> propose() is skipped
    agent = _make_agent(tmp_path, monkeypatch, items, store, dup_ids={'hyp_1'})

    res = agent._deep_think()
    assert res['status'] == 'success'
    assert res['hypotheses_formed'] == 0
    assert res.get('dedup_skipped', 0) >= 1
    details = ' '.join(res.get('dedup_details') or [])
    assert 'LLM-dedup' in details
