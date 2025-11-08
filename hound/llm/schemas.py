"""Pydantic schemas for LLM structured outputs."""
from typing import Any, Literal

from pydantic import BaseModel, Field

# Type aliases
Gran = Literal["micro", "meso", "macro"]
IssueStatus = Literal["hypothesis", "suspected", "confirmed", "retracted"]
Severity = Literal["low", "medium", "high", "critical"]


class SAspect(BaseModel):
    """Security aspect node."""
    model_config = {"extra": "forbid"}
    temp_id: str = Field(description="Temporary ID for this extraction session")
    label: str = Field(description="Human-readable label for the aspect")
    kind: str | None = Field(None, description="Type/category of aspect")
    granularity: Gran = Field(description="Granularity level: micro/meso/macro")
    source_files: list[str] = Field(default_factory=list, description="Source files this aspect relates to")
    tags: list[str] = Field(default_factory=list, description="Tags/flags for this aspect")
    salience: float = Field(0.5, ge=0.0, le=1.0, description="Importance score 0-1")
    summary_128: str | None = Field(None, description="Short summary (max 128 chars)")
    summary_512: str | None = Field(None, description="Medium summary (max 512 chars)")
    memo_full: str | None = Field(None, description="Full description/memo")


class SEvidence(BaseModel):
    """Code evidence/snippet reference."""
    model_config = {"extra": "forbid"}
    temp_id: str = Field(description="Temporary ID for this extraction session")
    relpath: str = Field(description="Relative path to the file")
    char_start: int = Field(ge=0, description="Character offset start position")
    char_end: int = Field(gt=0, description="Character offset end position")
    snippet_preview: str | None = Field(None, description="Preview of the code snippet")


class SStatement(BaseModel):
    """Reified statement/relationship between aspects."""
    model_config = {"extra": "forbid"}
    temp_id: str = Field(description="Temporary ID for this extraction session")
    predicate: str = Field(description="Relationship type (PART_OF, CALLS, USES, IMPLEMENTS, etc.)")
    subject_temp_id: str = Field(description="Source aspect temp_id")
    object_temp_id: str | None = Field(None, description="Target aspect temp_id (optional)")
    confidence: float = Field(0.6, ge=0.0, le=1.0, description="Confidence score 0-1")
    evidence_temp_ids: list[str] = Field(default_factory=list, description="Supporting evidence temp_ids")


class SIssue(BaseModel):
    """Security issue/vulnerability."""
    model_config = {"extra": "forbid"}
    temp_id: str = Field(description="Temporary ID for this extraction session")
    title: str = Field(description="Issue title/summary")
    status: IssueStatus = Field("hypothesis", description="Issue status")
    severity: Severity | None = Field(None, description="Issue severity")
    explanation: str | None = Field(None, description="Detailed explanation")
    evidence_temp_ids: list[str] = Field(default_factory=list, description="Supporting evidence temp_ids")
    affected_aspect_ids: list[str] = Field(default_factory=list, description="Affected aspect temp_ids")


class GraphPatchProto(BaseModel):
    """Complete graph patch returned by LLM."""
    model_config = {"extra": "forbid"}
    aspects: list[SAspect] = Field(default_factory=list)
    evidences: list[SEvidence] = Field(default_factory=list)
    statements: list[SStatement] = Field(default_factory=list)
    issues: list[SIssue] = Field(default_factory=list)


# Analysis-specific schemas

class BundleInfo(BaseModel):
    """Information about a code bundle for LLM processing."""
    model_config = {"extra": "forbid"}
    id: str
    files: list[str]
    total_chars: int
    preview: str | None = None


class RepoConceptMap(BaseModel):
    """Repository-level concept map (P1 output)."""
    model_config = {"extra": "forbid"}
    macro_aspects: list[SAspect]
    relationships: list[SStatement]
    initial_hypotheses: list[SIssue] = Field(default_factory=list)


class BundleAnalysis(BaseModel):
    """Bundle-level analysis (P2 output)."""
    model_config = {"extra": "forbid"}
    bundle_id: str
    micro_aspects: list[SAspect]
    meso_aspects: list[SAspect]
    evidences: list[SEvidence]
    statements: list[SStatement]
    issues: list[SIssue] = Field(default_factory=list)


class CrossBundleStitch(BaseModel):
    """Cross-bundle stitching results (P3 output)."""
    model_config = {"extra": "forbid"}
    merge_suggestions: list[dict[str, Any]]
    new_relationships: list[SStatement]
    elevated_aspects: list[SAspect]
    refined_issues: list[SIssue]