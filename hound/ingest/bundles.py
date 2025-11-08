"""Adaptive bundling system for grouping related code cards."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# Disable joblib parallelization to avoid multiprocessing warnings
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

import networkx as nx
from sklearn.cluster import SpectralClustering


@dataclass
class Bundle:
    """A bundle of related cards."""
    id: str
    card_ids: list[str]
    file_paths: list[str]
    total_chars: int
    preview: str  # Brief description of bundle contents
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class AdaptiveBundler:
    """Groups cards into bundles based on similarity and size constraints."""
    
    def __init__(self, cards: list, files: list, config: dict):
        """Initialize bundler with cards and configuration."""
        self.cards = {c.id: c for c in cards}
        self.files = files
        self.config = config
        
        # Bundling parameters
        bundle_config = config.get("bundling", {})
        self.target_chars = bundle_config.get("target_chars", 25000)
        self.max_bundle_chars = int(self.target_chars * 1.5)  # Allow 50% over
        self.min_bundle_chars = int(self.target_chars * 0.3)  # At least 30% of target
        
        # Build similarity graph
        self.graph = self._build_similarity_graph()
    
    def _build_similarity_graph(self) -> nx.Graph:
        """Build weighted graph of card similarities."""
        G = nx.Graph()
        
        # Add all cards as nodes
        for card_id, card in self.cards.items():
            G.add_node(card_id, **{
                "relpath": card.relpath,
                "size": card.char_end - card.char_start,
                "shingle_hash": card.shingle_hash,
                "top_tokens": card.top_tokens
            })
        
        # Add edges based on similarity
        card_list = list(self.cards.values())
        for i, card1 in enumerate(card_list):
            for card2 in card_list[i+1:]:
                similarity = self._compute_similarity(card1, card2)
                if similarity > 0.1:  # Only add edge if meaningful similarity
                    G.add_edge(card1.id, card2.id, weight=similarity)
        
        return G
    
    def _compute_similarity(self, card1, card2) -> float:
        """Compute similarity between two cards."""
        score = 0.0
        
        # 1. File proximity (same file or same directory)
        path1 = Path(card1.relpath)
        path2 = Path(card2.relpath)
        
        if path1 == path2:
            score += 0.5  # Same file
        elif path1.parent == path2.parent:
            score += 0.3  # Same directory
        elif path1.parent.parent == path2.parent.parent:
            score += 0.1  # Same parent directory
        
        # 2. Token overlap (Jaccard similarity)
        if card1.top_tokens and card2.top_tokens:
            set1 = set(card1.top_tokens)
            set2 = set(card2.top_tokens)
            if set1 or set2:
                jaccard = len(set1 & set2) / len(set1 | set2)
                score += jaccard * 0.3
        
        # 3. Shingle hash similarity (simplified)
        if card1.shingle_hash == card2.shingle_hash:
            score += 0.2
        
        return min(score, 1.0)
    
    def create_bundles(self) -> list[Bundle]:
        """Create bundles using graph clustering."""
        if not self.cards:
            return []
        
        # Special case: very small repository
        total_chars = sum(c.char_end - c.char_start for c in self.cards.values())
        if total_chars <= self.target_chars:
            # Everything fits in one bundle
            return [self._create_single_bundle(list(self.cards.keys()))]
        
        # Use spectral clustering on the similarity graph
        bundles = self._cluster_cards()
        
        # Post-process bundles to respect size constraints
        bundles = self._optimize_bundle_sizes(bundles)
        
        return bundles
    
    def _cluster_cards(self) -> list[Bundle]:
        """Cluster cards using spectral clustering."""
        # Convert graph to adjacency matrix
        node_list = list(self.graph.nodes())
        n = len(node_list)
        
        if n == 0:
            return []
        
        if n == 1:
            return [self._create_single_bundle(node_list)]
        
        # Build adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph, nodelist=node_list).todense()
        
        # Estimate number of clusters based on total size
        total_chars = sum(self.graph.nodes[n]["size"] for n in node_list)
        estimated_clusters = max(1, int(total_chars / self.target_chars))
        estimated_clusters = min(estimated_clusters, n)  # Can't have more clusters than nodes
        
        # Apply spectral clustering with n_jobs=1 to disable parallelization
        try:
            clustering = SpectralClustering(
                n_clusters=estimated_clusters,
                affinity='precomputed',
                random_state=42,
                n_jobs=1  # Disable parallelization to avoid joblib warning
            )
            labels = clustering.fit_predict(adj_matrix)
        except Exception:
            # Fallback to simple grouping by file
            return self._fallback_clustering()
        
        # Group cards by cluster label
        clusters: dict[int, list[str]] = {}
        for card_id, label in zip(node_list, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(card_id)
        
        # Create bundles from clusters
        bundles = []
        for i, card_ids in enumerate(clusters.values()):
            bundle = self._create_bundle(f"bundle_{i:03d}", card_ids)
            if bundle:
                bundles.append(bundle)
        
        return bundles
    
    def _fallback_clustering(self) -> list[Bundle]:
        """Fallback clustering by grouping cards from same files."""
        bundles = []
        current_bundle_cards = []
        current_size = 0
        bundle_idx = 0
        
        # Group by file
        for file_info in self.files:
            file_cards = file_info.card_ids
            file_size = sum(
                self.cards[cid].char_end - self.cards[cid].char_start 
                for cid in file_cards if cid in self.cards
            )
            
            # If adding this file would exceed limit, create new bundle
            if current_size > 0 and current_size + file_size > self.max_bundle_chars:
                bundle = self._create_bundle(f"bundle_{bundle_idx:03d}", current_bundle_cards)
                if bundle:
                    bundles.append(bundle)
                bundle_idx += 1
                current_bundle_cards = []
                current_size = 0
            
            current_bundle_cards.extend(file_cards)
            current_size += file_size
        
        # Add remaining cards
        if current_bundle_cards:
            bundle = self._create_bundle(f"bundle_{bundle_idx:03d}", current_bundle_cards)
            if bundle:
                bundles.append(bundle)
        
        return bundles
    
    def _optimize_bundle_sizes(self, bundles: list[Bundle]) -> list[Bundle]:
        """Optimize bundle sizes to meet constraints."""
        optimized = []
        
        for bundle in bundles:
            # If bundle is too large, split it
            if bundle.total_chars > self.max_bundle_chars:
                split_bundles = self._split_bundle(bundle)
                optimized.extend(split_bundles)
            # If bundle is too small, try to merge with next
            elif bundle.total_chars < self.min_bundle_chars:
                # For now, just keep it (merging logic could be added)
                optimized.append(bundle)
            else:
                optimized.append(bundle)
        
        return optimized
    
    def _split_bundle(self, bundle: Bundle) -> list[Bundle]:
        """Split a large bundle into smaller ones."""
        card_ids = bundle.card_ids
        num_splits = int(bundle.total_chars / self.target_chars) + 1
        chunk_size = len(card_ids) // num_splits
        
        splits = []
        for i in range(num_splits):
            start = i * chunk_size
            end = start + chunk_size if i < num_splits - 1 else len(card_ids)
            split_cards = card_ids[start:end]
            
            if split_cards:
                new_bundle = self._create_bundle(f"{bundle.id}_split{i}", split_cards)
                if new_bundle:
                    splits.append(new_bundle)
        
        return splits
    
    def _create_bundle(self, bundle_id: str, card_ids: list[str]) -> Bundle | None:
        """Create a bundle from card IDs."""
        valid_cards = [cid for cid in card_ids if cid in self.cards]
        if not valid_cards:
            return None
        
        # Get unique file paths
        file_paths = list(set(self.cards[cid].relpath for cid in valid_cards))
        
        # Calculate total size
        total_chars = sum(
            self.cards[cid].char_end - self.cards[cid].char_start 
            for cid in valid_cards
        )
        
        # Create preview
        preview = self._generate_preview(valid_cards, file_paths)
        
        return Bundle(
            id=bundle_id,
            card_ids=valid_cards,
            file_paths=sorted(file_paths),
            total_chars=total_chars,
            preview=preview
        )
    
    def _create_single_bundle(self, card_ids: list[str]) -> Bundle:
        """Create a single bundle containing all cards."""
        return self._create_bundle("bundle_000", card_ids)
    
    def _generate_preview(self, card_ids: list[str], file_paths: list[str]) -> str:
        """Generate a preview description of bundle contents."""
        # Simple preview: list main files
        if len(file_paths) == 1:
            return f"Single file: {file_paths[0]}"
        elif len(file_paths) <= 3:
            return f"Files: {', '.join(file_paths)}"
        else:
            # Show first 2 and count
            return f"Files: {file_paths[0]}, {file_paths[1]} (+{len(file_paths)-2} more)"
    
    def save_bundles(self, output_dir: Path) -> dict:
        """Save bundles to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        bundles = self.create_bundles()
        
        # Save bundles
        bundles_data = [b.to_dict() for b in bundles]
        with open(output_dir / "bundles.json", "w") as f:
            json.dump(bundles_data, f, indent=2)
        
        # Save summary
        summary = {
            "num_bundles": len(bundles),
            "total_chars": sum(b.total_chars for b in bundles),
            "avg_bundle_size": sum(b.total_chars for b in bundles) / len(bundles) if bundles else 0,
            "min_bundle_size": min(b.total_chars for b in bundles) if bundles else 0,
            "max_bundle_size": max(b.total_chars for b in bundles) if bundles else 0,
        }
        
        return summary