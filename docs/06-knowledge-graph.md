# Phase 6 — Knowledge Graph and Visualization

Deliverable 4: knowledge-graph (KG) structure and NVIDIA visualization
approach, with explicit mapping to the hierarchical legal concept taxonomy
from `00-overview/glossary.md`. Built on `cuGraph` for GPU graph
analytics, visualized with `cuxfilter` and `datashader`, presented in the UI
via a lightweight D3/Cytoscape renderer for interactivity.

## 1. Ontology

Nodes and edges follow the taxonomy. Every node and edge carries a stable
`kg_id` that is the storage primary key (Postgres UUID or composite key).

### Node types

Grouped by glossary kind. All `legal_type` artifacts are first-class KG
nodes (siblings), not nested under `case_file`.

**legal_type (siblings)**

| Type | Source table | Key attributes |
|---|---|---|
| `legal_situation` | `vila.legal_situations` | `situation_id`, `incident_date`, `location` |
| `case_file` | `vila.case_files` | `case_id`, `court_id`, `case_type`, `legal_relation`, `judgment_date`, `outcome` |
| `indictment` | `vila.indictments` | `indictment_id`, `indictment_number`, `issue_date`, `issuing_authority` |
| `lawsuit` | `vila.lawsuits` | `lawsuit_id`, `plaintiff_name`, `civil_defendant_name`, `relief_sought` |
| `investigation_conclusion` | `vila.investigation_conclusions` | `conclusion_id`, `issue_date`, `recommendation` |
| `ruling` | `vila.rulings` | `ruling_id`, `ruling_kind`, `issue_date` |
| `verdict` | `vila.verdicts` | `verdict_id`, `verdict_number`, `trial_level`, `pronounced_date`, `disposition` |
| `precedent` | `vila.precedents` | `precedent_id`, `precedent_number`, `adopted_date` |

**participant**

| Type | Source table | Key attributes |
|---|---|---|
| `court` | `vila.courts` | `court_id`, `court_level`, `province` |
| `procuracy` | dimension | `agency_id`, `name`, `level` |
| `investigation_body` | dimension | `agency_id`, `name`, `level` |
| `defendant` | `vila.defendants` | `defendant_id`, `person_id`, `age_determined`, `gender` |
| `person` | `vila.persons` | `person_id`, `full_name_hash` |

**legal_source**

| Type | Source table | Key attributes |
|---|---|---|
| `statute_article` | `vila.statute_articles` | `article_id`, `code_id`, `article_number`, `effective_from`/`to` |
| `code` | `vila.codes` | `code_id`, `short_name` |

**constituent_attribute (attach to one or more legal_type artifacts)**

| Type | Source table | Key attributes |
|---|---|---|
| `charge` | `vila.charges` | `charge_id`, `charge_name`, `severity_band` |
| `sentence` | `vila.sentences` | `sentence_id`, `penalty_type`, `sentence_term` |
| `evidence_item` | `vila.evidence_items` | `evidence_id`, `item_kind`, `item_value` |
| `case_event` | `vila.case_events` | `event_id`, `event_kind`, `event_ts` |
| `factor` | `vila.case_factors` | `factor_code`, `factor_kind` |

**classifiers**

| Type | Source table | Key attributes |
|---|---|---|
| `legal_relation` | dimension | `legal_relation` string |
| `procedure_type` | dimension | `procedure_type` string |
| `taxonomy_concept` | static from glossary | for example `tu_phap`, `legal_type`, `participant` |

### Edge types

Siblings under `legal_type` are connected by explicit relations (not
containment), matching the relations catalog in `00-overview/glossary.md`.

**legal_type sibling relations**

| From | Relation | To | Source |
|---|---|---|---|
| `legal_situation` | `may_spawn` | `case_file` | `situation_cases` |
| `case_file` | `appeal_of` | `case_file` | derived from case-code linkage + trial level |
| `case_file` | `initiated_by` | `lawsuit` | `lawsuits.case_id` |
| `case_file` | `indicted_by` | `indictment` | `indictments.case_id` |
| `indictment` | `preceded_by` | `investigation_conclusion` | `investigation_conclusions.case_id` + issue dates |
| `case_file` | `decided_by` | `verdict` | `verdicts.case_id` |
| `case_file` | `ordered_by` | `ruling` | `rulings.case_id` |
| `verdict` | `may_become` | `precedent` | `precedents.source_case_id` + verdict selection |
| `case_file` | `grounded_on` | `precedent` | retrieval-derived (above similarity threshold) |

**participant relations**

| From | Relation | To | Source |
|---|---|---|---|
| `case_file` | `tried_by` | `court` | `case_files.court_id` |
| `verdict` | `pronounced_by` | `court` | `verdicts.court_id` |
| `indictment` | `issued_by` | `procuracy` | `indictments.issuing_authority` |
| `investigation_conclusion` | `issued_by` | `investigation_body` | `investigation_conclusions.issuing_authority` |
| `case_file` | `has_defendant` | `defendant` | `defendants.case_id` |
| `defendant` | `is_person` | `person` | `defendants.person_id` |

**constituent_attribute attachment**

| From | Relation | To | Source |
|---|---|---|---|
| `indictment` / `verdict` | `has_charge` | `charge` | `charges.case_id` plus carrier inference |
| `charge` | `cites_article` | `statute_article` | `charge_articles` |
| `verdict` | `has_sentence` | `sentence` | `sentences.verdict_id` |
| `charge` | `sentenced_with` | `sentence` | `sentences.charge_id` |
| `case_file` | `has_evidence` | `evidence_item` | `evidence_items.case_id` |
| `case_file` | `has_event` | `case_event` | `case_events.case_id` |
| `case_file` | `has_factor` | `factor` | `case_factors` |
| `precedent` | `applies_article` | `statute_article` | `precedents.applied_article_id` |

**legal_source relations**

| From | Relation | To | Source |
|---|---|---|---|
| `statute_article` | `belongs_to_code` | `code` | `statute_articles.code_id` |
| `statute_article` | `amends` | `statute_article` | supersession table |

**classifier relations**

| From | Relation | To | Source |
|---|---|---|---|
| `case_file` | `classified_as` | `legal_relation` | `case_files.legal_relation` |
| `case_file` | `follows` | `procedure_type` | `case_files.procedure_type` |
| `taxonomy_concept` | `groups` | any node | static mapping (one of five groupings) |

Edges are typed and versioned with the effective date range when applicable
(for example `grounded_on` is scoped by the embedding model version used
for retrieval).

## 2. Build pipeline (cuGraph-based)

```
     Postgres             Milvus              MongoDB
         |                   |                   |
         v                   v                   v
  [ETL extract]       [k-NN retrieval]      [text slices]
         \                   |                   /
          \                  |                  /
           +-----------------+-----------------+
                             |
                             v
                +--------------------------+
                | build edge list (cudf)   |
                +-------------+------------+
                              |
                              v
                +--------------------------+
                | cugraph.Graph            |
                | (or cugraph.MultiGraph)  |
                +-------------+------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
         +----------+    +----------+    +----------+
         | PageRank |    | Louvain  |    | SSSP/BFS |
         +-----+----+    +-----+----+    +-----+----+
               |               |               |
               v               v               v
           node-prop        community       shortest
           scores           membership      paths
               |               |               |
               +---------------+---------------+
                               |
                               v
                +--------------------------+
                | cuxfilter dashboards     |
                | + API                    |
                +--------------------------+
```

### Build code (sketch)

```python
# services/kg/src/vila_kg/build.py
from __future__ import annotations
import cudf
import cugraph
from vila_kg.schema import NODE_TYPES, EDGE_TYPES
from vila_kg.extract import (
    load_nodes_from_postgres,
    load_edges_from_postgres,
    load_similar_case_edges_from_milvus,
)

def build_graph(similarity_threshold: float = 0.82) -> cugraph.Graph:
    """Assemble the ViLA knowledge graph on GPU."""
    nodes: cudf.DataFrame = load_nodes_from_postgres()
    edges: cudf.DataFrame = cudf.concat(
        [
            load_edges_from_postgres(),
            load_similar_case_edges_from_milvus(threshold=similarity_threshold),
        ],
        ignore_index=True,
    )
    g = cugraph.MultiGraph()
    g.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="weight")
    return g

def analytics(g: cugraph.Graph) -> cudf.DataFrame:
    """Run the analytics battery: PageRank, Louvain, degree centrality."""
    pr = cugraph.pagerank(g)
    louvain, _ = cugraph.louvain(g)
    degree = g.degree()
    return pr.merge(louvain, on="vertex").merge(degree, on="vertex")
```

PageRank over `statute_article` nodes gives a strong signal for "most
influential articles" within a legal relation, which is a top-level metric
on the statute-focused UI page.

### Incremental updates

Full rebuilds are nightly. For hot updates (user uploads a new case), a
diff ETL inserts just the new nodes/edges into a mutable overlay graph.
cuGraph's multi-GPU variants are available if the corpus exceeds a single
node's memory; initial deployment is single-GPU.

## 3. Query API

`services/kg/src/vila_kg/query.py` exposes HTTP endpoints used by the agent
and UI:

| Endpoint | Purpose |
|---|---|
| `GET /kg/case/{case_id}/neighborhood?hops=2` | 2-hop subgraph JSON |
| `GET /kg/charge/{charge_name}/articles` | statute articles for a charge |
| `GET /kg/article/{article_id}/cases?limit=50` | all cases citing an article |
| `GET /kg/precedent/{precedent_id}/applied-cases` | cases where precedent cited |
| `POST /kg/path` | shortest path between two nodes |
| `GET /kg/taxonomy` | the full static taxonomy tree |

Response is JSON-LD-like for easy UI consumption:

```json
{
  "nodes": [
    {"id": "case:uuid-1", "type": "case_file", "labels": {"vi": "Bản án 123/2024", "en": "Judgment 123/2024"}},
    {"id": "article:uuid-2", "type": "statute_article", "labels": {"vi": "Điều 173 BLHS 2015", "en": "Art. 173 BLHS 2015"}}
  ],
  "edges": [
    {"src": "case:uuid-1", "dst": "article:uuid-2", "type": "cites_article", "weight": 1.0}
  ]
}
```

Labels are bilingual at the API boundary; the UI does not translate on its
own for KG nodes.

## 4. Visualization

### 4.1 cuxfilter dashboard (analytical)

For power-user exploration we ship a `cuxfilter` dashboard with:

- Scatter: UMAP 2-d of case embeddings, colored by `legal_relation`, sized
  by degree centrality in the KG.
- Histogram: cases per year.
- Bar: top 20 charges.
- Choropleth: cases per province.
- Table: currently selected cross-filtered set.

Scaffold:

```python
# services/kg/src/vila_kg/cuxfilter_app.py
import cuxfilter
import cudf

def build_dashboard(df: cudf.DataFrame) -> cuxfilter.DashBoard:
    """Interactive dashboard over the case case-embedding UMAP."""
    cx_df = cuxfilter.DataFrame.from_dataframe(df)
    scatter = cuxfilter.charts.datashader.scatter(
        x="umap_x", y="umap_y", aggregate_col="case_count"
    )
    year_hist = cuxfilter.charts.bokeh.bar("judgment_year")
    relation_bar = cuxfilter.charts.bokeh.bar("legal_relation")
    charts = [scatter, year_hist, relation_bar]
    return cx_df.dashboard(charts, theme=cuxfilter.themes.light)
```

### 4.2 Case-level subgraph (UI)

The interactive node-link view the user sees on a case page is a small,
curated subgraph (<= 200 nodes) rendered client-side (React + Cytoscape.js
or D3). The server returns the full JSON; the client renders with force-
directed layout, hover tooltips, zoom, and click-to-expand.

Tooltip contents (bilingual):

- `case_file`: case code + court + outcome + year.
- `statute_article`: code + article number + short text.
- `precedent`: precedent number + adoption date + applied principle.
- `charge`: charge name + severity band.

### 4.3 Timeline visualization

Case timeline is rendered from `case_events` filtered to the case, sorted
by `event_ts`. D3-based vertical timeline with stacked event cards;
lanes for defendants when multiple are present. Hover surfaces the
`evidence_span` (source markdown chunk) so the user can jump to the
highlighted passage in the original document.

### 4.4 Legal-concept taxonomy view

A dedicated UI view renders the taxonomy from `00-overview/glossary.md`.
The top level has a fixed shape: `Pháp luật thông thường` -> `Tư pháp`
-> one of five groupings
(`legal_type`, `legal_relation`, `procedure_type`, `participant`,
`legal_source`). `legal_type` is a **sibling list**, not a chain:
`Tình huống`, `Vụ án`, `Cáo trạng`, `Đơn khởi kiện`,
`Kết luận điều tra`, `Quyết định`, `Bản án`, `Án lệ`. Each sibling shows
a count (for example how many cases carry the `Trộm cắp tài sản`
`legal_relation`, or how many `Cáo trạng` reference BLHS Article 173).

The view doubles as a filter: selecting a node applies a constraint
across other views. Constraints from sibling `legal_type` nodes compose
as the conjunction of their induced case sets, not as tree descent.

## 5. Using the KG in the agent (preview for Phase 8)

The agent's `predict_outcome` tool flow:

1. Retrieve 2-hop neighborhood of the new case in the KG, constrained by
   `case_type` and `legal_relation`.
2. Promote neighbors that share a `cites_article` relation with the query
   case or that are linked via `grounded_on` to a precedent the query
   cites.
3. Feed the subgraph + retrieved case markdown into the LLM prompt.
4. Constrain the LLM to cite only nodes in the provided subgraph (citation
   binding).

This closes the loop: the KG is not just a visualization; it is the
retrieval substrate for prediction.

## 6. Quality and coverage metrics

- **Node coverage**: fraction of `case_files` rows represented as KG
  nodes. Target 100%; alert below 99.8%.
- **Edge density** by type: charge->article edges should match the count
  in `charge_articles`. Drift flags a build bug.
- **Orphan nodes**: any node with degree 0 is inspected.
- **Cluster purity**: Louvain communities should align with
  `legal_relation`; normalized mutual information (NMI) tracked per build.
- **Build time**: build + analytics under 15 minutes on a single A100 for
  ~1.5 M cases. Regressions trigger a perf review.
