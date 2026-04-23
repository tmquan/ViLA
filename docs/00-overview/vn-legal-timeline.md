# Vietnamese Legal Timeline (life-span reference)

This is the canonical reference for the Vietnamese legal codes ViLA is
built against, their effective dates, supersession chains, and notable
amendments. It seeds the `vila.codes` and `vila.statute_articles` tables
(Phase 5) and defines the temporal semantics the curator, agent, and UI
use when resolving `effective_from` / `effective_to`.

Planning cut-off for this document: April 2026. Laws in force at the
cut-off are marked `IN FORCE`. Where a replacement law has been passed
but scheduled to take force later, both entries appear with overlapping
dates.

> Note on authority: this file is for ViLA's operational modelling. It
> is not a legal authority. The `vbpl.vn` / `vanban.chinhphu.vn`
> portals remain the primary source of truth, and the curator refreshes
> versions from them. If any date in this table conflicts with the
> primary source, the primary source wins and this file is updated.

## 1. In-force codes and laws (April 2026)

Each row is one `code_id`. The `status` column is derived for
convenience; the authoritative signal is `effective_from` /
`effective_to` (NULL means still in force).

| code_id | short_name | long_name | Law number | Promulgated | effective_from | effective_to | status | Notes |
|---|---|---|---|---|---|---|---|---|
| `HP-2013` | HP | Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam 2013 | — (Constitution) | 2013-11-28 | 2014-01-01 | NULL | IN FORCE | — |
| `BLHS-2015` | BLHS | Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017) | 100/2015/QH13; amendment 12/2017/QH14 | 2015-11-27 / 2017-06-20 | 2018-01-01 | NULL | IN FORCE | The 2015 Code was scheduled for 2016-07-01 but postponed; the 2017 amendment and the combined effective date is 2018-01-01 |
| `BLTTHS-2015` | BLTTHS | Bộ luật Tố tụng Hình sự 2015 | 101/2015/QH13 | 2015-11-27 | 2018-01-01 | NULL | IN FORCE | — |
| `BLDS-2015` | BLDS | Bộ luật Dân sự 2015 | 91/2015/QH13 | 2015-11-24 | 2017-01-01 | NULL | IN FORCE | — |
| `BLTTDS-2015` | BLTTDS | Bộ luật Tố tụng Dân sự 2015 | 92/2015/QH13 | 2015-11-25 | 2016-07-01 | NULL | IN FORCE | — |
| `LTTHC-2015` | LTTHC | Luật Tố tụng Hành chính 2015 | 93/2015/QH13 | 2015-11-25 | 2016-07-01 | NULL | IN FORCE | — |
| `BLLD-2019` | BLLĐ | Bộ luật Lao động 2019 | 45/2019/QH14 | 2019-11-20 | 2021-01-01 | NULL | IN FORCE | Replaced BLLĐ 2012 |
| `LHNGD-2014` | LHNGĐ | Luật Hôn nhân và Gia đình 2014 | 52/2014/QH13 | 2014-06-19 | 2015-01-01 | NULL | IN FORCE | — |
| `LTM-2005` | LTM | Luật Thương mại 2005 | 36/2005/QH11 | 2005-06-14 | 2006-01-01 | NULL | IN FORCE | — |
| `LDN-2020` | LDN | Luật Doanh nghiệp 2020 | 59/2020/QH14 | 2020-06-17 | 2021-01-01 | NULL | IN FORCE | — |
| `LTHAHS-2019` | LTHAHS | Luật Thi hành án hình sự 2019 | 41/2019/QH14 | 2019-06-14 | 2020-01-01 | NULL | IN FORCE | Replaced 2010 law |
| `LTHADS-2008` | LTHADS | Luật Thi hành án dân sự 2008 (sửa đổi 2014, 2022) | 26/2008/QH12 + 64/2014/QH13 + 03/2022/QH15 (partial) | — | 2009-07-01 | NULL | IN FORCE | Multiple amendment vintages |
| `LXLVPHC-2012` | LXLVPHC | Luật Xử lý vi phạm hành chính 2012 (sửa đổi 2020) | 15/2012/QH13; amendment 67/2020/QH14 | 2012-06-20 / 2020-11-13 | 2013-07-01 | NULL | IN FORCE | 2020 amendment effective 2022-01-01 for most provisions |
| `LTCTAND-2024` | LTCTAND | Luật Tổ chức Tòa án nhân dân 2024 | 34/2024/QH15 | 2024-06-24 | 2025-01-01 | NULL | IN FORCE | Replaced LTCTAND 2014 (62/2014/QH13) |
| `LTCVKSND-2014` | LTCVKSND | Luật Tổ chức Viện kiểm sát nhân dân 2014 | 63/2014/QH13 | 2014-11-24 | 2015-06-01 | NULL | IN FORCE | — |
| `LTPCTN-2024` | LTPCTN | Luật Tư pháp người chưa thành niên 2024 | Passed 2024-11 at 15th National Assembly, 8th session | 2024-11 | 2026-01-01 | NULL | IN FORCE | Unified juvenile-justice framework; directly changes juvenile subtree in Phase 7. See section 4. |
| `LGD-2009` | LGĐ | Luật Giám định tư pháp 2012 (sửa đổi 2020) | 13/2012/QH13; 56/2020/QH14 | 2012-06-20 / 2020-06-10 | 2013-01-01 | NULL | IN FORCE | Governs forensic examinations ViLA consumes via `determination.mental_health_assessment` |

## 2. Historical arcs of the Vietnamese legal system

ViLA's primary retrieval targets the **modern codification** (1985 →
present). Earlier legal layers are retained for academic, comparative,
and provenance purposes. Each arc below is a distinct regime with its
own primary sources, vocabulary, and assumptions about who is governed.

| Arc | Span | Summary | ViLA handling |
|---|---|---|---|
| A1 — Imperial (pre-modern) | pre-1858 | Quốc triều hình luật (Lê Code, 1483); Hoàng Việt luật lệ (Gia Long / Nguyễn Code, 1815) | Historical reference only; not in `vila.codes` by default |
| A2 — Colonial | 1858 – 1954 | French civil code applied in Cochinchina; mixed regimes in Tonkin and Annam; Franco-Vietnamese hybrid in urban courts | Historical reference only |
| A3 — Divided period | 1945 – 1975 | DRV (North): Constitutions 1946, 1959; early decrees. RVN (South): Constitutions 1956, 1967; retained French civil code; US-advised revisions | Historical reference; selected DRV decrees archived |
| A4 — Unification → Đổi Mới | 1975 – 1985 | Constitution 1980; administrative-law primacy; criminal law codified via stand-alone ordinances (pháp lệnh) rather than a unified code | Historical reference |
| A5 — First-generation modern codes | 1985 – 2000 | BLHS 1985 (first modern Criminal Code, eff. 1986-01-01); BLTTHS 1988; BLDS 1995; Constitution 1992 (Đổi Mới market-economy charter); Luật HN&GĐ 1959 → 1986 → 2000 | In `vila.codes` as repealed entries; queried for old incidents |
| A6 — Consolidation | 2000 – 2015 | BLHS 1999 (sửa đổi 2009); BLTTHS 2003; BLDS 2005; BLTTDS 2004; LXLVPHC 2012; Luật Thương mại 2005; Luật Thi hành án dân sự 2008 | In `vila.codes` as repealed-or-amended; primary for incidents 2000–2017 |
| A7 — Current codification | 2015 – 2024 | BLHS 2015 (sửa đổi 2017); BLTTHS 2015; BLDS 2015; BLTTDS 2015; LTTHC 2015; BLLĐ 2019; LTHAHS 2019 | `IN FORCE` (§1) |
| A8 — Post-2024 reforms | 2024 → | LTCTAND 2024 (eff. 2025-01-01); LTPCTN 2024 (eff. 2026-01-01); ongoing court-organization restructuring | `IN FORCE` (§1); juvenile regime change handled by `case_files.juvenile_regime` tag |

Constitutions across arcs (distinct `code_id` per document):

| code_id | Long name | effective_from | effective_to | Arc |
|---|---|---|---|---|
| `HP-1946` | Hiến pháp nước Việt Nam Dân chủ Cộng hòa 1946 | 1946-11-09 | 1958-12-31 | A3 (DRV) |
| `HP-1959` | Hiến pháp nước Việt Nam Dân chủ Cộng hòa 1959 | 1959-12-31 | 1980-12-17 | A3 (DRV) |
| `HP-1980` | Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam 1980 | 1980-12-18 | 1992-04-14 | A4 |
| `HP-1992` | Hiến pháp 1992 (sửa đổi 2001) | 1992-04-15 | 2013-12-31 | A5–A6 |
| `HP-2013` | Hiến pháp 2013 | 2014-01-01 | NULL | A7 (IN FORCE) |

Constitutions of the Republic of Vietnam (1956, 1967) are retained as
historical-only metadata for case records that may reference a party or
event from that jurisdiction; they are not used for statute linking in
modern cases.

## 2a. Superseded codes (retained for old-case queries)

Past-effective laws are retained in `vila.codes` so queries about a case
whose `incident_date` predates a reform resolve correctly. The scraper
populates the full history from `vbpl.vn` on first sync. The table
below lists every superseded code that ViLA's retrieval actively
targets for incidents in the 1986–present range.

| code_id | long_name | effective_from | effective_to | Superseded by | Arc |
|---|---|---|---|---|---|
| `BLHS-1985` | Bộ luật Hình sự 1985 | 1986-01-01 | 2000-06-30 | `BLHS-1999` | A5 |
| `BLHS-1999` | Bộ luật Hình sự 1999 (sửa đổi 2009) | 2000-07-01 | 2017-12-31 | `BLHS-2015` | A6 |
| `BLTTHS-1988` | Bộ luật Tố tụng Hình sự 1988 | 1989-01-01 | 2004-06-30 | `BLTTHS-2003` | A5 |
| `BLTTHS-2003` | Bộ luật Tố tụng Hình sự 2003 | 2004-07-01 | 2017-12-31 | `BLTTHS-2015` | A6 |
| `BLDS-1995` | Bộ luật Dân sự 1995 | 1996-07-01 | 2005-12-31 | `BLDS-2005` | A5 |
| `BLDS-2005` | Bộ luật Dân sự 2005 | 2006-01-01 | 2016-12-31 | `BLDS-2015` | A6 |
| `BLTTDS-2004` | Bộ luật Tố tụng Dân sự 2004 (sửa đổi 2011) | 2005-01-01 | 2016-06-30 | `BLTTDS-2015` | A6 |
| `LHNGD-1959` | Luật Hôn nhân và Gia đình 1959 | 1960-01-13 | 1986-12-31 | `LHNGD-1986` | A5 |
| `LHNGD-1986` | Luật Hôn nhân và Gia đình 1986 | 1987-01-03 | 2000-12-31 | `LHNGD-2000` | A5 |
| `LHNGD-2000` | Luật Hôn nhân và Gia đình 2000 | 2001-01-01 | 2014-12-31 | `LHNGD-2014` | A6 |
| `BLLD-1994` | Bộ luật Lao động 1994 (sửa đổi 2002, 2006, 2007) | 1995-01-01 | 2013-04-30 | `BLLD-2012` | A5–A6 |
| `BLLD-2012` | Bộ luật Lao động 2012 | 2013-05-01 | 2020-12-31 | `BLLD-2019` | A6 |
| `LTHADS-1993` | Pháp lệnh Thi hành án dân sự 1993 | 1993-06-21 | 2009-06-30 | `LTHADS-2008` | A5–A6 |
| `LTCTAND-1960` | Luật Tổ chức Tòa án nhân dân 1960 | 1960-07-14 | 1981-07-03 | `LTCTAND-1981` | A3–A4 |
| `LTCTAND-1981` | Luật Tổ chức Tòa án nhân dân 1981 | 1981-07-04 | 1992-10-06 | `LTCTAND-1992` | A4 |
| `LTCTAND-1992` | Luật Tổ chức Tòa án nhân dân 1992 | 1992-10-07 | 2002-10-01 | `LTCTAND-2002` | A5 |
| `LTCTAND-2002` | Luật Tổ chức Tòa án nhân dân 2002 | 2002-10-02 | 2015-05-31 | `LTCTAND-2014` | A6 |
| `LTCTAND-2014` | Luật Tổ chức Tòa án nhân dân 2014 | 2015-06-01 | 2024-12-31 | `LTCTAND-2024` | A7 |

Notes:
- Dates for A5 codes carry some uncertainty because the National
  Assembly gazette archive is less complete than for post-2000 laws;
  ingestion should treat A5 dates as authoritative when available and
  annotate uncertainty in `raw_documents.metadata` when not.
- Not every historical ordinance (pháp lệnh) is modelled as a separate
  `code_id`; ordinances on narrow subjects (for example the 1988
  Pháp lệnh thủ tục giải quyết các vụ án dân sự) are loaded into
  `vila.statute_articles` under the successor code with
  `effective_from`/`effective_to` reflecting the pháp lệnh era.
- Major RVN (1956 Constitution, 1967 Constitution, Civil Code of
  1972) references are represented only as `legal_situation` context
  for cases that cite them historically — not as `vila.codes` rows.

## 2b. Pre-modern references (A1–A2)

Retained as **documentary-only** metadata. ViLA does not link modern
cases to pre-modern codes. A separate `vila.historical_codes` table
(introduced in ontology v1.2.0 — see `ontology.md` §14) stores these
entries for research and for UI annotation when a verdict text cites
an imperial source by name:

| code_id | Long name | Origin | Notes |
|---|---|---|---|
| `QTHL-1483` | Quốc triều hình luật (Lê Code) | Hồng Đức reign, Lê dynasty | ~722 articles; mixed criminal / civil / administrative |
| `HVLL-1815` | Hoàng Việt luật lệ (Gia Long Code) | Nguyễn dynasty | Modelled on Qing Code with Vietnamese modifications |
| `FR-CODE` | French Civil Code (Bộ luật Dân sự Pháp) | French colonial administration | Applied unevenly across Cochinchina / Tonkin / Annam |

These never appear in statute-linking results for modern cases; they
are reachable via the UI's taxonomy browser under a "Lịch sử tư pháp"
(judicial history) node.

## 3. Temporal resolution rules

The rules below are executed by `packages/nlp/statute_linker.py` at
extract time and by the agent at retrieval time. They are the basis of
the `statute_article.effective_from` / `effective_to` semantics in
Phase 5.

1. **Case-date anchoring.** For any statute citation in a case, the
   effective article version is the one whose
   `(effective_from, effective_to]` window contains
   `case_files.incident_date`. If `incident_date` is missing, fall back
   to `acceptance_date`.
2. **Transitional provisions.** When a reform includes transitional
   provisions (for example BLHS 2015 → 2017 amendment), the linker
   loads a small rules table `packages/nlp/data/transitional_rules.yaml`
   and applies the rule that yields the more favorable result to the
   accused (nguyên tắc có lợi cho người phạm tội) where Vietnamese law
   requires it.
3. **Amendment chain.** A single conceptual article may exist across
   multiple code versions. Chains are represented in Postgres as
   `statute_articles.replaces_id` self-references. The FRBR framing:
   each chain is a `Work`; each row in `statute_articles` is an
   `Expression`; each PDF on `vbpl.vn` is a `Manifestation`.
4. **Refresh cadence.** `VbplDownloader` (Phase 3) runs an on-change
   poll with an RSS / diff poll; any detected text change creates a new
   `Expression` row with `effective_from` set to the date in the
   official gazette.
5. **Multi-code statutes.** When a concept spans multiple codes (for
   example the interaction of BLHS Articles 51/52 with BLTTHS
   procedural provisions), the agent retrieves both and presents them
   side by side; there is no merge step.
6. **Regime boundary guard.** The linker refuses to cite a law from a
   different legal arc (§2) than the case's arc. A 1990 incident
   cannot cite BLHS 2015, and a 2020 incident cannot cite BLHS 1985.
   Arc boundaries are 2000-07-01 (A5→A6), 2017-12-31 (A6→A7), and
   2026-01-01 for the juvenile-justice regime.
7. **Pre-1986 incidents.** For `incident_date < 1986-01-01`, the
   linker returns `insufficient_coverage` with a flag: historical
   records are for documentary context only, not statute resolution.
   The UI surfaces "Vụ án thuộc giai đoạn pháp luật trước Bộ luật
   Hình sự 1985; không có cơ sở pháp lý hiện đại áp dụng" in VI, and
   the English equivalent.
8. **Constitutional anchoring.** Every statute article carries an
   implicit reference to the constitution in force on its
   `effective_from`. The UI and agent can traverse
   `statute_article → code → constitutional arc`.

## 4. Juvenile-justice regime change (effective 2026-01-01)

The `Luật Tư pháp người chưa thành niên 2024` (code `LTPCTN-2024`)
unifies the previously scattered juvenile-justice provisions (Chapter
XXVIII of BLTTHS 2015, parts of BLHS 2015, administrative measures
under LXLVPHC 2012, education-at-commune provisions). ViLA's juvenile
subtree (Phase 7 §6 and Phase 8 §8.1) must handle both regimes:

- Cases with `incident_date` **before 2026-01-01** resolve under the
  BLTTHS 2015 Chapter XXVIII + BLHS 2015 age-band framework.
- Cases with `incident_date` **on or after 2026-01-01** resolve under
  `LTPCTN-2024`, subject to "favorable-to-accused" transitional rules
  that may pull pre-2026 cases forward where applicable.

The decision tree (`services/agent/src/vila_agent/decision_tree.yaml`)
branches in `D5` / juvenile subtree `J1` on `incident_date` to select
the correct regime. The extractor tags each juvenile-case record with
`juvenile_regime = 'pre-2026' | 'ltpctn-2024'` so analytics can
disaggregate.

## 5. Court structure (after LTCTAND 2024)

`LTCTAND-2024` restructures aspects of the court system. The
`vila.courts` dimension stores the current structure and seeds from
official `toaan.gov.vn` lookups.

| Court level | Vietnamese name | Notes |
|---|---|---|
| Apex | Tòa án nhân dân tối cao (TANDTC) | Cassation / retrial (giám đốc thẩm, tái thẩm) |
| High | Tòa án nhân dân cấp cao | Appellate review; three high courts (Hanoi, Da Nang, HCMC) |
| Provincial | Tòa án nhân dân tỉnh / thành phố trực thuộc trung ương | First-instance for serious matters; appellate over district |
| District | Tòa án nhân dân huyện / quận / thị xã / thành phố thuộc tỉnh | First-instance for most matters |
| Specialized | Tòa án quân sự các cấp | Military courts |

The 2024 law also authorizes creation of specialized divisions within
existing courts. Where the 2024 restructuring is still being
implemented administratively, the `courts` table carries an
`active_from` / `active_to` so the agent correctly dates references in
historical case text (a "cấp tỉnh" court today may be restructured
tomorrow; historical verdicts stay attributable to the court that
decided them).

## 6. Prosecutorial and investigation authorities

Separate from courts; these are the `procuracy` and
`investigation_body` participant node types in Phase 6.

| Agency | Vietnamese name | Authoritative law |
|---|---|---|
| Viện kiểm sát nhân dân tối cao (VKSNDTC) | Supreme People's Procuracy | LTCVKSND-2014 |
| VKSND cấp cao | High Procuracy | LTCVKSND-2014 |
| VKSND tỉnh / thành phố | Provincial Procuracy | LTCVKSND-2014 |
| VKSND huyện / quận | District Procuracy | LTCVKSND-2014 |
| VKS quân sự các cấp | Military Procuracy | LTCVKSND-2014 |
| Cơ quan Cảnh sát điều tra | Police Investigation body | BLTTHS-2015; Luật Tổ chức Cơ quan điều tra hình sự 2015 |
| Cơ quan An ninh điều tra | Security Investigation body | BLTTHS-2015; Luật TCCQĐT 2015 |
| Cơ quan điều tra VKSND | VKS Investigation body | BLTTHS-2015; LTCVKSND-2014 |
| Cơ quan điều tra Quân đội | Military Investigation | BLTTHS-2015 |

## 7. Seed data for `vila.codes` (Phase 5 reference)

The SQL below is the minimal seed to load on a fresh Postgres. Full
article-level seed data (`vila.statute_articles` rows) comes from the
`VbplDownloader` + extractor. Dates are in `effective_from`
descending, grouped by family.

```sql
-- ========================================================================
-- vila.codes seed
-- Every row is either IN FORCE (repealed_date IS NULL) or repealed with a
-- successor. Ordered: current codes first, then A6 (2000-2015), then A5
-- (1985-2000), then pre-A5 where relevant.
-- ========================================================================

-- A7/A8 (IN FORCE)  — Criminal
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('BLHS-2015',  'BLHS',  'Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017)',              '2015-11-27', NULL),
  ('BLTTHS-2015','BLTTHS','Bộ luật Tố tụng Hình sự 2015',                              '2015-11-27', NULL);

-- A6 — Criminal (repealed)
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('BLHS-1999',  'BLHS',  'Bộ luật Hình sự 1999 (sửa đổi 2009)',                       '1999-12-21', '2017-12-31'),
  ('BLTTHS-2003','BLTTHS','Bộ luật Tố tụng Hình sự 2003',                              '2003-11-26', '2017-12-31');

-- A5 — Criminal (first-generation modern codes)
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('BLHS-1985',  'BLHS',  'Bộ luật Hình sự 1985',                                      '1985-06-27', '2000-06-30'),
  ('BLTTHS-1988','BLTTHS','Bộ luật Tố tụng Hình sự 1988',                              '1988-06-28', '2004-06-30');

-- A7 — Civil family (IN FORCE)
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('BLDS-2015',  'BLDS',  'Bộ luật Dân sự 2015',                                       '2015-11-24', NULL),
  ('BLTTDS-2015','BLTTDS','Bộ luật Tố tụng Dân sự 2015',                               '2015-11-25', NULL),
  ('LHNGD-2014', 'LHNGĐ', 'Luật Hôn nhân và Gia đình 2014',                            '2014-06-19', NULL);

-- A6 — Civil family (repealed)
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('BLDS-2005',  'BLDS',  'Bộ luật Dân sự 2005',                                       '2005-06-14', '2016-12-31'),
  ('BLTTDS-2004','BLTTDS','Bộ luật Tố tụng Dân sự 2004 (sửa đổi 2011)',                '2004-06-15', '2016-06-30'),
  ('LHNGD-2000', 'LHNGĐ', 'Luật Hôn nhân và Gia đình 2000',                            '2000-06-09', '2014-12-31');

-- A5 — Civil family
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('BLDS-1995',  'BLDS',  'Bộ luật Dân sự 1995',                                       '1995-10-28', '2005-12-31'),
  ('LHNGD-1986', 'LHNGĐ', 'Luật Hôn nhân và Gia đình 1986',                            '1986-12-29', '2000-12-31'),
  ('LHNGD-1959', 'LHNGĐ', 'Luật Hôn nhân và Gia đình 1959',                            '1959-12-29', '1986-12-31');

-- Administrative
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('LTTHC-2015', 'LTTHC', 'Luật Tố tụng Hành chính 2015',                              '2015-11-25', NULL),
  ('LXLVPHC-2012','LXLVPHC','Luật Xử lý vi phạm hành chính 2012 (sửa đổi 2020)',        '2012-06-20', NULL);

-- Labor
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('BLLD-2019',  'BLLĐ',  'Bộ luật Lao động 2019',                                     '2019-11-20', NULL),
  ('BLLD-2012',  'BLLĐ',  'Bộ luật Lao động 2012',                                     '2012-06-18', '2020-12-31'),
  ('BLLD-1994',  'BLLĐ',  'Bộ luật Lao động 1994 (sửa đổi 2002, 2006, 2007)',          '1994-06-23', '2013-04-30');

-- Commerce
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('LTM-2005',   'LTM',   'Luật Thương mại 2005',                                      '2005-06-14', NULL),
  ('LDN-2020',   'LDN',   'Luật Doanh nghiệp 2020',                                    '2020-06-17', NULL);

-- Execution of judgments
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('LTHAHS-2019','LTHAHS','Luật Thi hành án hình sự 2019',                             '2019-06-14', NULL),
  ('LTHADS-2008','LTHADS','Luật Thi hành án dân sự 2008 (sửa đổi 2014, 2022)',         '2008-11-14', NULL),
  ('LTHADS-1993','LTHADS','Pháp lệnh Thi hành án dân sự 1993',                         '1993-01-21', '2009-06-30');

-- Organization of authorities
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('LTCTAND-2024','LTCTAND','Luật Tổ chức Tòa án nhân dân 2024',                       '2024-06-24', NULL),
  ('LTCVKSND-2014','LTCVKSND','Luật Tổ chức Viện kiểm sát nhân dân 2014',              '2014-11-24', NULL),
  ('LTCTAND-2014','LTCTAND','Luật Tổ chức Tòa án nhân dân 2014',                       '2014-11-24', '2024-12-31'),
  ('LTCTAND-2002','LTCTAND','Luật Tổ chức Tòa án nhân dân 2002',                       '2002-04-02', '2015-05-31'),
  ('LTCTAND-1992','LTCTAND','Luật Tổ chức Tòa án nhân dân 1992',                       '1992-10-06', '2002-10-01'),
  ('LTCTAND-1981','LTCTAND','Luật Tổ chức Tòa án nhân dân 1981',                       '1981-07-03', '1992-10-06'),
  ('LTCTAND-1960','LTCTAND','Luật Tổ chức Tòa án nhân dân 1960',                       '1960-07-14', '1981-07-03');

-- Juvenile justice
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('LTPCTN-2024','LTPCTN','Luật Tư pháp người chưa thành niên 2024',                   '2024-11-30', NULL);

-- Forensic examinations
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('LGD-2012',    'LGĐ',   'Luật Giám định tư pháp 2012 (sửa đổi 2020)',               '2012-06-20', NULL);

-- Constitutions (full arc)
INSERT INTO vila.codes (code_id, short_name, long_name, enacted_date, repealed_date) VALUES
  ('HP-2013',     'HP',    'Hiến pháp 2013',                                           '2013-11-28', NULL),
  ('HP-1992',     'HP',    'Hiến pháp 1992 (sửa đổi 2001)',                            '1992-04-15', '2013-12-31'),
  ('HP-1980',     'HP',    'Hiến pháp 1980',                                           '1980-12-18', '1992-04-14'),
  ('HP-1959',     'HP',    'Hiến pháp VNDCCH 1959',                                    '1959-12-31', '1980-12-17'),
  ('HP-1946',     'HP',    'Hiến pháp VNDCCH 1946',                                    '1946-11-09', '1958-12-31');
```

### 7a. Seed data for `vila.historical_codes` (documentary-only)

```sql
-- Pre-modern references. Never used for statute resolution on modern cases.
CREATE TABLE IF NOT EXISTS vila.historical_codes (
  code_id      text PRIMARY KEY,
  short_name   text NOT NULL,
  long_name    text NOT NULL,
  era          text NOT NULL,           -- 'imperial' / 'colonial' / 'rvn'
  approximate_start_year int,
  approximate_end_year   int,
  notes        text
);

INSERT INTO vila.historical_codes (code_id, short_name, long_name, era, approximate_start_year, approximate_end_year, notes) VALUES
  ('QTHL-1483', 'QTHL',  'Quốc triều hình luật (Lê Code)',                 'imperial', 1483, 1802, '~722 articles; criminal/civil/administrative mix'),
  ('HVLL-1815', 'HVLL',  'Hoàng Việt luật lệ (Gia Long / Nguyễn Code)',    'imperial', 1815, 1884, 'Modelled on Qing Code with Vietnamese modifications'),
  ('FR-CODE',   'FR-CC', 'Code civil français (colonial application)',     'colonial', 1858, 1954, 'Applied unevenly across Cochinchina / Tonkin / Annam'),
  ('RVN-HP-1956','HP-RVN', 'Hiến pháp Việt Nam Cộng hòa 1956',              'rvn',      1956, 1967, NULL),
  ('RVN-HP-1967','HP-RVN', 'Hiến pháp Việt Nam Cộng hòa 1967',              'rvn',      1967, 1975, NULL);
```

## 8. Identifier patterns that depend on this timeline

- **Statute URI (ELI-shaped)**:
  `eli:vn:law:<code_id>:article:<n>[:clause:<k>[:point:<p>]]`
  Example: `eli:vn:law:BLHS-2015:article:173:clause:1`.
- **Case URI (ECLI-shaped)**:
  `ECLI:VN:<court_code>:<year>:<ordinal>`.
  Example: `ECLI:VN:TAND-HN:2024:001234`.
- **Precedent URI**:
  `eli:vn:precedent:<precedent_number>` (for example
  `eli:vn:precedent:AL-47-2021`).

The court codes used in ECLI-VN are derived from `vila.courts` via a
stable slug (`court_code = slugify(court_name + '-' + province)`).
