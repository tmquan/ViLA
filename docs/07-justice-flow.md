# Phase 7 — Criminal Justice Flow and Decision Architecture

ViLA organizes every case's lifecycle around a **five-phase analytical
frame**: Entry into the system, Prosecution and pretrial services,
Adjudication, Sentencing and sanctions, and Corrections. This frame
generalizes the procedural shape of most modern criminal-justice systems
and makes the agent's decision tree (Phase 8) portable across
jurisdictions. This document maps the frame onto Vietnamese criminal and
non-criminal procedure and specifies the hierarchical decision tree used
for prediction.

## 1. The five-phase frame

| Phase | Scope | Vietnamese primary authorities |
|---|---|---|
| Entry into the system | Tiếp nhận tố giác, tin báo về tội phạm, kiến nghị khởi tố; điều tra; tạm giữ | Cơ quan điều tra (Công an điều tra; VKS in narrow cases); Bộ đội biên phòng; Hải quan (narrow) |
| Prosecution and pretrial services | Khởi tố bị can; áp dụng biện pháp ngăn chặn (tạm giữ, tạm giam, bảo lĩnh, cấm đi khỏi nơi cư trú, đặt tiền bảo đảm); truy tố; trả hồ sơ điều tra bổ sung | Viện kiểm sát (VKS). Tòa án has related authority at the pre-trial preparation stage. |
| Adjudication | Xét xử sơ thẩm; phúc thẩm; giám đốc thẩm; tái thẩm; các phán quyết tuyên không phạm tội, miễn trách nhiệm hình sự, miễn hình phạt | Tòa án nhân dân (TAND) — Hội đồng xét xử (HĐXX) |
| Sentencing and sanctions | Quyết định hình phạt: cảnh cáo, phạt tiền, cải tạo không giam giữ, tù có thời hạn, án treo, tù chung thân, tử hình, trục xuất; hình phạt bổ sung; biện pháp tư pháp | Tòa án nhân dân |
| Corrections | Thi hành án hình sự; giảm án; tha tù trước thời hạn có điều kiện; đặc xá; hết thời hạn chấp hành án | Cơ quan thi hành án hình sự / trại giam; Chủ tịch nước (đặc xá); Tòa án (giảm án, tha tù trước thời hạn) |

The frame also makes visible the **side exits** where cases leave the
main path (dismissal, diversion, acquittal, out-of-system after sentence,
etc.). Exits are enumerated in section 4 and mapped to
`case_files.outcome` / `diversion_reason` in the Phase 5 schema.

## 2. Vietnam-specific procedural vocabulary

Some procedural constructs that the generic frame assumes have
Vietnam-specific shapes or no equivalent at all. These shape the
decision tree and the data model.

| Procedural concept | Vietnamese form | Notes |
|---|---|---|
| Arrest | Bắt người / tạm giữ | Vietnam distinguishes bắt khẩn cấp, bắt quả tang, bắt theo lệnh, bắt người bị truy nã |
| Charge filing | Khởi tố vụ án / khởi tố bị can | Two-step: open the matter, then charge the person |
| Charge drop (pre-trial) | Đình chỉ điều tra / đình chỉ vụ án | May occur at điều tra or truy tố stage |
| Initial pre-trial hearing | Lấy lời khai ban đầu / quyết định tạm giữ, tạm giam | 24/48-hour decisions under BLTTHS |
| Preliminary hearing | No direct equivalent | Kết luận điều tra + cáo trạng cover the charging function together |
| Grand jury | Not present | Vietnam uses VKS alone; no lay grand jury |
| Indictment instrument | Cáo trạng | Prosecutor-authored by VKS |
| Information (prosecutor-filed charging doc) | Cáo trạng | Same instrument covers both roles |
| Refusal to indict | Đình chỉ vụ án của VKS | VKS declines to indict |
| Bail / detention hearing | Quyết định áp dụng biện pháp ngăn chặn / thay thế | Bail (bảo lĩnh, đặt tiền bảo đảm) exists but is used less than in common-law systems |
| Arraignment | Phiên tòa sơ thẩm — thủ tục bắt đầu phiên tòa, công bố cáo trạng | Vietnamese law combines arraignment + trial opening |
| Trial panel | HĐXX gồm thẩm phán và hội thẩm nhân dân | No lay jury; hội thẩm nhân dân are lay assessors on the panel |
| Guilty plea | Bị cáo nhận tội | Confession shortens proceedings but does not remove the trial |
| Conviction / acquittal | Tuyên có tội / tuyên không phạm tội | |
| Reduction of charge | Thay đổi tội danh trong quá trình xét xử | Allowed when evidence changes the charge |
| Probation-like non-custodial | Án treo; cải tạo không giam giữ | Not identical to common-law probation: án treo is a suspended sentence with supervision |
| Jail vs prison | No sentence-level distinction; corrections institution is trại giam | Pre-trial detention in nhà tạm giữ / nhà tạm giam |
| Intermediate sanctions | Cải tạo không giam giữ; phạt tiền; cảnh cáo; biện pháp tư pháp | |
| Death penalty | Tử hình | Permitted under BLHS for specified offenses; mandatory senior review |
| Parole | Tha tù trước thời hạn có điều kiện | |
| Revocation | Buộc thi hành phần còn lại của hình phạt | Applied when conditional release is breached |
| Pardon / clemency | Đặc xá; ân giảm án tử hình | Presidential authority |
| Habeas corpus | No direct equivalent | Kháng cáo, giám đốc thẩm, tái thẩm are the corrective mechanisms |
| Appeal | Kháng cáo (đương sự); kháng nghị (VKS / higher court) | |
| Automatic appeal (death) | Bản án tử hình phải trình Chánh án TANDTC và Viện trưởng VKSNDTC xem xét | Effectively automatic review |
| Juvenile diversion | Xử lý chuyển hướng (người dưới 18 tuổi) | Mostly administrative or educational measures |
| Juvenile court | No separate juvenile court | TAND applies special procedure in BLTTHS Chapter XXVIII |
| Waiver to adult court | Truy cứu trách nhiệm hình sự over age threshold | Age thresholds in BLHS |
| Aftercare | Giám sát sau khi chấp hành xong án phạt | Plus đăng ký tư pháp / xóa án tích |
| Out of system | Chấp hành xong bản án và xóa án tích | |

Non-criminal case types (dân sự, hôn nhân - gia đình, lao động, kinh
doanh - thương mại, hành chính) follow a parallel shape; see section 7.

## 3. Vietnamese criminal case lifecycle (full procedural detail)

The diagram below arranges the Vietnamese procedural events into the
five phases as columns. Exit / diversion points are marked with
`[E#.exit.*]`, `[P#.exit.*]`, etc. Node labels in square brackets are
used as `decision_path` identifiers (see section 8).

```
  [ ENTRY ]          [ PROSECUTION & PRETRIAL ]     [ ADJUDICATION ]           [ SENTENCING ]       [ CORRECTIONS ]

  Tin báo / tố giác
      |
      v
  [E1] Tiếp nhận, kiểm tra tin báo
      |
      +- [E1.exit.no-crime]  Không có tội phạm                ---------------------> out-of-system
      +- [E1.exit.no-charge] Không khởi tố vụ án              ---------------------> out-of-system
      |
      +- [E2] Khởi tố vụ án
                |
                v
           [E3] Điều tra
                |
                +- biện pháp ngăn chặn
                |     tạm giữ / tạm giam /
                |     bảo lĩnh / cấm đi khỏi nơi cư trú /
                |     đặt tiền bảo đảm
                |
                +- xác định tuổi; giám định sức khỏe tâm thần
                |
                +- kết luận điều tra
                      |
                      +- [E3.exit.dismiss]  Đình chỉ điều tra       -----> out-of-system
                      +- [E3.exit.suspend]  Tạm đình chỉ            -----> HOLD
                      +- [E4] Đề nghị truy tố
                              |
                              v
                         [P1] VKS thụ lý
                              |
                              +- [P1.exit.return]  Trả hồ sơ điều tra bổ sung -- back to [E3]
                              +- [P1.exit.dismiss] Đình chỉ vụ án            -----> out-of-system
                              +- [P2] Ban hành Cáo trạng, truy tố
                                    |
                                    v
                               [A1] Tòa án thụ lý, chuẩn bị xét xử
                                    |
                                    +- [A1.exit.return]  Trả hồ sơ điều tra bổ sung -- back to [E3]
                                    +- [A1.exit.dismiss] Đình chỉ / tạm đình chỉ   -----> out-of-system / HOLD
                                    +- [A2] Quyết định đưa vụ án ra xét xử
                                          |
                                          v
                                     [A3] Xét xử sơ thẩm
                                          |
                                          +- [A3.plea]    Bị cáo nhận tội                (shortens but does not replace trial)
                                          +- [A3.reduce]  Thay đổi tội danh              (evidence-driven charge change)
                                          +- [A3.acquit]  Tuyên không phạm tội      -----> out-of-system
                                          +- [A3.tnhs]    Miễn trách nhiệm hình sự  -----> out-of-system
                                          +- [A3.mien]    Miễn hình phạt            -----> out-of-system-lite (conviction, no penalty)
                                          +- [A3.convict] Tuyên có tội + quyết định hình phạt
                                                |
                                                v
                                           [S1] Mức hình phạt
                                                |
                                                +- [S1.canhcao]   Cảnh cáo
                                                +- [S1.phattien]  Phạt tiền
                                                +- [S1.caitao]    Cải tạo không giam giữ
                                                +- [S1.treo]      Tù có thời hạn kèm án treo
                                                +- [S1.tu]        Tù có thời hạn
                                                +- [S1.chungthan] Tù chung thân
                                                +- [S1.tuhinh]    Tử hình                  (mandatory senior review)
                                                +- [S1.tructuat]  Trục xuất
                                                +- [S1.bosung]    Hình phạt bổ sung         (cấm đảm nhiệm chức vụ, ...)
                                                +- [S1.tuphap]    Biện pháp tư pháp         (tịch thu, bồi thường, ...)
                                                |
                                                v
                                           [A4] Kháng cáo / kháng nghị?
                                                |
                                                +- không -> bản án có hiệu lực  -----> [C1]
                                                +- có
                                                      |
                                                      v
                                                 [A5] Phúc thẩm
                                                      |
                                                      +- giữ nguyên / sửa / hủy
                                                      +- [A6] Giám đốc thẩm / tái thẩm (exceptional review)
                                                      |
                                                      v
                                                     [C1]
                               [C1] Thi hành án
                                    |
                                    +- [C1.tu]      Thi hành án phạt tù (trại giam)
                                    +- [C1.treo]    Giám sát thi hành án treo
                                    +- [C1.tien]    Thi hành án phạt tiền
                                    +- [C1.tuhinh]  Thi hành án tử hình (sau khi ân giảm xem xét)
                                    +- [C1.giam]    Xét giảm án
                                    +- [C1.tha]     Tha tù trước thời hạn có điều kiện
                                    +- [C1.dacxa]   Đặc xá
                                    +- [C1.revoke]  Buộc thi hành phần hình phạt còn lại
                                    +- [C1.ht]      Chấp hành xong; xóa án tích
```

## 4. Exit and diversion catalog

Mapped to `case_files.outcome` and `diversion_reason` in the schema
(Phase 5). Each row is a place a case may leave the main path.

| Exit code | Vietnamese | Description | Outcome label |
|---|---|---|---|
| EX-01 | Không khởi tố vụ án | No case opened | `dismissed` |
| EX-02 | Đình chỉ điều tra | Charges dropped in investigation | `dismissed` |
| EX-03 | Đình chỉ truy tố (VKS) | VKS declines to indict / drops | `dismissed` |
| EX-04 | Đình chỉ vụ án (Tòa án) | Charge dismissed at adjudication | `dismissed` |
| EX-05 | Tuyên không phạm tội | Acquitted | `acquitted` |
| EX-06 | Miễn trách nhiệm hình sự | Diversion out of system | `dismissed` |
| EX-07 | Miễn hình phạt | Convicted, no penalty | `convicted` (no penalty) |
| EX-08 | Thỏa thuận / hòa giải (dân sự) | Settlement (non-criminal) | `settled` |
| EX-09 | Trả hồ sơ điều tra bổ sung | Procedural remand | `remanded` |
| EX-10 | Xử lý chuyển hướng (người dưới 18 tuổi) | Juvenile diversion | `dismissed` |
| EX-11 | Tạm đình chỉ điều tra / vụ án | Temporary hold | `remanded` |

Every `case_file` with a terminal outcome carries exactly one exit code.
`EX-09` and `EX-11` are non-terminal holds; the case later resolves with
a different code.

## 5. Discretionary authority points

Each phase surfaces one or more authorities who exercise discretion.
These are the pivots ViLA models as predictable variance at the
**agency or court/chamber level**, never at the individual officer or
judge level.

| Actor | Discretionary decisions |
|---|---|
| Cơ quan điều tra (Công an điều tra) | Khởi tố vụ án, khởi tố bị can; khám xét (theo thủ tục BLTTHS); bắt tạm giữ |
| Viện kiểm sát (VKS) | Phê chuẩn khởi tố; ban hành cáo trạng; đình chỉ / tạm đình chỉ; thay đổi tội danh; kháng nghị |
| Thẩm phán / Hội đồng xét xử | Áp dụng / thay đổi biện pháp ngăn chặn; chấp nhận nhận tội; đình chỉ; quyết định hình phạt; hủy án treo khi vi phạm |
| Cơ quan thi hành án hình sự / trại giam | Phân loại quản lý, chế độ giam giữ; khen thưởng, kỷ luật phạm nhân; đề nghị giảm án / tha tù trước thời hạn |
| Tòa án (thi hành án) + Chủ tịch nước (đặc xá) | Quyết định giảm án, tha tù trước thời hạn có điều kiện, đặc xá; xem xét hủy bỏ khi vi phạm |

ViLA only models this variance aggregated at the agency or court/chamber
level, following Phase 1 (section 14) guidance.

## 6. Parallel track: juvenile offenders

Vietnam operates two juvenile-justice regimes ViLA must handle in
parallel, selected by the case's `incident_date` (see
`00-overview/vn-legal-timeline.md` §4 and ontology axiom AX-17).

### 6.1 Pre-2026 regime (`incident_date < 2026-01-01`)

- No separate juvenile court system; BLTTHS 2015 Chapter XXVIII
  establishes a **special procedure for defendants under 18** inside
  the general courts.
- Age bands from BLHS 2015 (Article 12): under 14 is outside criminal
  responsibility; 14-16 is liable only for specific serious or very
  serious crimes; 16-18 is liable with mitigation (BLHS 2015 Part Four
  juvenile sentencing caps).
- Diversion measures: giáo dục tại xã, phường, thị trấn; hòa giải tại
  cộng đồng; biện pháp giáo dục tại trường giáo dưỡng — applied case-
  by-case by authorities under BLTTHS / BLHS and administrative law
  provisions.
- Residential placement analog: trường giáo dưỡng (administrative) or
  cơ sở giáo dục bắt buộc.
- There is no "tried as an adult" waiver; once over the applicable age
  threshold for the charge, regular criminal procedure applies with
  BLTTHS Chapter XXVIII modifications.

### 6.2 LTPCTN-2024 regime (`incident_date >= 2026-01-01`)

- `Luật Tư pháp người chưa thành niên 2024` (code_id `LTPCTN-2024`)
  unifies previously-scattered provisions (parts of BLTTHS 2015
  Chapter XXVIII, BLHS 2015 juvenile provisions, administrative-measure
  provisions under LXLVPHC 2012) into one framework. Effective
  2026-01-01.
- Xử lý chuyển hướng is formalized as the first-line response for
  eligible offenders, with a statutory list of measures and stricter
  criteria for custodial outcomes.
- Age bands and criminal responsibility thresholds remain tied to BLHS
  2015 Article 12; procedural handling is governed by LTPCTN-2024 in
  addition to BLTTHS 2015.
- Residential placement: trường giáo dưỡng and cơ sở giáo dục bắt
  buộc remain; LTPCTN-2024 tightens eligibility and review cadence.
- Transitional / favorable-to-accused rules may pull some pre-2026
  cases into the LTPCTN-2024 regime where it produces a more favorable
  outcome; the statute_linker (Phase 3) applies the Vietnamese
  "nguyên tắc có lợi cho người phạm tội" rule.

### 6.3 Regime selection in the decision tree

The agent selects the regime at `D5` / `J1`:

1. Read `Defendant.age_determined` and `CaseFile.incident_date`.
2. If `age_determined >= 18` -> exit juvenile subtree.
3. If `age_determined < 18`:
   - If `incident_date >= 2026-01-01` -> tag
     `juvenile_regime = 'ltpctn-2024'`.
   - Else -> tag `juvenile_regime = 'pre-2026'`.
4. If a transitional favorable-to-accused rule applies, upgrade a
   pre-2026 case to `ltpctn-2024` (recorded in
   `case_files.case_file_history`).

The juvenile subtree in §8.1 takes the regime tag as input and applies
the corresponding age-band caps, diversion catalog, and procedural
requirements.

A juvenile-specific subtree is invoked from `D5` (diversion) and `D6`
(sentence band) via the condition `age_determined < 18`. See §8.1.

## 7. Parallel track: non-criminal case types

ViLA handles civil, family, administrative, labor, and commercial
matters alongside criminal. They share a five-phase shape:

| Phase | Civil / commercial | Administrative |
|---|---|---|
| Entry | Nộp đơn khởi kiện; thụ lý | Nộp đơn khởi kiện hành chính |
| Prosecution / pretrial | Hòa giải (often mandatory); thu thập chứng cứ; chuẩn bị xét xử | Đối thoại; thu thập tài liệu |
| Adjudication | Xét xử sơ thẩm / phúc thẩm / giám đốc thẩm / tái thẩm | Xét xử sơ thẩm / phúc thẩm |
| "Sentencing" | Quyết định chia tài sản, bồi thường, công nhận / không công nhận quyền | Hủy / giữ nguyên quyết định hành chính |
| Corrections | Thi hành án dân sự | Thi hành án hành chính |

Sub-specific subtrees live under `decision_tree.yaml -> civil.*`,
`decision_tree.yaml -> admin.*`, etc. Their shape parallels the criminal
tree but swaps the sentencing node for the civil disposition node.

## 8. Decision tree for case-outcome prediction (agent-facing)

The tree is declared as YAML in
`services/agent/src/vila_agent/decision_tree.yaml` and documented here.
Each node is an agent step: a deterministic branch plus an optional tool
invocation. Node IDs are reused in `predictions.decision_path[]`.

```
root: predict_outcome
|
+- [D0] gatekeeping
|    input: case_file
|    branch:
|      - if case_type != "Hình sự" -> civil.root (parallel subtree, section 7)
|      - if incident_date missing  -> request_clarification
|      - else -> D1
|
+- [D1] legal_relation classification       (tool: classify_charge)
|    ensures case_files.legal_relation is set
|    branch:
|      - low_confidence       -> D1.a  (LLM w/ KG context)
|      - otherwise            -> D2
|
+- [D2] charge enumeration                  (tool: enumerate_charges)
|    one or more charges with articles       (statute linker)
|    branch:
|      - any charge lacks article  -> D2.a  (LLM + statute linker + KG)
|      - otherwise                  -> D3
|
+- [D3] precedent retrieval                 (tool: retrieve_precedents)
|    kNN over precedent_embeddings + scalar filter on charge family
|    branch:
|      - top-k w/ sim >= 0.82 -> D4
|      - otherwise            -> D4 with weak-evidence flag
|
+- [D4] similar-case retrieval              (tool: retrieve_similar_cases)
|    kNN over case_embeddings + scalar filter on (legal_relation, case_type)
|    plus 2-hop KG expand via services/kg
|
+- [D5] diversion / exit check              (tool: evaluate_diversion)
|    Consumes aggravating / mitigating factors + precedents + demographic
|    branch:
|      - age_determined < 18 + eligible offense -> juvenile.subtree
|      - mitigating >= threshold + first offense + cooperative
|          -> D9 emit EX-06 / EX-07 with high probability
|      - map to exit codes EX-01..EX-05 when conditions match (no-charge,
|        dismissal, refusal to indict, acquittal)
|      - otherwise -> D6
|
+- [D6] base sentence band                  (tool: estimate_sentence)
|    From statute's prescribed range (BLHS article range)
|    + precedent central tendency
|    + similar-case empirical distribution
|    Output: penalty_type + term range + confidence
|    Special path:
|      - if penalty_type == "Tử hình" -> mandatory senior-review flag
|
+- [D7] factor adjustment                   (tool: apply_factors)
|    Aggravating / mitigating factor catalog from BLHS Art 51/52
|    Output: adjusted term range
|
+- [D8] appeal likelihood                   (tool: estimate_appeal_likelihood)
|    From historical kháng cáo / kháng nghị rates at the court + charge family
|
+- [D9] formatting and provenance           (tool: render_prediction)
     Emit structured prediction (Phase 8 schema) with decision_path,
     evidence list, refusal flag.
```

### 8.1 Juvenile subtree (invoked from D5 when `age_determined < 18`)

```
juvenile.root
|
+- [J0] regime selection (reads incident_date; see section 6.3)
|    incident_date >= 2026-01-01  -> juvenile_regime = 'ltpctn-2024'
|    else                           -> juvenile_regime = 'pre-2026'
|    (favorable-to-accused transitional rule may upgrade pre-2026 to ltpctn-2024)
|
+- [J1] age-band classification (BLHS 2015 Art. 12)
|    under_14  -> J1.exit.no-responsibility            (out-of-system; EX-06)
|    14_16     -> liable only for specific serious offenses -> J2
|    16_18     -> generally liable with mitigation          -> J2
|
+- [J2] diversion evaluation (xử lý chuyển hướng)
|    Regime-specific eligibility table:
|      pre-2026   (BLTTHS 2015 Ch. XXVIII + BLHS 2015 + LXLVPHC 2012 measures)
|      ltpctn-2024 (LTPCTN-2024 unified framework; first-line response)
|    Candidate disposition set:
|      giáo dục tại xã / phường / thị trấn,
|      hòa giải tại cộng đồng,
|      biện pháp giáo dục tại trường giáo dưỡng,
|      plus additional LTPCTN-2024 measures for incidents >= 2026
|    If eligible -> emit EX-10 with probability and exit.
|
+- [J3] proceed to D6 with juvenile mitigation applied
|    Regime selects the caps table:
|      pre-2026   -> BLHS 2015 Part Four caps
|      ltpctn-2024 -> LTPCTN-2024 caps (tighter custodial eligibility)
|    Applied cap is min(statute_range_max, juvenile_cap).
|
+- [J4] record regime tag on the case
     Writes case_files.juvenile_regime so analytics can disaggregate.
```

### 8.2 Civil subtree (invoked from D0 when case_type is civil family)

```
civil.root
|
+- [V1] claim classification (legal_relation)
+- [V2] mediation outcome estimation        (tool: estimate_mediation_settlement)
|       If mediation probability >= threshold -> emit EX-08 (settled)
+- [V3] base disposition estimation         (tool: estimate_civil_disposition)
|       Split of contested property / compensation / relief
+- [V4] appeal likelihood
+- [V5] formatting and provenance
```

## 9. Decision tree node ID format

Every node ID the agent emits on `predictions.decision_path[]` uses the
two-namespace convention in this document:

- **Procedure-phase IDs** from section 3: `E1`..`E4`, `P1`..`P2`,
  `A1`..`A6`, `S1`, `C1`, with dotted suffixes for exits
  (`E3.exit.dismiss`) or variants (`S1.treo`, `C1.giam`).
- **Agent-step IDs** from section 8: `D0`..`D9`, with colon suffixes for
  outcomes (`D5:no-diversion`, `D6:tu-co-thoi-han`) and sub-tree
  identifiers (`J1`..`J3`, `V1`..`V5`).

Example path for a simple theft case convicted at sơ thẩm with
mitigating factors:

```
["D0", "D1", "D2", "D3", "D4", "D5:no-diversion",
 "D6:tu-co-thoi-han", "D7:MIT-03+MIT-04", "D8:appeal-prob-0.12", "D9",
 "E2", "E3", "E4", "P1", "P2", "A1", "A2", "A3.convict", "S1.tu", "C1.tu"]
```

The first list is the **agent decision path**; the second (after the
`"D9"` terminal) is the **procedural path** the agent asserts the case
will take.

## 10. Factor catalog for `apply_factors`

A compact, machine-readable factor code set sourced from BLHS 2015
Articles 51 (giảm nhẹ) and 52 (tăng nặng), kept in
`packages/nlp/data/factors.yaml`. Selected examples:

| Factor code | Kind | Vietnamese |
|---|---|---|
| `MIT-01` | mitigating | Người phạm tội đã tự nguyện khắc phục hậu quả |
| `MIT-02` | mitigating | Người phạm tội tự thú |
| `MIT-03` | mitigating | Người phạm tội thành khẩn khai báo, ăn năn hối cải |
| `MIT-04` | mitigating | Lần đầu phạm tội và thuộc trường hợp ít nghiêm trọng |
| `MIT-05` | mitigating | Phạm tội do bị người khác đe dọa hoặc cưỡng bức |
| `MIT-06` | mitigating | Người phạm tội là người dưới 18 tuổi |
| `AGG-01` | aggravating | Phạm tội có tổ chức |
| `AGG-02` | aggravating | Phạm tội có tính chất chuyên nghiệp |
| `AGG-03` | aggravating | Tái phạm hoặc tái phạm nguy hiểm |
| `AGG-04` | aggravating | Phạm tội đối với người dưới 16 tuổi, phụ nữ có thai, người già, ốm... |
| `AGG-05` | aggravating | Xúi giục người dưới 18 tuổi phạm tội |

Each factor has a default adjustment direction and magnitude fit on
historical data, not hard-coded.

## 11. Integration with the legal concept taxonomy

The decision tree operates across several independent `legal_type`
artifacts (siblings under `Tư pháp`, see `00-overview/glossary.md`),
not a strict containment chain. The mapping below shows which decision
nodes consume or produce which artifacts and their constituent
attributes.

| Decision node | Reads (legal_type or attribute) | Writes / emits |
|---|---|---|
| `D0` gatekeeping | `Vụ án.loại vụ án` (case_type), `Vụ án.Ngày gây án` (incident_date) | routes to criminal vs civil subtree |
| `D1` classify | `Cáo trạng` (if present) else `Đơn khởi kiện` / `Bản án` text; `Vụ án.Tóm tắt vụ việc` | `Vụ án.Quan hệ pháp luật` (legal_relation) |
| `D2` enumerate charges | `Cáo trạng.Tội danh`, `Cáo trạng.Căn cứ pháp luật` (statute articles); fallback to `Bản án` | list of `charge` entities with links to `statute_article` |
| `D3` precedent retrieval | charge vector; `Án lệ` embeddings | candidate `Án lệ` hits |
| `D4` similar cases | `Vụ án` embedding; KG 2-hop around current `Vụ án` | candidate similar `Vụ án` / `Bản án` hits |
| `D5` diversion / exit | `Cáo trạng.Danh sách bị can.age_determined`; `Đoán định vụ việc` factors; participant profile | candidate exit code (EX-01..EX-11) |
| `D6` base sentence band | BLHS `Điều luật` ranges; `Án lệ` central tendency; similar `Bản án.Mức hình phạt` distribution | `sentence_band` candidate |
| `D7` factor adjustment | `Đoán định vụ việc.Tình tiết tăng nặng / giảm nhẹ` | adjusted `sentence_band` |
| `D8` appeal likelihood | historical `Bản án` -> appealed-`Vụ án` rates at the court + charge family | `appeal_likelihood` |
| `D9` render | all of the above | `PredictionResponse` with `decision_path` + `provenance` |

Key observations about inputs:

- The agent prefers `Cáo trạng` when it exists (criminal matters
  post-truy tố). If only `Kết luận điều tra` is available (pre-cáo
  trạng criminal), the agent runs with lower confidence and marks the
  prediction `indicted=false`.
- For non-criminal matters, `Đơn khởi kiện` is the primary input;
  `Cáo trạng` is never populated.
- For appeal-stage cases, the agent can read both `Bản án (sơ thẩm)`
  and the appellate-stage `Vụ án` and reason about reversal likelihood.

Every leaf of the taxonomy that is actually populated for a given case
maps to either an input to a decision node, a tool invocation, or an
output field. Absent leaves are tolerated: the agent never assumes a
sibling `legal_type` is present just because a related one is.
