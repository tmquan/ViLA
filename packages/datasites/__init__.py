"""Per-site Curator pipeline factories.

Each site is a subpackage holding the four Curator download primitives
plus a :class:`nemo_curator.pipeline.Pipeline` factory:

    packages/datasites/<site>/
        url_generation.py  # nemo_curator.stages.text.download.base.URLGenerator
        download.py        # nemo_curator.stages.text.download.base.DocumentDownloader
        iterator.py        # nemo_curator.stages.text.download.base.DocumentIterator
        extract.py         # nemo_curator.stages.text.download.base.DocumentExtractor
        pipeline.py        # build_<site>_pipeline(cfg) -> nemo_curator.pipeline.Pipeline
        __main__.py        # CLI wrapper: load cfg, build pipeline, run executor
        configs/
          default.yaml
          <site>.yaml
        README.md
        requirements.txt

Stages 2-5 (parser / extractor / embedder / reducer) are site-agnostic
:class:`ProcessingStage` subclasses under :mod:`packages.parser`,
:mod:`packages.extractor`, :mod:`packages.embedder`,
:mod:`packages.reducer`. The pipeline factory wires a site's four
download primitives into Curator's
:class:`DocumentDownloadExtractStage` composite and chains the
site-agnostic stages onto the end.

Operators run a site's pipeline in one of two styles (see each site's
README):

    # Distributed CLI (vbpl + the Family-B harvesters):
    python -m packages.datasites.vbpl --pipeline all --executor xenna
    python -m packages.datasites.vbpl --pipeline embed \\
        --executor ray_actor_pool --ray-address ray://<head>:10001

    # Single-IP in-process drivers on the GB10 (anle, congbobanan):
    python -m packages.datasites.anle.pipeline --records-out anle_records.jsonl
    python -m packages.datasites.congbobanan.extract_text --shard 0 --nshards 1

Currently shipping datasites:

* :mod:`packages.datasites.anle` -- anle.toaan.gov.vn (Vietnamese án
  lệ / precedents). ~80 documents; the reference integration.
* :mod:`packages.datasites.congbobanan` -- congbobanan.toaan.gov.vn
  (bản án / quyết định; the full VN judgment portal). Integer-ID
  enumeration over ~2.1 M cases; requires Vietnamese egress IP.
"""
