"""Spec / runner contract tests."""

from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from preframr_experiments.base import (
    Arm,
    ArmArtefacts,
    DATA_DIR,
    ExperimentSpec,
    _palette_extra_volumes,
    build_eval_reglogs_arg,
    canonical_paths,
    read_list,
    smoke_paths,
    stage_dumps,
)
from preframr_experiments.metrics import (
    METRIC_REGISTRY,
    compute_metrics,
    validate_metric_names,
)
from preframr_experiments.report import render


class TestExperimentSpecValidation(unittest.TestCase):
    def test_rejects_unknown_tier(self):
        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="x",
                doc="d",
                tier="bogus",
                arms=[Arm("a")],
                metrics=["alphabet_size"],
            )

    def test_rejects_no_arms(self):
        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="x", doc="d", tier="smoke", arms=[], metrics=["alphabet_size"]
            )

    def test_rejects_duplicate_arm_labels(self):
        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="x",
                doc="d",
                tier="smoke",
                arms=[Arm("a"), Arm("a")],
                metrics=["alphabet_size"],
            )

    def test_rejects_two_baseline_arms(self):
        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="x",
                doc="d",
                tier="smoke",
                arms=[Arm("a", baseline=True), Arm("b", baseline=True)],
                metrics=["alphabet_size"],
            )

    def test_smoke_default_seeds_is_one(self):
        s = ExperimentSpec(
            name="x", doc="d", tier="smoke", arms=[Arm("a")], metrics=["alphabet_size"]
        )
        self.assertEqual(s.seeds, 1)

    def test_canonical_default_seeds_is_three(self):
        s = ExperimentSpec(
            name="x",
            doc="d",
            tier="canonical",
            arms=[Arm("a")],
            metrics=["alphabet_size"],
        )
        self.assertEqual(s.seeds, 3)

    def test_block_stride_defaults_to_quarter_seq_len(self):
        s = ExperimentSpec(
            name="x",
            doc="d",
            tier="smoke",
            arms=[Arm("a")],
            metrics=["alphabet_size"],
            seq_len=4096,
        )
        self.assertEqual(s.block_stride, 1024)


class TestPinnedListsParse(unittest.TestCase):
    """Exercises the list-file parser against the actual pinned data
    -- catches a future change that breaks the comment-stripping or
    blank-line handling."""

    def test_smoke_list_parses(self):
        paths = smoke_paths()
        self.assertGreater(len(paths), 0)
        for p in paths:
            self.assertFalse(p.startswith("#"), f"comment leaked: {p!r}")
            self.assertTrue(p.endswith(".dump.parquet"), p)

    def test_canonical_lists_parse(self):
        layout = canonical_paths()
        for key in ("train", "eval-A", "eval-B-daglish", "eval-B-follin"):
            self.assertIn(key, layout)
            self.assertGreater(len(layout[key]), 0, key)
        self.assertGreater(len(layout["train"]), len(layout["eval-A"]))

    def test_read_list_strips_comments_and_blanks(self):
        with tempfile.NamedTemporaryFile("w", suffix=".list", delete=False) as f:
            f.write("# header line\n")
            f.write("\n")
            f.write("a/b/c.dump.parquet\n")
            f.write("d/e/f.dump.parquet  # composer note\n")
            f.write("   # leading space comment\n")
            tmp = f.name
        try:
            paths = read_list(Path(tmp))
            self.assertEqual(paths, ["a/b/c.dump.parquet", "d/e/f.dump.parquet"])
        finally:
            Path(tmp).unlink()


class TestMetricRegistry(unittest.TestCase):
    def test_validate_unknown_metric_raises(self):
        s = ExperimentSpec(
            name="x",
            doc="d",
            tier="smoke",
            arms=[Arm("a")],
            metrics=["alphabet_size", "this_does_not_exist"],
        )
        with self.assertRaises(ValueError):
            validate_metric_names(s)

    def test_validate_derived_metric_satisfies(self):
        s = ExperimentSpec(
            name="x",
            doc="d",
            tier="smoke",
            arms=[Arm("a")],
            metrics=["alphabet_size", "spec_specific_thing"],
            derived_metrics={"spec_specific_thing": lambda art: 42.0},
        )
        validate_metric_names(s)

    def test_registry_has_canonical_set(self):
        for name in (
            "alphabet_size",
            "encoded_tokens_per_song",
            "val_loss_best",
            "val_acc_at_best_loss",
            "epochs_to_best_val_loss",
            "wallclock_train_min",
        ):
            self.assertIn(name, METRIC_REGISTRY, name)


class TestReportRender(unittest.TestCase):
    """Exercises the markdown + JSON renderer against synthetic
    metric data so a future change to columns / formatting is caught
    here rather than at the end of a 6-hour canonical run."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.spec = ExperimentSpec(
            name="dummy",
            doc="A dummy experiment for testing the renderer.",
            tier="smoke",
            arms=[
                Arm("baseline", baseline=True),
                Arm("variant_a"),
                Arm("variant_b", training_overrides={"learning_rate": 5e-4}),
            ],
            metrics=["alphabet_size", "val_loss_best"],
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_renders_markdown_and_json(self):
        results = {
            "baseline": [{"alphabet_size": 1000, "val_loss_best": 0.50}],
            "variant_a": [{"alphabet_size": 800, "val_loss_best": 0.45}],
            "variant_b": [{"alphabet_size": 1200, "val_loss_best": 0.55}],
        }
        md = render(self.spec, results, self.tmpdir)
        self.assertTrue(md.exists())
        text = md.read_text()
        self.assertIn("# Experiment: dummy", text)
        for label in ("baseline", "variant_a", "variant_b"):
            self.assertIn(f"`{label}`", text)
        self.assertIn("Δ alphabet_size", text)
        self.assertIn("-200", text)
        self.assertIn("Training overrides", text)
        self.assertIn("variant_b", text.split("Training overrides")[1])

        data = json.loads((self.tmpdir / "report.json").read_text())
        self.assertEqual(data["name"], "dummy")
        labels = [a["label"] for a in data["arms"]]
        self.assertEqual(labels, ["baseline", "variant_a", "variant_b"])
        self.assertEqual(data["arms"][0]["metrics"]["alphabet_size"]["mean"], 1000)

    def test_aggregates_across_seeds(self):
        spec = ExperimentSpec(
            name="ms",
            doc="multi-seed",
            tier="smoke",
            arms=[Arm("a")],
            metrics=["alphabet_size"],
            seeds=2,
        )
        results = {"a": [{"alphabet_size": 100}, {"alphabet_size": 110}]}
        render(spec, results, self.tmpdir)
        text = (self.tmpdir / "report.md").read_text()
        self.assertIn("105", text)
        self.assertIn("±", text)


class TestComputeMetrics(unittest.TestCase):
    """compute_metrics() resolves spec-declared metrics through the
    registry + derived_metrics; failing extractor returns NaN with
    an _err sidecar so the report still renders."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.art = ArmArtefacts(
            arm=Arm("a"),
            seed=0,
            work_dir=self.tmpdir,
            tokens_csv=self.tmpdir / "tokens.csv",
            df_map_csv=self.tmpdir / "df-map.csv",
            tb_logs=self.tmpdir / "tb_logs",
            train_log=self.tmpdir / "train.log",
            parse_log=self.tmpdir / "parse.log",
            tokenize_log=self.tmpdir / "tokenize.log",
            metrics_json=self.tmpdir / "metrics.json",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_nan_when_artefact_missing(self):
        spec = ExperimentSpec(
            name="x",
            doc="d",
            tier="smoke",
            arms=[Arm("a")],
            metrics=["alphabet_size"],
        )
        out = compute_metrics(spec, self.art)
        self.assertTrue(math.isnan(out["alphabet_size"]))

    def test_derived_metric_runs(self):
        spec = ExperimentSpec(
            name="x",
            doc="d",
            tier="smoke",
            arms=[Arm("a")],
            metrics=["spec_x"],
            derived_metrics={"spec_x": lambda art: 7.0},
        )
        out = compute_metrics(spec, self.art)
        self.assertEqual(out["spec_x"], 7.0)

    def test_failing_extractor_does_not_kill_run(self):
        def boom(_art):
            raise RuntimeError("disk on fire")

        spec = ExperimentSpec(
            name="x",
            doc="d",
            tier="smoke",
            arms=[Arm("a")],
            metrics=["broken"],
            derived_metrics={"broken": boom},
        )
        out = compute_metrics(spec, self.art)
        self.assertTrue(math.isnan(out["broken"]))
        self.assertIn("disk on fire", out["_broken_error"])


class TestPhase2SpecsImport(unittest.TestCase):
    """The three Phase-2 specs (memorize / generalize / multi_composer)
    must all import cleanly and validate via ``validate_metric_names``.
    Catches a typo in a metric name or a forgotten import in any of
    them at unit-test time, before the build gate runs."""

    def test_memorize_spec_validates(self):
        from preframr_experiments.specs import memorize  # noqa: WPS433

        validate_metric_names(memorize.spec)
        self.assertEqual(memorize.spec.tier, "smoke")
        baselines = [a for a in memorize.spec.arms if a.baseline]
        self.assertEqual(len(baselines), 1)

    def test_generalize_spec_validates(self):
        from preframr_experiments.specs import generalize  # noqa: WPS433

        validate_metric_names(generalize.spec)
        self.assertEqual(generalize.spec.tier, "canonical")
        self.assertIsNotNone(generalize.spec.predict_gate)

    def test_prodlike_4x_goto80_in_train(self):
        train = read_list(DATA_DIR / "prodlike_4x" / "train.list")
        goto80 = [r for r in train if "/Goto80/" in r]
        self.assertGreater(len(goto80), 100)

    def test_prodlike_4x_holdout_composers_not_in_train(self):
        train = read_list(DATA_DIR / "prodlike_4x" / "train.list")
        holdout = (
            "Crisps",
            "Daglish_Ben",
            "Dobek_Eric",
            "Follin_Tim",
            "Marquis_Dave",
            "Mibri",
            "Wilson_Mark",
            "Winterberg_Michael",
        )
        for rel in train:
            for h in holdout:
                self.assertNotIn(f"/{h}/", rel, rel)


class TestRunCli(unittest.TestCase):
    """``run.py``'s ``load_spec`` must surface helpful errors on bad
    input (typo'd experiment names, modules without ``spec``)."""

    def test_load_spec_unknown_module_raises(self):
        from preframr_experiments.run import load_spec  # noqa: WPS433

        with self.assertRaises(SystemExit):
            load_spec("definitely_not_an_experiment_name")

    def test_load_spec_module_without_spec_raises(self):
        from preframr_experiments.run import load_spec  # noqa: WPS433

        with patch("preframr_experiments.run.importlib.import_module") as imp:

            class _NoSpec:
                pass

            imp.return_value = _NoSpec()
            with self.assertRaises(SystemExit):
                load_spec("placeholder")


class TestStageDumpsCollision(unittest.TestCase):
    """``stage_dumps`` puts each dump under a per-composer subdir so
    paths sharing a basename across composers don't overwrite each
    other (the prodlike train.list has 46 colliding basenames; before
    this layout, 50 of 4437 dumps got lost at stage time)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.src_root = self.tmp / "dumps"
        self.dst_dir = self.tmp / "staged"

    def _seed(self, rel: str, body: bytes) -> None:
        path = self.src_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def test_colliding_basenames_both_stage(self):
        """Two paths sharing a basename across different composer
        dirs both survive stage_dumps."""
        rels = [
            "MUSICIANS/A/Foo/Bar.1.dump.parquet",
            "MUSICIANS/B/Baz/Bar.1.dump.parquet",
        ]
        self._seed(rels[0], b"FROM_FOO")
        self._seed(rels[1], b"FROM_BAZ")

        logger = logging.getLogger("test")
        n = stage_dumps(rels, self.src_root, self.dst_dir, logger)
        self.assertEqual(n, 2)

        foo_dst = self.dst_dir / "Foo" / "Bar.1.dump.parquet"
        baz_dst = self.dst_dir / "Baz" / "Bar.1.dump.parquet"
        self.assertTrue(foo_dst.exists())
        self.assertTrue(baz_dst.exists())
        self.assertEqual(foo_dst.read_bytes(), b"FROM_FOO")
        self.assertEqual(baz_dst.read_bytes(), b"FROM_BAZ")

    def test_missing_files_skip_without_aborting(self):
        rels = [
            "MUSICIANS/A/Foo/Present.1.dump.parquet",
            "MUSICIANS/B/Baz/Absent.1.dump.parquet",
        ]
        self._seed(rels[0], b"PRESENT")

        logger = logging.getLogger("test")
        n = stage_dumps(rels, self.src_root, self.dst_dir, logger)
        self.assertEqual(n, 1)
        self.assertTrue((self.dst_dir / "Foo" / "Present.1.dump.parquet").exists())
        self.assertFalse((self.dst_dir / "Baz").exists())

    def test_bare_filename_falls_back_to_root_bucket(self):
        """Paths with no parent dir bucket under ``_root_`` so they
        still stage somewhere addressable rather than colliding at
        the dst_dir root."""
        rels = ["Bare.1.dump.parquet"]
        self._seed(rels[0], b"BARE")
        logger = logging.getLogger("test")
        n = stage_dumps(rels, self.src_root, self.dst_dir, logger)
        self.assertEqual(n, 1)
        self.assertTrue((self.dst_dir / "_root_" / "Bare.1.dump.parquet").exists())


class TestBuildEvalReglogsArg(unittest.TestCase):
    """Eval-reglogs string must carry the extra ``*`` segment so the
    glob picks up dumps from per-composer subdirs."""

    def test_legacy_single_eval_subdir(self):
        layout = {"train": [], "eval": []}
        got = build_eval_reglogs_arg(Path("/work"), layout)
        self.assertEqual(got, "/work/eval/*/*.dump.parquet")

    def test_multi_eval_subdirs(self):
        layout = {
            "train": [],
            "eval_a": [],
            "eval_b_daglish": [],
            "eval_b_follin": [],
        }
        got = build_eval_reglogs_arg(Path("/work"), layout)
        parts = got.split(";")
        self.assertEqual(len(parts), 3)
        for p in parts:
            name, glob = p.split("=", 1)
            self.assertTrue(glob.endswith("/*/*.dump.parquet"), p)
            self.assertEqual(glob, f"/work/{name}/*/*.dump.parquet")

    def test_no_eval_subdirs(self):
        self.assertEqual(build_eval_reglogs_arg(Path("/work"), {"train": []}), "")


class TestHvscVersionCheck(unittest.TestCase):
    """``preflight_check`` must abort when the HVSC tree's version
    differs from the tier's pin (or pass silently when the tree
    isn't on disk).
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_hvsc_tree(self, version: int) -> Path:
        root = self.tmpdir / "hvsc"
        (root / "DOCUMENTS").mkdir(parents=True)
        (root / "DOCUMENTS" / "HVSC.txt").write_text(
            "HIGH VOLTAGE SID COLLECTION\n\n"
            f"                              Release {version}\n\n"
        )
        return root

    def test_match_passes(self):
        from preframr_experiments.base import _hvsc_version_check

        hvsc = self._make_hvsc_tree(84)
        spec = ExperimentSpec(
            name="x",
            doc="d",
            tier="mini",
            arms=[Arm("a")],
            metrics=["alphabet_size"],
            hvsc_root=str(hvsc),
        )
        _hvsc_version_check(spec, logging.getLogger("test"))

    def test_mismatch_raises(self):
        from preframr_experiments.base import _hvsc_version_check

        hvsc = self._make_hvsc_tree(85)
        spec = ExperimentSpec(
            name="x",
            doc="d",
            tier="mini",
            arms=[Arm("a")],
            metrics=["alphabet_size"],
            hvsc_root=str(hvsc),
        )
        with self.assertRaises(RuntimeError) as ctx:
            _hvsc_version_check(spec, logging.getLogger("test"))
        self.assertIn("HVSC version mismatch", str(ctx.exception))

    def test_missing_tree_skips_silently(self):
        from preframr_experiments.base import _hvsc_version_check

        spec = ExperimentSpec(
            name="x",
            doc="d",
            tier="mini",
            arms=[Arm("a")],
            metrics=["alphabet_size"],
            hvsc_root=str(self.tmpdir / "does_not_exist"),
        )
        _hvsc_version_check(spec, logging.getLogger("test"))


class TestPaletteExtraVolumes(unittest.TestCase):
    """The runner must bind-mount the tier's canonical-palette JSON into
    the training container at the path
    ``_CANONICAL_PALETTES_CANDIDATES`` expects.
    """

    def test_existing_tier_yields_mount(self):
        src = DATA_DIR / "mini" / "engine_fp_palettes.json"
        if not src.exists():
            self.skipTest(f"missing fixture {src}")
        vols = _palette_extra_volumes("mini")
        self.assertEqual(len(vols), 1)
        host, container = vols[0]
        self.assertEqual(host, src.resolve())
        self.assertEqual(
            container, "/integration_tests/data/mini/engine_fp_palettes.json"
        )

    def test_missing_palette_returns_empty(self):
        if (DATA_DIR / "canonical" / "engine_fp_palettes.json").exists():
            self.skipTest("canonical palette unexpectedly present")
        self.assertEqual(_palette_extra_volumes("canonical"), [])


if __name__ == "__main__":
    unittest.main()
