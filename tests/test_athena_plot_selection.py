import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ATHENA_BACKEND = Path(__file__).resolve().parents[1] / "backends" / "athena"
sys.path.insert(0, str(ATHENA_BACKEND))

import plot_athena_kolmogorov_hd as plot_athena
import run_athena_kolmogorov_hd as run_athena


class AthenaSnapshotSelectionTest(unittest.TestCase):
    def test_selects_six_snapshots_across_full_time_span(self):
        paths = [Path(f"snapshot_{index:05d}.vtk") for index in range(21)]
        times = {path: index * 0.5 for index, path in enumerate(paths)}

        with mock.patch.object(plot_athena, "read_vtk_time", side_effect=lambda path: times[path]):
            selected = plot_athena.select_snapshot_paths(paths)

        self.assertEqual([times[path] for path in selected], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    def test_existing_data_must_reach_requested_end_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            analysis_dir = case_dir / "analysis"
            snapshot_dir = case_dir / "snapshots"
            analysis_dir.mkdir()
            snapshot_dir.mkdir()
            (analysis_dir / "case_metadata.toml").write_text("end_time = 10.0\n", encoding="utf-8")
            hst_path = snapshot_dir / "athena_kolmogorov.hst"
            vtk_path = snapshot_dir / "athena_kolmogorov.00000.vtk"
            cfg = SimpleNamespace(
                analysis_dir=analysis_dir,
                snapshot_dir=snapshot_dir,
                end_time=10.0,
                snapshot_dt=0.5,
            )

            hst_path.write_text("# [1]=time\n0.0\n5.0\n", encoding="utf-8")
            vtk_path.write_bytes(b"# vtk DataFile Version 3.0\ntime=5.0\nBINARY\n")
            self.assertFalse(run_athena.case_has_data(cfg))

            hst_path.write_text("# [1]=time\n0.0\n10.0\n", encoding="utf-8")
            vtk_path.write_bytes(b"# vtk DataFile Version 3.0\ntime=10.0\nBINARY\n")
            self.assertTrue(run_athena.case_has_data(cfg))


if __name__ == "__main__":
    unittest.main()
