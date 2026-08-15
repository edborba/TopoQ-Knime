import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event, Lock

import knime.extension as knext
from topoq_category import topoq_category
import pandas as pd

KNIME_LOGGER = logging.getLogger(__name__)


MOPAC_MOLECULE_COLUMN = "Molecule (MOPAC)"

NUMERIC_COLUMNS = [
    "FINAL HEAT OF FORMATION (KCAL/MOL)",
    "COSMO AREA (SQUARE ANGSTROMS)",
    "COSMO VOLUME (CUBIC ANGSTROMS)",
    "GRADIENT NORM",
    "IONIZATION POTENTIAL (EV)",
    "HOMO (EV)",
    "LUMO (EV)",
]

REGEX_DICT = {
    "FINAL HEAT OF FORMATION (KCAL/MOL)": r"FINAL HEAT OF FORMATION = +([-0-9.]+)",
    "COSMO AREA (SQUARE ANGSTROMS)": r"COSMO AREA += +([0-9.]+) SQUARE ANGSTROMS",
    "COSMO VOLUME (CUBIC ANGSTROMS)": r"COSMO VOLUME += +([0-9.]+) CUBIC ANGSTROMS",
    "GRADIENT NORM": r"GRADIENT NORM += +([0-9.]+)",
    "IONIZATION POTENTIAL (EV)": r"IONIZATION POTENTIAL += +([-0-9.]+)",
    "homo_lumo": r"HOMO LUMO ENERGIES \(EV\) += *([-0-9.]+) +([-0-9.]+)",
    "alpha_homo_lumo": r"ALPHA SOMO LUMO \(EV\) += *([-0-9.]+) +([-0-9.]+)",
    "beta_homo_lumo": r"BETA  SOMO LUMO \(EV\) += *([-0-9.]+) +([-0-9.]+)",
}

# Closed-shell outputs report HOMO LUMO ENERGIES; open-shell outputs report ALPHA and
# BETA SOMO/LUMO pairs. The first key with a match wins, so the ALPHA pair takes
# precedence and BETA is only used when ALPHA is absent.
HOMO_LUMO_KEYS = ("homo_lumo", "alpha_homo_lumo", "beta_homo_lumo")

CANCEL_MESSAGE = "Execution cancelled by the user."

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED_FILENAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


class _ExecutionCancelled(RuntimeError):
    """Raised when the user cancels the node.

    Kept separate from RuntimeError so that per-molecule failures can be logged and
    skipped while cancellation always aborts the whole run.
    """


def _find_mopac_executable() -> str:
    executable = shutil.which("mopac")
    if executable:
        return executable

    windows_default = r"C:\mopac\bin\mopac.exe"
    if os.path.exists(windows_default):
        return windows_default

    raise RuntimeError(
        "MOPAC executable was not found. Configure the mopac.exe path in the node dialog."
    )


def _find_openbabel_executable(configured_path: str) -> str:
    if configured_path and os.path.exists(configured_path):
        return configured_path

    for executable_name in ("obabel", "babel"):
        executable = shutil.which(executable_name)
        if executable:
            return executable

    windows_default = r"C:\babel\OpenBabel-3.1.1\obabel.exe"
    if os.path.exists(windows_default):
        return windows_default

    raise RuntimeError(
        "OpenBabel executable was not found. Configure the obabel.exe path in the node dialog."
    )


def _split_pair(value: str) -> tuple[str, str]:
    parts = value.split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def _resolve_output_folder(exec_context, output_folder: str) -> str:
    if os.path.isabs(output_folder):
        return output_folder

    workflow_data_dir = exec_context.get_workflow_data_area_dir()
    return os.path.join(workflow_data_dir, output_folder)


def _replace_mopac_keywords(mop_file: str, keywords: str):
    with open(mop_file, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()

    text = text.replace("PUT KEYWORDS HERE", keywords)

    with open(mop_file, "w", encoding="utf-8") as file:
        file.write(text)


def _is_valid_filename_stem(value: str) -> bool:
    # Windows silently strips trailing spaces and dots from file names, so an ID that
    # ends with either would produce a file that no longer matches the ID.
    if not value or value != value.rstrip(" ."):
        return False
    if INVALID_FILENAME_CHARS.search(value):
        return False
    return value.split(".")[0].upper() not in RESERVED_FILENAMES


def _preview(values: list, limit: int = 10) -> str:
    shown = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        shown += f" (and {len(values) - limit} more)"
    return shown


def _validate_molecule_ids(ids: pd.Series):
    if ids.isna().any():
        raise ValueError("The ID column contains missing values.")

    as_str = ids.astype(str)
    duplicated = sorted(as_str[as_str.duplicated()].unique())
    if duplicated:
        raise ValueError(
            "The ID column contains duplicated values, which would make molecules "
            "overwrite each other's files: " + _preview(duplicated)
        )

    invalid = sorted(value for value in as_str.unique() if not _is_valid_filename_stem(value))
    if invalid:
        raise ValueError(
            "These ID values cannot be used as file names (empty, reserved on Windows, "
            'containing <>:"/\\|?* or control characters, or ending with a space or '
            "dot): " + _preview(invalid)
        )


def _resolve_homo_lumo_columns(results: pd.DataFrame) -> pd.DataFrame:
    def _resolve(row):
        for key in HOMO_LUMO_KEYS:
            if row.get(key, ""):
                return pd.Series(_split_pair(row[key]))
        return pd.Series(("", ""))

    results = results.copy()
    results[["HOMO (EV)", "LUMO (EV)"]] = results.apply(_resolve, axis=1)
    return results.drop(columns=list(HOMO_LUMO_KEYS))


def _build_results_table(ordered_ids: list[str], results: pd.DataFrame, id_col: str):
    """Build the output table: one row per input molecule, in input order.

    Molecules without a parsed result keep missing values in every result column.
    """
    output = pd.DataFrame({id_col: ordered_ids})

    if results.empty:
        output[MOPAC_MOLECULE_COLUMN] = None
        for col in NUMERIC_COLUMNS:
            output[col] = pd.NA
    else:
        results = _resolve_homo_lumo_columns(results)
        results = results.rename(
            columns={"ID": id_col, "Molecule": MOPAC_MOLECULE_COLUMN}
        )
        output = output.merge(results, on=id_col, how="left")

    for col in NUMERIC_COLUMNS:
        output[col] = pd.to_numeric(output[col], errors="coerce")

    # Left-merge fills missing molecules with NaN; normalise them to None so the
    # column holds only SDF strings or missing values. Assigning a Series with an
    # explicit object dtype keeps None as None instead of pandas re-inferring
    # StringDtype and converting it back to a missing value.
    molecule_col = output[MOPAC_MOLECULE_COLUMN]
    output[MOPAC_MOLECULE_COLUMN] = pd.Series(
        [
            value if isinstance(value, str) and value else None
            for value in molecule_col.astype(object)
        ],
        index=output.index,
        dtype=object,
    )
    return output


class _MopacBatchRunner:
    """Runs the three phases: OpenBabel conversion, MOPAC calculation, output parsing.

    A failure of a single molecule is logged and the molecule is skipped in every
    phase; its output row keeps missing values. Cancellation always aborts the run.

    Worker threads only touch the cancellation Event; every call into the KNIME
    execution context (progress, cancellation polling) happens on the main thread,
    since the context is not guaranteed to be thread-safe.
    """

    def __init__(
        self,
        exec_context,
        output_folder: str,
        mopac_executable: str,
        openbabel_executable: str,
        mopac_keywords: str,
        molecule_format: str,
        molecules_by_id: dict[str, str],
        max_workers: int,
    ):
        self._exec_context = exec_context
        self._output_folder = output_folder
        self._mopac_executable = mopac_executable
        self._openbabel_executable = openbabel_executable
        self._openbabel_parameters = ["-xH"]
        self._mopac_keywords = mopac_keywords
        self._molecule_format = molecule_format
        self._molecules_by_id = molecules_by_id
        self._max_workers = max_workers
        self._cancel_event = Event()
        self._process_lock = Lock()
        self._progress_lock = Lock()
        self._running_processes: dict[subprocess.Popen, str] = {}
        self._completed = 0
        self._parsed_rows: list[dict] = []

    def run(self) -> pd.DataFrame:
        molecule_ids = list(self._molecules_by_id)
        self._run_phase(
            self._prepare_mop_file, molecule_ids,
            "OpenBabel", "molecules converted", 0.0, 0.1,
        )
        self._run_phase(
            self._run_mopac, molecule_ids,
            "MOPAC", "calculations finished", 0.1, 0.7,
        )
        parseable = [
            molecule_id
            for molecule_id in molecule_ids
            if os.path.exists(self._path(molecule_id, ".out"))
        ]
        self._run_phase(
            self._parse_out_file, parseable,
            "Results", "molecules parsed", 0.8, 0.15,
        )
        return pd.DataFrame(self._parsed_rows)

    def _path(self, molecule_id: str, extension: str) -> str:
        return os.path.join(self._output_folder, molecule_id + extension)

    def _run_phase(
        self, worker, molecule_ids, phase_name, progress_noun,
        progress_start, progress_span,
    ):
        total = len(molecule_ids)
        self._completed = 0
        self._report(progress_start, progress_span, 0, total, phase_name, progress_noun)
        if not molecule_ids:
            return

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            try:
                pending = {
                    executor.submit(self._run_worker, worker, phase_name, molecule_id)
                    for molecule_id in molecule_ids
                }
                while pending:
                    if self._exec_context.is_canceled():
                        self._cancel_event.set()
                        for future in pending:
                            future.cancel()
                        self._stop_running_processes()
                        raise _ExecutionCancelled(CANCEL_MESSAGE)

                    done, pending = wait(pending, timeout=0.5)
                    with self._progress_lock:
                        completed = self._completed
                    self._report(
                        progress_start, progress_span, completed, total,
                        phase_name, progress_noun,
                    )
                    for future in done:
                        future.result()
            except BaseException:
                self._cancel_event.set()
                self._stop_running_processes()
                raise

    def _run_worker(self, worker, phase_name: str, molecule_id: str):
        if self._cancel_event.is_set():
            raise _ExecutionCancelled(CANCEL_MESSAGE)
        try:
            worker(molecule_id)
        except _ExecutionCancelled:
            raise
        except Exception as error:
            # One bad molecule must not abort the batch: log it and let the output
            # table carry missing values for this row.
            KNIME_LOGGER.warning(f"{phase_name} failed for {molecule_id}: {error}")
        with self._progress_lock:
            self._completed += 1

    def _report(self, start, span, completed, total, phase_name, noun):
        fraction = start + span * completed / total if total else start
        self._exec_context.set_progress(
            fraction, f"{phase_name}: {completed}/{total} {noun}"
        )

    def _stop_process(self, process: subprocess.Popen):
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _cleanup_interrupted_outputs(self, filename: str):
        stem = os.path.splitext(filename)[0]
        for extension in (".out", ".arc", ".end"):
            interrupted_file = os.path.join(self._output_folder, stem + extension)
            try:
                os.remove(interrupted_file)
                KNIME_LOGGER.warning(f"Partial file removed: {interrupted_file}.")
            except FileNotFoundError:
                pass

    def _stop_running_processes(self):
        with self._process_lock:
            processes = list(self._running_processes.items())

        for process, filename in processes:
            self._stop_process(process)
            self._cleanup_interrupted_outputs(filename)

    def _run_checked_process(self, command: list[str], filename: str):
        # stderr goes to a temporary file instead of PIPE: with PIPE, a process that
        # writes more than the pipe buffer would block before finishing, hanging the
        # polling loop forever.
        with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        ) as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=self._output_folder,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )

            with self._process_lock:
                self._running_processes[process] = filename

            try:
                # The interval grows from 10 ms up to 0.5 s: short calls (OpenBabel,
                # ~70 ms) do not pay a 0.5 s floor, and long MOPAC runs keep the
                # polling cost negligible without losing cancellation responsiveness.
                poll_interval = 0.01
                while process.poll() is None:
                    if self._cancel_event.is_set():
                        self._stop_process(process)
                        self._cleanup_interrupted_outputs(filename)
                        raise _ExecutionCancelled(CANCEL_MESSAGE)
                    time.sleep(poll_interval)
                    poll_interval = min(poll_interval * 2, 0.5)

                if process.returncode != 0:
                    stderr_file.seek(0)
                    stderr = stderr_file.read()
                    raise RuntimeError(
                        f"Process exited with code {process.returncode} for {filename}: "
                        f"{stderr.strip()}"
                    )
            finally:
                with self._process_lock:
                    self._running_processes.pop(process, None)

    def _prepare_mop_file(self, molecule_id: str):
        mop_path = self._path(molecule_id, ".mop")
        out_path = self._path(molecule_id, ".out")

        if os.path.exists(out_path):
            KNIME_LOGGER.warning(
                f"{out_path} already exists, conversion and calculation skipped."
            )
            return

        if os.path.exists(mop_path) and os.path.getsize(mop_path) > 0:
            KNIME_LOGGER.warning(f"{mop_path} already exists, conversion skipped.")
            return

        input_filename = f"{molecule_id}.{self._molecule_format}"
        input_path = os.path.join(self._output_folder, input_filename)
        with open(input_path, "w", encoding="utf-8") as input_file:
            input_file.write(self._molecules_by_id[molecule_id])

        try:
            os.remove(mop_path)
        except FileNotFoundError:
            pass

        mop_filename = f"{molecule_id}.mop"
        command = [
            self._openbabel_executable,
            f"-i{self._molecule_format}",
            input_filename,
            "-omop",
            "-O",
            mop_filename,
            *self._openbabel_parameters,
        ]
        try:
            self._run_checked_process(command, mop_filename)
            if not os.path.exists(mop_path) or os.path.getsize(mop_path) == 0:
                raise RuntimeError(
                    "OpenBabel finished without creating a valid MOPAC file "
                    f"{mop_filename}. Check the input format and molecule content."
                )
            _replace_mopac_keywords(mop_path, self._mopac_keywords)
        finally:
            try:
                os.remove(input_path)
            except FileNotFoundError:
                pass

    def _run_mopac(self, molecule_id: str):
        mop_path = self._path(molecule_id, ".mop")
        if not os.path.exists(mop_path):
            # Conversion failed for this molecule (already logged); nothing to run.
            return
        if os.path.exists(self._path(molecule_id, ".out")):
            return

        self._run_checked_process(
            [self._mopac_executable, f"{molecule_id}.mop"], f"{molecule_id}.mop"
        )

    def _parse_out_file(self, molecule_id: str):
        with open(
            self._path(molecule_id, ".out"), "r", encoding="utf-8", errors="replace"
        ) as file:
            text = file.read()

        row = {
            "ID": molecule_id,
            "Molecule": self._convert_out_to_sdf(molecule_id),
        }
        for key, regex in REGEX_DICT.items():
            match = re.search(regex, text)
            if not match:
                row[key] = ""
            elif key in HOMO_LUMO_KEYS:
                row[key] = f"{match.group(1)} {match.group(2)}"
            else:
                row[key] = match.group(1)

        self._parsed_rows.append(row)

    def _convert_out_to_sdf(self, molecule_id: str) -> str:
        out_filename = f"{molecule_id}.out"
        sdf_filename = f"{molecule_id}_from_out.sdf"
        sdf_path = os.path.join(self._output_folder, sdf_filename)

        try:
            os.remove(sdf_path)
        except FileNotFoundError:
            pass

        command = [
            self._openbabel_executable,
            "-iout",
            out_filename,
            "-osdf",
            "-O",
            sdf_filename,
        ]
        self._run_checked_process(command, sdf_filename)

        if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
            raise RuntimeError(
                "OpenBabel finished without creating a valid SDF file from "
                f"{out_filename}."
            )

        with open(sdf_path, "r", encoding="utf-8", errors="replace") as file:
            sdf_text = file.read()

        try:
            os.remove(sdf_path)
        except FileNotFoundError:
            pass

        return sdf_text


@knext.node(
    name="Batch Geometry Optimizer",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/topoq.png",
    category=topoq_category,
)
@knext.input_table(
    name="Molecules",
    description="Input table with an ID column and a molecule column in the selected input format.",
)
@knext.output_table(
    name="MOPAC Results",
    description="One row per input molecule: ID, the MOPAC-optimized structure, and the parsed properties.",
)
class MopacRunner2:
    """Run MOPAC for molecule input rows and parse selected output properties.

    The node writes one .mop file per input row, runs MOPAC in parallel, and reads the
    generated .out files. Existing .mop and .out files in the output folder are reused,
    so re-running the node only calculates what is missing. Enable "Delete saved files
    before running" to discard the saved files and recalculate everything.

    The output "Molecule (MOPAC)" column contains the raw OpenBabel conversion of the
    MOPAC .out file: The coordinates are optimized, but the bond table is reconstructed
    from interatomic distances and may be incorrect. Connect this output to the
    "Structure Correction" node together with the original molecule table, to reapply
    the optimized coordinates to the original bond table.
    """

    openbabel_executable = knext.StringParameter(
        "OpenBabel executable",
        "Path to obabel.exe.",
        "",
    )

    mopac_executable = knext.StringParameter(
        "MOPAC executable",
        "Path to mopac.exe. Leave empty to search the system PATH, or enter the full "
        "path to the executable.",
        "",
    )

    mopac_keywords = knext.StringParameter(
        "MOPAC keywords",
        "Keywords used in newly generated .mop files. Reused .mop files retain their "
        "original keywords.",
        "",
    )

    output_folder = knext.StringParameter(
        "Output folder",
        "Folder where .mop and .out files are written. Relative paths are resolved "
        "inside the current workflow data area.",
        "",
    )

    refresh_files = knext.BoolParameter(
        "Delete saved files before running",
        "When checked, the output folder is emptied before running and everything is "
        "recalculated. The output folder must be used exclusively by this node. When "
        "unchecked (default), existing .mop and .out files are reused and only missing "
        "results are calculated.",
        False,
    )

    max_workers = knext.IntParameter(
        "Parallel MOPAC processes",
        "Maximum number of MOPAC processes to run at the same time.",
        1,
        min_value=1,
    )

    id_column = knext.ColumnParameter(
        "ID column",
        "Column used as the molecule identifier and .mop file name stem. Values must "
        "be unique and usable as file names.",
        port_index=0,
    )

    molecule_column = knext.ColumnParameter(
        "Molecule column",
        "Column containing the molecule structure in the selected input format.",
        port_index=0,
    )

    input_molecule_format = knext.StringParameter(
        "Input molecule format",
        "Input format passed to OpenBabel for the Molecule column.",
        "sdf",
        enum=["sdf", "mol"],
    )

    def configure(self, config_context, input_table_schema):
        id_col = self.id_column
        mol_col = self.molecule_column

        if not id_col or not mol_col:
            # Columns not selected yet: keep the input schema unchanged.
            return input_table_schema

        input_cols = list(input_table_schema.column_names)
        missing = [col for col in (id_col, mol_col) if col not in input_cols]
        if missing:
            raise ValueError(
                "Selected columns are missing from the input table: " + ", ".join(missing)
            )

        try:
            import knime.types.chemistry as cet

            molecule_type = knext.logical(cet.SdfValue)
        except ImportError:
            KNIME_LOGGER.warning(
                "knime.types.chemistry is not available - the molecule column will be "
                "of type String. Install the KNIME Chemistry Types extension for full "
                "SDF support."
            )
            molecule_type = knext.string()

        # Preserve every input column except the original molecule column (which is
        # replaced by the computed Molecule (MOPAC) column), forcing the ID column to
        # string because the IDs are the .mop/.out file name stems. The .append() keeps
        # the preserved input columns and their types untouched.
        preserved_cols = [col for col in input_cols if col != mol_col]
        output_types = [
            knext.string() if col == id_col else input_table_schema[col].ktype
            for col in preserved_cols
        ]
        output_schema = knext.Schema(output_types, preserved_cols)

        extra = [
            (name, column_type)
            for name, column_type in (
                [(MOPAC_MOLECULE_COLUMN, molecule_type)]
                + [(col, knext.double()) for col in NUMERIC_COLUMNS]
            )
            if name not in preserved_cols
        ]
        if extra:
            output_schema = output_schema.append(
                knext.Schema(
                    [column_type for _, column_type in extra],
                    [name for name, _ in extra],
                )
            )
        return output_schema

    def execute(self, exec_context, input_table):
        id_col = self.id_column
        mol_col = self.molecule_column
        if not id_col or not mol_col:
            raise ValueError(
                "Select the 'ID column' and 'Molecule column' in the node dialog."
            )

        molecules = input_table.to_pandas()
        missing_columns = {id_col, mol_col} - set(molecules.columns)
        if missing_columns:
            raise ValueError(
                "Input table must contain these columns: "
                + ", ".join(sorted(missing_columns))
            )

        _validate_molecule_ids(molecules[id_col])

        if self.refresh_files and not self.output_folder.strip():
            raise ValueError(
                "'Delete saved files before running' would empty the whole workflow "
                "data area because 'Output folder' is empty. Configure a dedicated "
                "folder for this node."
            )

        output_folder = os.path.abspath(
            _resolve_output_folder(exec_context, self.output_folder)
        )
        os.makedirs(output_folder, exist_ok=True)

        if self.refresh_files:
            # The output folder is for this node's exclusive use: empty it to force a
            # full reconversion and recalculation.
            for entry in os.listdir(output_folder):
                entry_path = os.path.join(output_folder, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                else:
                    try:
                        os.remove(entry_path)
                    except FileNotFoundError:
                        pass

        ordered_ids: list[str] = []
        molecules_by_id: dict[str, str] = {}
        for _, row in molecules.iterrows():
            molecule_id = str(row[id_col])
            ordered_ids.append(molecule_id)
            if pd.isna(row[mol_col]):
                KNIME_LOGGER.warning(
                    f"Molecule column is empty for ID {molecule_id}; row skipped."
                )
                continue
            molecules_by_id[molecule_id] = str(row[mol_col])

        runner = _MopacBatchRunner(
            exec_context=exec_context,
            output_folder=output_folder,
            mopac_executable=self.mopac_executable.strip() or _find_mopac_executable(),
            openbabel_executable=_find_openbabel_executable(self.openbabel_executable),
            mopac_keywords=self.mopac_keywords,
            molecule_format=self.input_molecule_format,
            molecules_by_id=molecules_by_id,
            max_workers=self.max_workers,
        )
        results = runner.run()

        exec_context.set_progress(0.95, "Building the results table...")

        # Start from the full input table so every input column is carried through. The
        # original molecule column is dropped because it is replaced by the computed
        # Molecule (MOPAC) column. The ID column is delivered as string because the IDs
        # are the .mop/.out file name stems.
        output = molecules.copy()
        output[id_col] = output[id_col].astype(str)
        if mol_col in output.columns:
            output = output.drop(columns=[mol_col])

        # The results table has one row per input row, in input order, so the computed
        # columns can be assigned positionally. A computed column whose name already
        # exists in the input overwrites that slot instead of being duplicated.
        results_table = _build_results_table(ordered_ids, results, id_col)
        for col in results_table.columns:
            if col == id_col:
                continue
            output[col] = results_table[col].to_numpy()

        try:
            import knime.types.chemistry as cet

            output[MOPAC_MOLECULE_COLUMN] = output[MOPAC_MOLECULE_COLUMN].apply(
                lambda sdf: cet.SdfValue(sdf) if isinstance(sdf, str) and sdf else None
            )
        except ImportError:
            pass

        exec_context.set_progress(1.0, "Done.")
        return knext.Table.from_pandas(output)
