"""Run a notebook reproducibly, saving the executed copy under local results."""

from config.paths import load_paths


def execute_notebook(source, timeout=3600):
    import sys
    from jupyter_client import KernelManager
    import nbformat
    from nbclient import NotebookClient

    paths = load_paths()
    source = paths.resolve(source)
    from engine.runner import _new_directory

    output = _new_directory(paths, "notebooks") / source.name
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    NotebookClient(
        notebook,
        km=manager,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(paths.root)}},
    ).execute(cleanup_kc=True)
    nbformat.write(notebook, output)
    return output
