#!/usr/bin/env python3
"""Main script to run all examples."""

import argparse
import subprocess
import sys
from pathlib import Path


def discover_scripts():
    """Find all Python scripts in Examples/ directory."""
    scripts = [
        p
        for p in Path("Examples").rglob("*.py")
        if p.is_file() and p.name != "__init__.py"
    ]

    compute_scripts = sorted([s for s in scripts if "compute" in s.name])
    plot_scripts = sorted([s for s in scripts if "plot" in s.name])

    return compute_scripts, plot_scripts


def run_scripts(scripts):
    """Run scripts sequentially and report results."""
    if not scripts:
        print("  No scripts to run")
        return

    print(f"\nRunning {len(scripts)} scripts...\n")

    success_count = 0
    fail_count = 0

    for script in scripts:
        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script)],
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode == 0:
                print(f"  ✓ {script.relative_to('Examples')}")
                success_count += 1
            else:
                print(
                    f"  ✗ {script.relative_to('Examples')} (exit {result.returncode})"
                )
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print(f"  ✗ {script.relative_to('Examples')} (timeout)")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ {script.relative_to('Examples')} ({e})")
            fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed\n")


def copy_figures():
    """Copy figures to report directory."""
    import shutil

    source = Path("figures")
    dest = Path("docs/reports/assignment_2_report/figures")

    print("\nCopying figures...")

    if not source.exists():
        print(f"  No figures found in {source}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(source, dest)
    print(f"  ✓ Copied {source} → {dest}\n")


def build_docs():
    """Build Sphinx documentation."""
    docs_dir = Path("docs")
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    print("\nBuilding Sphinx documentation...")

    if not source_dir.exists():
        print(f"  Error: Documentation source directory not found: {source_dir}")
        return False

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "sphinx-build",
                "-M",
                "html",
                str(source_dir),
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print("  ✓ Documentation built successfully")
            print(f"  → Open: {build_dir / 'html' / 'index.html'}\n")
            return True
        else:
            print(f"  ✗ Documentation build failed (exit {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ✗ Documentation build timed out")
        return False
    except FileNotFoundError:
        print("  ✗ sphinx-build not found. Install with: uv sync")
        return False
    except Exception as e:
        print(f"  ✗ Documentation build failed: {e}")
        return False


def build_latex_docs():
    """Build Sphinx LaTeX documentation and copy to report directory."""
    import shutil

    docs_dir = Path("docs")
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"
    latex_dir = build_dir / "latex"
    report_dir = Path("docs/reports/assignment_2_report")

    print("\nBuilding LaTeX documentation...")

    if not source_dir.exists():
        print(f"  Error: Documentation source directory not found: {source_dir}")
        return False

    # Step 1: Build LaTeX with Sphinx
    try:
        print("  Building LaTeX files with Sphinx...")

        # Create marker file so conf.py knows we're building LaTeX
        marker_file = source_dir / ".building_latex"
        marker_file.touch()

        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "sphinx-build",
                    "-b",
                    "latex",
                    str(source_dir),
                    str(latex_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"  ✗ Sphinx LaTeX build failed (exit {result.returncode})")
                if result.stderr:
                    print(f"    Error: {result.stderr[:500]}")
                return False

            print("  ✓ LaTeX files generated")

        finally:
            # Always remove marker file
            if marker_file.exists():
                marker_file.unlink()

    except subprocess.TimeoutExpired:
        print("  ✗ Sphinx LaTeX build timed out")
        return False
    except FileNotFoundError:
        print("  ✗ sphinx-build not found. Install with: uv sync")
        return False
    except Exception as e:
        print(f"  ✗ Sphinx LaTeX build failed: {e}")
        return False

    # Step 2: Extract content and copy Sphinx files to report directory
    try:
        sphinx_styles_dir = report_dir / "sphinx_styles"
        sphinx_styles_dir.mkdir(parents=True, exist_ok=True)

        # Find the generated LaTeX file (main document, not support files)
        # Look for the largest .tex file which should be the main document
        tex_files = [
            f for f in latex_dir.glob("*.tex") if not f.name.startswith("sphinx")
        ]
        if not tex_files:
            print(f"  ✗ No .tex file found in {latex_dir}")
            return False

        # Take the largest file (main document should be largest)
        source_tex = max(tex_files, key=lambda f: f.stat().st_size)

        # Read the source file and extract content only (skip preamble)
        print(f"  Extracting content from {source_tex.name}...")
        with open(source_tex, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find \begin{document} and extract content from there
        begin_doc_idx = None
        end_doc_idx = None
        for i, line in enumerate(lines):
            if "\\begin{document}" in line:
                begin_doc_idx = i
            if "\\end{document}" in line:
                end_doc_idx = i
                break

        if begin_doc_idx is None:
            print(f"  ✗ Could not find \\begin{{document}} in {source_tex.name}")
            return False

        # Extract content between \begin{document} and \end{document}
        # Skip the title and TOC pages (usually first ~15 lines after \begin{document})
        content_start = begin_doc_idx + 1
        # Look for first \chapter or \section
        for i in range(begin_doc_idx + 1, min(begin_doc_idx + 50, len(lines))):
            if "\\chapter{" in lines[i] or "\\section{" in lines[i]:
                content_start = i
                break

        content_lines = lines[content_start:end_doc_idx]

        # Write content-only file
        content_file = sphinx_styles_dir / "sphinx_content_only.tex"
        with open(content_file, "w", encoding="utf-8") as f:
            f.writelines(content_lines)

        print(f"  ✓ Extracted {len(content_lines)} lines to sphinx_content_only.tex")

        # Copy all Sphinx style files
        style_files = list(latex_dir.glob("sphinx*.sty")) + list(
            latex_dir.glob("sphinx*.cls")
        )
        for style_file in style_files:
            shutil.copy2(style_file, sphinx_styles_dir / style_file.name)

        print(f"  ✓ Copied {len(style_files)} Sphinx style files")
        print(f"  → Sphinx content available at: {content_file}")
        print(
            "  → Include in LaTeX with: \\input{sphinx_styles/sphinx_content_only.tex}\n"
        )
        return True

    except Exception as e:
        print(f"  ✗ Failed to copy Sphinx files: {e}")
        import traceback

        traceback.print_exc()
        return False


def clean_docs():
    """Clean built Sphinx documentation."""
    import shutil

    build_dir = Path("docs/build")

    print("\nCleaning Sphinx documentation...")

    if not build_dir.exists():
        print(f"  No build directory found at {build_dir}")
        return

    try:
        shutil.rmtree(build_dir)
        print(f"  ✓ Cleaned {build_dir}\n")
    except Exception as e:
        print(f"  ✗ Failed to clean documentation: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run example scripts and manage documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --compute                     Run data generation scripts
  python main.py --plot                        Run plotting scripts
  python main.py --copy                        Copy figures to report
  python main.py --build-docs                  Build Sphinx HTML documentation
  python main.py --build-latex                 Build LaTeX PDF and copy to report
  python main.py --clean-docs                  Clean built documentation
  python main.py --compute --plot --copy       Run everything
        """,
    )

    parser.add_argument(
        "--compute", action="store_true", help="Run data generation (compute) scripts"
    )
    parser.add_argument("--plot", action="store_true", help="Run plotting scripts")
    parser.add_argument("--copy", action="store_true", help="Copy figures to report")
    parser.add_argument(
        "--build-docs", action="store_true", help="Build Sphinx HTML documentation"
    )
    parser.add_argument(
        "--build-latex",
        action="store_true",
        help="Build LaTeX PDF and copy to report directory",
    )
    parser.add_argument(
        "--clean-docs", action="store_true", help="Clean built Sphinx documentation"
    )

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n Error: No arguments provided. Please specify at least one option.\n")
        sys.exit(1)

    args = parser.parse_args()

    # Handle documentation commands first (don't need script discovery)
    if args.clean_docs:
        clean_docs()

    if args.build_docs:
        build_docs()

    if args.build_latex:
        build_latex_docs()

    # Handle example scripts and figures
    if args.compute or args.plot:
        compute_scripts, plot_scripts = discover_scripts()
        print(
            f"\nFound {len(compute_scripts)} compute scripts and {len(plot_scripts)} plot scripts"
        )

        if args.compute:
            run_scripts(compute_scripts)
        if args.plot:
            run_scripts(plot_scripts)

    if args.copy:
        copy_figures()


if __name__ == "__main__":
    main()
