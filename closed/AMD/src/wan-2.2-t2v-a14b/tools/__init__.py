"""Out-of-band tools that are not part of the runtime harness.

Anything in here is meant to be invoked directly from the command line
(e.g. ``python -m tools.fetch_data``) and must depend only on the
standard library so it works in fresh clones before
``pip install -e .`` has been run.
"""
