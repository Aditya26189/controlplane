"""Test package.

An ``__init__.py`` so ``tests`` is a package and shared helpers can be imported
as ``from .factories import ...`` rather than relying on pytest's sys.path
insertion, which depends on the invocation directory.
"""
