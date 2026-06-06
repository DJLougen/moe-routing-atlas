# Releasing MoE Routing Atlas

This project uses [Semantic Versioning](https://semver.org/):

| Bump | When |
|------|------|
| **PATCH** `0.2.x` | Bug fixes, security patches, no API changes |
| **MINOR** `0.x.0` | New features, backward-compatible changes |
| **MAJOR** `x.0.0` | Breaking changes |

## Release checklist

1. Update `CHANGELOG.md` under `[Unreleased]` → move entries to a new `## [x.y.z] - YYYY-MM-DD` section.
2. Bump version in **both** files:
   - `pyproject.toml` → `version = "x.y.z"`
   - `src/moe_routing_atlas/__version__.py` → `__version__ = "x.y.z"`
3. Commit: `git commit -m "chore: release vX.Y.Z"`
4. Tag and push:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin master
   git push origin vX.Y.Z
   ```
5. GitHub Actions (`.github/workflows/release.yml`) will:
   - Build sdist + wheel
   - Create a GitHub Release with changelog notes and attached artifacts

## Install a specific version

```bash
pip install git+https://github.com/DJLougen/moe-routing-atlas.git@v0.2.0
```

Or download wheel/sdist assets from the [Releases](https://github.com/DJLougen/moe-routing-atlas/releases) page.