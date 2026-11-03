# GKG packages: shared content and import contract

> Applies to Galatea-Core v3.16.7. This is the user guide and public container contract for community tools and Link readers. Model Protocol V4, Checkpoint3 and the network are unchanged.

## Choosing a format

| Exported content | Container | Reader requirement |
|---|---|---|
| Existing models, runtime assets, optional semantic sources and announcement add-on pool | GKG4 | Existing GKG4 reader or new Core |
| The above plus any decks, replays, notes or presets | GKG5 | Core or Link reader supporting GKG5 |

The new Core reads both formats. Without shared attachments it still writes GKG4; existing GKG4 packages
do not need rebuilding. GKG5 extends the container, not the model. Retraining or replacing model UUIDs
is unnecessary. `V4` in the default `Galatea_V4_Pkg_<timestamp>.gkg` name identifies the model protocol;
read `manifest.json` to determine package compatibility.

Models, ONNX external data, `card_vocab.json` and compiled runtime assets retain their checks.
The optional knowledge-base/vector/index source trio remains inseparable. Packaging does not re-export
ONNX; use [ONNX maintenance in Resource Sync](code_semantics_en.md#onnx-engine-updates) first when required.

## Export and import in WebUI

1. Open Model Deployment and choose model files from one UUID pool
2. Expand Optional Shared Content, enable the required categories and read a local source directory. Select directories or individual files; overlapping selections are deduplicated
3. Include linked pages and images with documentation. Presets may use existing JSON, YAML, TOML and similar files; this release defines no preset parameter schema
4. Check the attachment count and GKG4/5 indicator, then generate the local package
5. Unpack into staging and select models, semantic maintenance components and shared files independently. Attachments start unselected; choose entire categories or individual files

Attachment-only import does not install models, vocabulary, runtime semantics or resumable sources.
After import, Recently Imported Notes and Presets opens the existing document viewer. UTF-8 Markdown
supports local links and images; other preset text is displayed verbatim. No training, Arena, league,
temperature or pool parameters are applied. Imported Markdown does not execute embedded HTML.
Open PDFs and larger attachments locally; text preview is limited to 4 MiB.

## Import locations

Core derives a stable group from the package name, model references and content records. Reimporting
identical bytes reuses existing files. Different existing content is reported as a conflict and is never
overwritten. Attachment failures remove files created by that import while preserving earlier imports.
These rules govern attachments; model, root-asset and attachment import is not one package-wide transaction.

| Category | Example archive path | Core destination and behavior |
|---|---|---|
| `decks` | `content/decks/Albaz/Build.ydk` | `decks/GKG_<group>__<original-directory-id>/Build.ydk`; separate physical pools for original directories preserve same-name builds |
| `replays` | `content/replays/holographic/Game1.json` | `ai_thoughts/GKG_<group>_<path-id>_<name>.json`, joining the existing replay list |
| `replays` | `content/replays/raw/link_captures/duel.core.jsonl.gz` | `replays/imported/<group>/link_captures/duel.core.jsonl.gz`, preserving original bytes |
| `documents` | `content/documents/README.md` | `deploy_packages/imported/<group>/documents/README.md`, preserving relative document directories |
| `presets` | `content/presets/deployment.yaml` | `deploy_packages/imported/<group>/presets/deployment.yaml`, stored and viewed only |

Deck import does not overwrite `global_weights.json` or `virtual_pools.json`. New physical pools follow
existing default weight rules and may enter subsequent random sampling; adjust them in Asset Management.
Imported YDK suffixes are normalized to lowercase for the existing loader; receipts retain original archive paths.
Existing deck preflight still determines CDB/vocabulary compatibility. Carrying deck files does not expand
model observations or independently teach new cards.

Storing raw recordings does not add parsers for every extension. The existing replay workflow still decides
whether a recording is supported and complete enough for conversion or imitation. Import neither starts
learning nor sends raw trajectories automatically into PPO or BC.

`deploy_packages/imported/<group>/contents.json` is generated automatically with model references,
original archive paths, actual destinations, sizes and hashes. No manual upkeep is needed. Removing an
unpacked staging folder does not remove installed files.

## GKG5 files and manifest

`.gkg` remains ZIP. Model/runtime files retain the flat GKG4 root layout; attachments alone enter `content/`:

```text
manifest.json
galatea_iter_200.onnx
galatea_iter_200.onnx.data
card_vocab.json
semantic_lookup_v1.json
semantic_lookup_v1.npz
content/decks/Albaz/Build.ydk
content/replays/holographic/Game1.json
content/replays/raw/duel.yrp3d
content/documents/README.md
content/documents/images/example.png
content/presets/deployment.yaml
```

The complete manifest retains existing model/protocol fields. Its additional fields are shown below.
The sample hash is for the 23 UTF-8 bytes `#main\n100\n#extra\n!side\n`:

```json
{
  "package_format_version": 5,
  "optional_content": {
    "format_version": 1,
    "files": [
      {
        "category": "decks",
        "path": "content/decks/Pool/A.ydk",
        "relative_path": "Pool/A.ydk",
        "size": 23,
        "sha256": "6969174311fe19974964da39551c0f1bdd8ff0d70ac639272f79cacda4fdc77d"
      }
    ]
  }
}
```

- Categories are `decks`, `replays`, `documents` and `presets`. `path` must begin with its matching `content/<category>/`; `relative_path` must equal the remaining suffix
- `size` is the uncompressed byte count; `sha256` is the 64-character lowercase digest of actual archived bytes. Readers verify each file; checksums are not publisher signatures
- GKG4 has no `optional_content`. GKG5 requires content format1 and a nonempty file list. Reject undeclared/missing attachments, wrong hashes and unknown content versions
- Unicode, spaces and relative directories are supported. Reject absolute/parent paths, backslashes, control characters, platform-reserved names, symbolic links, directory junctions and case/Unicode-equivalent path conflicts
- Model/root assets retain their flat filename rules. Other nested locations cannot masquerade as models or protocol assets
- At most 16,384 attachment files; 16,640 total GKG5 members versus 256 for GKG4. Expanded total is limited to 64 GiB, each member to 32 GiB. Manifest limits are 32 MiB for GKG5 and 2 MiB for GKG4; existing compression-ratio checks remain

Holographic replays use the `holographic/` prefix within `replays` and must be JSON. Prefer `raw/` for
original files; Core archives other replay-relative paths as raw files. Notes support Markdown, TXT,
PDF and common images. Presets support JSON, YAML, TOML, INI, CFG and TXT and are never executed as scripts.

## Core APIs and Link readers

APIs remain in [model_artifacts.py](../model_artifacts.py), with no new runtime module:

```python
from model_artifacts import create_deployment_package, install_deployment_content

create_deployment_package(
    "deploy_packages/Shared.gkg", "models", ["galatea_iter_200.onnx"],
    package_name="Shared",
    optional_files={
        "decks": {"Pool/A.ydk": "decks/Pool/A.ydk"},
        "documents": {"README.md": "model_docs/README.md"},
        "presets": {"deployment.yaml": "presets/deployment.yaml"},
    },
)

# After unpacking and validation, install selected attachments only
install_deployment_content(
    "deploy_packages/unpacked/Shared", ".",
    selected_paths=["content/documents/README.md"],
)
```

`discover_deployment_content` lists selectable files. `safe_extract_zip` provides bounded extraction
and `validate_deployment_package` verifies the full package. `plan_deployment_content_import` previews
destinations and existing-file states. Callers explicitly provide category → relative path → local source
through `optional_files`; the packager does not silently scan and include other directories.

Link can use this ZIP/JSON contract to display and selectively import attachments without running Core
training or semantic generation. Its reader must recognize GKG4/5, retain model identity checks and
verify content records/bytes; its own local destinations may differ. This release adds no Link preset
application or change to Link's real-server protocol.
