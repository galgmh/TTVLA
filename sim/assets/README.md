# sim/assets

In-repo assets for the task configs. The sim reads them **at run time**:
`sim/scene.py::resolve_mesh_path` / `resolve_urdf_path` try the configured path
verbatim, then relative to this directory, then the Genesis assets directory.
`mesh_file` / `urdf` in `sim/configs/*.yaml` are therefore written relative to
`sim/assets/` (e.g. `robosuite/models/assets/objects/meshes/can.obj`), and a
checkout is self-contained.

Only the files a shipped config actually loads are vendored — the source trees
contain many more objects. OBJ + MTL + the texture image(s) sit together so the
engine's texture lookup (and the trimesh fix in `sim/compat/mesh_load_compat.py`,
docs §9-R41) works unchanged.

```
assets/
├── urdf/
│   └── panda_bullet/           Franka Panda: panda.urdf + meshes/ + LICENSE.txt
└── robosuite/                  used by sim/configs/robosuite_tabletop.yaml
    └── models/assets/
        ├── objects/meshes/     can.obj, lemon.obj, milk.obj (+ .mtl)
        └── textures/           soda.png, lemon.png, ceramic.png
```

The robosuite tree keeps upstream's `models/assets/...` layout because its MTLs
reference textures as `../../textures/<name>.png`; flattening it would silently
drop the textures (the failure mode behind docs §9-R41).

LIBERO assets are deliberately **not** vendored: the LIBERO tabletop scene was
removed on 2026-10-18, and no remaining config references that library.

## Provenance

| Subtree | Source on this machine | Upstream |
|---|---|---|
| `urdf/panda_bullet/` | `<envs>/tempvla/lib/python3.11/site-packages/genesis/assets/urdf/panda_bullet/` | Genesis 1.4.1 (PyBullet Panda, `LICENSE.txt` included) |
| `robosuite/` | `~/robosuite/robosuite/models/assets/` | robosuite 1.5.2 (MIT) |

These files are data, not code: they are unmodified copies of the upstream
assets. Check the upstream licenses before redistributing a dataset or the repo
publicly.
