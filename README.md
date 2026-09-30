# avatar

A stylised 3D avatar creator with an inked-outline look. The character is
generated in code from smooth meshes: a sculpted head (nose, brows, cheekbones,
jaw, ears), eyes with irises and eyelids, lips, ten hair styles, facial hair,
glasses and sunglasses, hats, and a full body with outfits (tee, long sleeve,
tank, hoodie, jacket; jeans, shorts, skirt; sneakers). Change any feature live
from the side panel.

## Run

```
pip install -r requirements.txt
python avatar.py
```

## Controls

| Input | Action |
| --- | --- |
| ↑ / ↓ | select an option |
| ← / → | change it (or click a row; left/right side of the value) |
| R | randomize the whole avatar |
| V | cycle camera: bust / face / full body |
| Drag / wheel | orbit / zoom |
| Space | toggle auto-spin |
| S | save a PNG of the viewport to `exports/` |
| E | export the avatar as OBJ + MTL to `exports/` |
| K / L | save / load `avatar.json` |

## Command line

Render a still without opening the editor workflow:

```
python avatar.py --shot me.png --view face --yaw 20 --set hair=Long --set skin=Tan
python avatar.py --shot random.png --seed 7
```

`--set OPTION=VALUE` uses the option keys and names from the panel
(`skin`, `face`, `nose`, `mouth`, `eyes`, `eyesize`, `brows`, `hair`,
`haircolor`, `facial`, `glasses`, `hat`, `hatcolor`, `build`, `top`, `topcolor`, `pants`,
`pantscolor`, `shoes`, `bg`).
