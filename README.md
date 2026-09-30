# avatar

A stylised 3D avatar creator. The character is generated in code from smooth
meshes and drawn with a cel-shading shader and inked outlines. Pick every
feature from a tabbed side panel, strike a pose, and export the result.

- **Face:** sculpted head (nose, cheekbones, jaw), eyes with a procedural iris,
  eyelids that blink, lips, six expressions, freckles, glasses, earrings.
- **Hair:** twelve styles built from thousands of real strands (grown from the
  scalp, clumped into locks, pushed out of the head and shoulders, with
  anisotropic highlights): short, curly, long, bob, bangs, bun, ponytail, quiff,
  mohawk, afro. Beards, moustaches and eyebrows are strands too, hugging the
  skin. Plus hats.
- **Body:** three body types, build and height, hands with fingers.
- **Outfit:** tee, long sleeve, tank, hoodie, jacket; jeans, shorts, skirt;
  sneakers, boots or barefoot; stripes, dots, plaid or a chest emblem.
- **Extras:** necklace, scarf, watch, backpack.
- **Poses:** relaxed, A-pose, wave, hands on hips, cheer, walk, plus idle
  animation (breathing sway, head movement, blinking).
- **Presets and sharing:** eight outfit presets, and a short share code that
  captures a whole avatar.

## Run

```bash
pip install -r requirements.txt
python avatar.py
```

Optional (`requirements-optional.txt`): OpenCV 4 for face detection in the photo
import (OpenCV 5 dropped the classifier it uses), and trimesh / scikit-image for
extra tests.

## Controls

| Input | Action |
| --- | --- |
| Click | pick a swatch or chip; click a tab; drag the scene to orbit |
| Tab / Shift+Tab | switch panel tab |
| ↑ / ↓ , ← / → | move the focus between options / change the focused option |
| Wheel | scroll the panel, or zoom when over the scene |
| H | show or hide the shortcut sheet |
| R | randomize everything |
| [ / ] | previous / next outfit preset |
| V | cycle camera: bust / face / full body |
| Drag / wheel | orbit / zoom |
| Space | toggle auto-spin |
| S / G | save PNG / save PNG with transparent background (to `exports/`) |
| E / O | export GLB (rigged parts, current pose) / OBJ + MTL |
| C / Ctrl+V | copy share code / load avatar from a code on the clipboard |
| P | pick a photo and set skin tone, hair colour and length, facial hair |
| K / L | save / load `avatar.json` |

## Interface

A floating card holds six tabs (Face, Hair, Body, Outfit, Extras, Scene). Colour
options are swatches, everything else is a row of chips; long tabs scroll. The
share code and the Random / Photo / PNG / GLB buttons sit at the bottom of the
card. A pill over the scene switches between bust, face and full-body views and
toggles the turntable, and short toasts confirm saves and exports. The window
can be resized; the 3D scene fills it and the avatar stays centred in the space
left of the card.

## Command line

```bash
python avatar.py --shot me.png --bare --view face --set hair=Long --set skin=Tan
python avatar.py --shot pose.png --view full --set pose=Wave --time 0.3
python avatar.py --shot logo.png --transparent --preset Streetwear
python avatar.py --sheet 12 --shot gallery.png --seed 42     # contact sheet of random avatars
python avatar.py --code AV2-101...                           # start from a share code
python avatar.py --photo selfie.jpg                          # suggest options from a photo
```

`--set OPTION=VALUE` takes the option keys and value names shown in the panel
(for example `hair=Mohawk`, `pattern=Stripes`, `pose=Hands on hips`).
`python avatar.py --help` lists every flag.

## Exports

- **GLB** keeps the skeleton as glTF nodes (head, arms, forearms, hands, legs...)
  so you can repose or animate it in Blender, Unity, Godot, and similar tools.
  Materials are plain colours; procedural extras (iris, freckles, cloth
  patterns) are render-only.
- **OBJ + MTL** bakes the current pose into one static mesh per part.
- **PNG** is the viewport only, optionally with a transparent background.

## Photo import

`P` (or `--photo`) reads skin tone, hair colour, hair length and facial hair
from a picture. It is a colour heuristic, not face recognition: with OpenCV
installed it locates the face first, otherwise it assumes a centred face. It
works best on an evenly lit, front-facing portrait. It corrects overall exposure
but not coloured lighting, and it won't detect eye colour, glasses or face shape.

## Layout

```text
avatar.py            entry point
avatarkit/
  options.py         option tables, presets, share codes, save/load
  mesh.py mathutil.py   mesh primitives (tube, loft, ellipsoid...) and math
  head.py hair.py    head, face details, hair layers, hats, facial hair
  strands.py         strand growth (physics-lite) and strand mesh builder
  body.py            torso, limbs, clothes, hands, accessories (as rig nodes)
  rig.py poses.py    skeleton and pose / animation functions
  builder.py         assembles an avatar from options
  render.py          GLSL cel shader, outline pass, camera, capture
  ui.py app.py       interface overlay (card, chips, toasts), input, command line
  exporters.py       OBJ and GLB writers
  photo.py           photo -> option suggestions
tests/               python -m unittest discover -s tests
```
